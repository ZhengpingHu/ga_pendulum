#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import multiprocessing as mp
from multiprocessing.connection import Client
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import time

# 绘图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("OMP_NUM_THREADS", "1")
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# ==========================================
# 1. 基础组件与可复现性
# ==========================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"🔒 Global Seed set to: {seed}")

class PendulumNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 4输入 -> 16隐藏 -> 1输出(Tanh) -> 放大2倍
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * 2.0

# 基因操作
def get_weights_vector(m): return torch.cat([p.data.flatten() for p in m.parameters()]).cpu().numpy()
def set_weights_vector(m, vec):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n
def mutate(vec, sigma): return vec + np.random.randn(vec.size) * sigma
def uniform_crossover(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(~mask, p1, p2)

# ==========================================
# 2. 种子库管理
# ==========================================
class SeedPortfolioManager:
    def __init__(self, pool_size, subset_k):
        self.master_pool = [2025 + i for i in range(pool_size)]
        self.subset_k = subset_k
        self.active_subset = random.sample(self.master_pool, self.subset_k)
        print(f"🌱 Initial subset: {self.active_subset}")

    def get_active_subset(self): return self.active_subset

    def update_and_refresh(self, results_matrix: np.ndarray, refresh_rate: float):
        # 计算每个种子的总得分 (Column Sum)
        seed_scores = results_matrix.sum(axis=0)
        sorted_indices = np.argsort(seed_scores) # 低分在前
        
        num_replace = int(self.subset_k * refresh_rate)
        if num_replace == 0: return

        # 替换最难的 (分数最低的)
        indices_to_replace = sorted_indices[:num_replace]
        available = [s for s in self.master_pool if s not in self.active_subset]
        
        if len(available) < num_replace:
            new_seeds = random.sample(self.master_pool, num_replace)
        else:
            new_seeds = random.sample(available, num_replace)
            
        for i, idx in enumerate(indices_to_replace):
            self.active_subset[idx] = new_seeds[i]

    # 用于保存 Checkpoint
    def state_dict(self):
        return {"master_pool": self.master_pool, "active_subset": self.active_subset}

# ==========================================
# 3. 评估与通信
# ==========================================
class RPCClient:
    def __init__(self, port, authkey=b"pendulum-rpc"):
        self.conn = Client(('127.0.0.1', port), authkey=authkey)
    def reset(self): self.conn.send(("reset", None)); self.conn.recv()
    def infer(self, frame):
        self.conn.send(("infer", frame)); ok, res = self.conn.recv()
        return res if ok else None
    def close(self): self.conn.close()

def evaluate_individual(args):
    pop_idx, seed_idx, weights, seed, rpc_port, max_steps = args
    model = PendulumNNPolicy()
    set_weights_vector(model, weights)
    total_reward = 0.0
    
    try:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        obs, _ = env.reset(seed=int(seed))
        last_state = np.zeros(4, dtype=np.float32)
        
        rpc = RPCClient(rpc_port)
        rpc.reset()
        
        for _ in range(max_steps):
            frame = env.render()
            state = rpc.infer(frame[..., ::-1].copy()) # RGB->BGR
            if state is None: state = last_state
            else: last_state = state
            
            s_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad(): action = model(s_tensor).item()
            
            _, reward, done, truncated, _ = env.step([action])
            total_reward += reward
            if done or truncated: break
            
        rpc.close(); env.close()
        return pop_idx, seed_idx, total_reward
    except Exception:
        return pop_idx, seed_idx, -2000.0

def calculate_fitness_sharing(results_matrix: np.ndarray) -> np.ndarray:
    min_val = np.min(results_matrix)
    shifted = results_matrix - min_val + 1e-5
    seed_totals = shifted.sum(axis=0)
    seed_totals[seed_totals == 0] = 1e-9
    shared_matrix = shifted / seed_totals
    return shared_matrix.sum(axis=1)

# ==========================================
# 4. 日志与 Checkpoint (核心修改)
# ==========================================
def save_detailed_logs(run_dir, gen, results_matrix, fitness_scores, subset_seeds):
    """
    保存这一代每个个体、在每个种子下的详细表现。
    """
    records = []
    pop_size, num_seeds = results_matrix.shape
    
    for i in range(pop_size):
        for j in range(num_seeds):
            records.append({
                "generation": gen,
                "individual_id": i,
                "seed_value": subset_seeds[j],
                "reward": results_matrix[i, j],
                "shared_fitness": fitness_scores[i] # 共享Fitness是个体的属性
            })
            
    df = pd.DataFrame(records)
    # 追加模式写入 (如果文件不存在则创建头)
    log_path = os.path.join(run_dir, "detailed_history.csv")
    write_header = not os.path.exists(log_path)
    df.to_csv(log_path, mode='a', header=write_header, index=False)

def save_checkpoint(run_dir, gen, population, portfolio):
    """
    保存完整的训练现场 (种群权重 + 种子库状态)
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    filename = os.path.join(ckpt_dir, f"checkpoint_gen_{gen:04d}.npz")
    np.savez_compressed(
        filename,
        generation=gen,
        population=np.array(population), # 保存所有个体的权重
        portfolio_state=portfolio.state_dict() # 保存种子库状态
    )
    # print(f"💾 Checkpoint saved: {filename}")

def plot_dual_axis(run_dir, df):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    c1 = 'tab:purple'; c2 = 'tab:green'
    
    ax1.set_xlabel('Gen')
    ax1.set_ylabel('Shared Fitness', color=c1)
    ax1.plot(df['gen'], df['best_fitness'], color=c1, label='Best Fitness')
    ax1.tick_params(axis='y', labelcolor=c1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Raw Reward', color=c2)
    ax2.plot(df['gen'], df['max_raw_reward'], color=c2, linestyle='--', label='Max Reward')
    ax2.plot(df['gen'], df['avg_raw_reward'], color='gray', linestyle=':', label='Avg Reward')
    ax2.tick_params(axis='y', labelcolor=c2)
    
    plt.title("Evolution Process")
    fig.tight_layout()
    plt.savefig(os.path.join(run_dir, "plot_metrics.png")); plt.close()

def plot_final_violin(run_dir, final_matrix):
    avg_scores = final_matrix.mean(axis=1)
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=avg_scores, inner="quartile", color="skyblue")
    plt.title("Final Robustness (All Seeds)")
    plt.ylabel("Avg Reward")
    plt.grid(True, axis='y', alpha=0.5)
    plt.savefig(os.path.join(run_dir, "plot_final_violin.png")); plt.close()

# ==========================================
# 5. 主程序
# ==========================================
def run_ga(args):
    ts = int(time.time())
    run_dir = f"runs_pendulum/ga_{ts}"
    os.makedirs(run_dir, exist_ok=True)
    
    # 初始化
    portfolio = SeedPortfolioManager(pool_size=100, subset_k=args.subset_k)
    model = PendulumNNPolicy()
    base_vec = get_weights_vector(model)
    pop = [mutate(base_vec, args.sigma) for _ in range(args.population)]
    
    history_metrics = []
    best_global_raw = -np.inf
    
    print(f"🚀 Training Started. Logs: {run_dir}")
    print(f"   Detailed logs: {run_dir}/detailed_history.csv")
    print(f"   Checkpoints:   {run_dir}/checkpoints/")

    for gen in range(1, args.generations + 1):
        subset = portfolio.get_active_subset()
        
        # 准备任务
        jobs = []
        for i in range(args.population):
            for j, seed in enumerate(subset):
                jobs.append((i, j, pop[i], seed, args.rpc_port, args.max_steps))
        
        # 并行评估
        results_matrix = np.zeros((args.population, args.subset_k))
        with mp.Pool(args.processes) as pool:
            for p_idx, s_idx, r in tqdm(pool.imap_unordered(evaluate_individual, jobs), 
                                       total=len(jobs), desc=f"Gen {gen}", leave=False):
                results_matrix[p_idx, s_idx] = r
        
        # 计算指标
        raw_avgs = results_matrix.mean(axis=1)
        gen_max = np.max(raw_avgs)
        gen_avg = np.mean(raw_avgs)
        fit_scores = calculate_fitness_sharing(results_matrix)
        best_fit = np.max(fit_scores)
        
        print(f"🏆 Gen {gen} | Fit: {best_fit:.2f} | MaxRaw: {gen_max:.1f} | Avg: {gen_avg:.1f}")
        
        # --- LOGGING & CHECKPOINTING ---
        # 1. 保存详细数据 (CSV)
        save_detailed_logs(run_dir, gen, results_matrix, fit_scores, subset)
        
        # 2. 保存指标概览 (CSV + Plot)
        history_metrics.append({
            "gen": gen, "best_fitness": best_fit, 
            "max_raw_reward": gen_max, "avg_raw_reward": gen_avg
        })
        if gen % 1 == 0:
            df = pd.DataFrame(history_metrics)
            df.to_csv(os.path.join(run_dir, "metrics_summary.csv"), index=False)
            plot_dual_axis(run_dir, df)
            
        # 3. 定期保存完整 Checkpoint (每5代)
        if gen % 5 == 0:
            save_checkpoint(run_dir, gen, pop, portfolio)
            
        # 4. 保存最佳模型 (Best so far)
        if gen_max > best_global_raw:
            best_global_raw = gen_max
            np.savez(os.path.join(run_dir, "best_model_running.npz"), weights=pop[np.argmax(fit_scores)])

        # --- 进化 ---
        portfolio.update_and_refresh(results_matrix, args.seed_refresh_rate)
        
        sorted_idx = np.argsort(fit_scores)
        elite_cnt = max(2, int(args.population * 0.2))
        elites = [pop[i] for i in sorted_idx[-elite_cnt:]]
        
        new_pop = elites.copy()
        while len(new_pop) < args.population:
            p1, p2 = random.sample(elites, 2)
            c1, c2 = uniform_crossover(p1, p2)
            new_pop.append(mutate(c1, args.sigma))
            if len(new_pop) < args.population:
                new_pop.append(mutate(c2, args.sigma))
        pop = new_pop

    # --- 终极全量考核 ---
    print("\n🏁 Final Robustness Check (All 100 Seeds)...")
    final_jobs = []
    for i in range(args.population):
        for j, seed in enumerate(portfolio.master_pool):
            final_jobs.append((i, j, pop[i], seed, args.rpc_port, args.max_steps))
            
    final_matrix = np.zeros((args.population, len(portfolio.master_pool)))
    with mp.Pool(args.processes) as pool:
        for p_idx, s_idx, r in tqdm(pool.imap_unordered(evaluate_individual, final_jobs), 
                                   total=len(final_jobs)):
            final_matrix[p_idx, s_idx] = r
            
    final_avgs = final_matrix.mean(axis=1)
    best_idx = np.argmax(final_avgs)
    
    print(f"🌟 Best Agent Score: {final_avgs[best_idx]:.2f}")
    np.savez(os.path.join(run_dir, "best_model_final.npz"), weights=pop[best_idx])
    plot_final_violin(run_dir, final_matrix)
    print(f"💾 Done. Results in {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-port", type=int, default=6000)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--subset-k", type=int, default=5)
    parser.add_argument("--seed-refresh-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    
    args = parser.parse_args()
    
    # 核心：设置全局种子
    set_global_seed(args.seed)
    
    run_ga(args)