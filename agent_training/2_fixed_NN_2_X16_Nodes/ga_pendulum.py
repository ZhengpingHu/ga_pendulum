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

# 绘图库
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("OMP_NUM_THREADS", "1")
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# ==========================================
# 1. 基础组件
# ==========================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"🔒 Global Seed set to: {seed}")

class PendulumNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 4 -> Output: 1
        self.net = nn.Sequential(
            nn.Linear(4, 1),
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
# 2. 种子库管理 (保留修复后的逻辑)
# ==========================================
class SeedPortfolioManager:
    def __init__(self, pool_size, subset_k, max_age=10):
        self.master_pool = [2025 + i for i in range(pool_size)]
        self.subset_k = subset_k
        self.max_age = max_age
        self.active_subset = random.sample(self.master_pool, self.subset_k)
        self.active_ages = {seed: 0 for seed in self.active_subset}
        print(f"🌱 Initial subset: {self.active_subset} (Max Age: {self.max_age})")

    def get_active_subset(self): return self.active_subset

    def update_and_refresh(self, results_matrix: np.ndarray, refresh_rate: float):
        for s in self.active_subset:
            self.active_ages[s] = self.active_ages.get(s, 0) + 1
            
        seeds_to_retire = {s for s, age in self.active_ages.items() if age >= self.max_age}
        
        seed_scores = results_matrix.sum(axis=0)
        sorted_indices = np.argsort(seed_scores)
        num_perf_replace = int(self.subset_k * refresh_rate)
        indices_perf = sorted_indices[:num_perf_replace]
        seeds_perf_replace = {self.active_subset[i] for i in indices_perf}
        
        seeds_to_remove = seeds_to_retire.union(seeds_perf_replace)
        if not seeds_to_remove: return

        num_replace = len(seeds_to_remove)
        available = [s for s in self.master_pool if s not in self.active_subset]
        
        if len(available) < num_replace:
            new_seeds = random.sample(self.master_pool, num_replace)
        else:
            new_seeds = random.sample(available, num_replace)
            
        for s in seeds_to_remove:
            if s in self.active_ages: del self.active_ages[s]
                
        current_seeds = [s for s in self.active_subset if s not in seeds_to_remove]
        for new_s in new_seeds:
            current_seeds.append(new_s)
            self.active_ages[new_s] = 0
            
        self.active_subset = current_seeds[:self.subset_k]
        while len(self.active_subset) < self.subset_k:
            extra = random.choice(self.master_pool)
            if extra not in self.active_subset:
                self.active_subset.append(extra)
                self.active_ages[extra] = 0

    def state_dict(self):
        return {"master_pool": self.master_pool, "active_subset": self.active_subset, "active_ages": self.active_ages}

# ==========================================
# 3. 评估逻辑 (双轨奖励)
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
    """
    返回: (pop_idx, seed_idx, raw_env_reward, shaped_fitness_reward)
    """
    pop_idx, seed_idx, weights, seed, rpc_port, max_steps = args
    model = PendulumNNPolicy()
    set_weights_vector(model, weights)
    
    total_raw_reward = 0.0
    total_shaped_reward = 0.0
    
    try:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        obs, _ = env.reset(seed=int(seed))
        last_state = np.zeros(4, dtype=np.float32)
        
        rpc = RPCClient(rpc_port)
        rpc.reset()
        
        for _ in range(max_steps):
            frame = env.render()
            state = rpc.infer(frame[..., ::-1].copy())
            if state is None: state = last_state
            else: last_state = state
            
            s_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad(): action = model(s_tensor).item()
            
            # --- 执行动作 ---
            _, reward, done, truncated, _ = env.step([action])
            
            # 1. 记录真实环境奖励 (Raw)
            total_raw_reward += reward
            
            # 2. 计算引导奖励 (Shaped) - 仅用于进化
            # state[1] 是 normalized y. 范围约 [-1, 1]. -1 是最高点。
            tip_y = state[1]
            bonus = 0.0
            
            # [关键修改] 奖励诱导逻辑
            # 如果杆子直立 (y < -0.8)，给予强烈暗示
            if tip_y < -0.8:
                bonus = 1.0 # 只要保持直立，每帧+1分 (这在Pendulum里是巨款)
            
            total_shaped_reward += (reward + bonus)
            
            if done or truncated: break
            
        rpc.close(); env.close()
        return pop_idx, seed_idx, total_raw_reward, total_shaped_reward
        
    except Exception:
        return pop_idx, seed_idx, -2000.0, -2000.0

def calculate_fitness_sharing(results_matrix: np.ndarray) -> np.ndarray:
    # 使用 shaped reward matrix 计算适应度
    min_val = np.min(results_matrix)
    shifted = results_matrix - min_val + 1e-5
    seed_totals = shifted.sum(axis=0)
    seed_totals[seed_totals == 0] = 1e-9
    shared_matrix = shifted / seed_totals
    return shared_matrix.sum(axis=1)

# ==========================================
# 4. 绘图与日志 (四图绘制)
# ==========================================
def save_plots(run_dir, df_history):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Fitness Sharing Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df_history['gen'], df_history['best_fitness'], color='tab:purple', linewidth=2)
    plt.title("Evolution Metric: Shared Fitness (with Induction)")
    plt.xlabel("Generation"); plt.ylabel("Shared Fitness")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "1_fitness_sharing.png"))
    plt.close()
    
    # 2. Raw Environment Reward Curve (Best & Avg)
    plt.figure(figsize=(10, 6))
    plt.plot(df_history['gen'], df_history['max_raw_reward'], color='tab:green', label='Best Raw', linewidth=2)
    plt.plot(df_history['gen'], df_history['avg_raw_reward'], color='gray', linestyle=':', label='Avg Raw')
    plt.title("Real Environment Performance (No Bonus Included)")
    plt.xlabel("Generation"); plt.ylabel("Total Reward")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "2_raw_reward.png"))
    plt.close()
    
    # 3. Combined Plot (Dual Axis)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    c1 = 'tab:purple'; c2 = 'tab:green'
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Shared Fitness', color=c1)
    ax1.plot(df_history['gen'], df_history['best_fitness'], color=c1, label='Fitness')
    ax1.tick_params(axis='y', labelcolor=c1)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Raw Reward', color=c2)
    ax2.plot(df_history['gen'], df_history['max_raw_reward'], color=c2, linestyle='--', label='Raw Reward')
    ax2.tick_params(axis='y', labelcolor=c2)
    
    plt.title("Combined Metrics: Learning vs Performance")
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, "3_combined_metrics.png"))
    plt.close()

def plot_final_violin(run_dir, final_matrix):
    # final_matrix 存储的是 Raw Reward
    avg_scores = final_matrix.mean(axis=1)
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=avg_scores, inner="quartile", color="skyblue")
    plt.title("Final Robustness Distribution (100 Seeds)")
    plt.ylabel("Average Real Environment Reward")
    plt.grid(True, axis='y', alpha=0.5)
    plt.savefig(os.path.join(run_dir, "plots", "4_final_violin.png"))
    plt.close()

def save_checkpoint(run_dir, gen, population, portfolio):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(ckpt_dir, f"checkpoint_gen_{gen:04d}.npz"),
        generation=gen,
        population=np.array(population),
        portfolio_state=portfolio.state_dict()
    )

def save_detailed_logs(run_dir, gen, raw_matrix, shaped_fitness, subset_seeds):
    records = []
    pop_size, num_seeds = raw_matrix.shape
    for i in range(pop_size):
        for j in range(num_seeds):
            records.append({
                "generation": gen,
                "individual_id": i,
                "seed_value": subset_seeds[j],
                "raw_reward": raw_matrix[i, j], # 记录真实的
                "shared_fitness": shaped_fitness[i] # 记录用于进化的
            })
    df = pd.DataFrame(records)
    log_path = os.path.join(run_dir, "detailed_history.csv")
    df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

# ==========================================
# 5. 主程序
# ==========================================
def run_ga(args):
    ts = int(time.time())
    run_dir = f"runs_pendulum/ga_v3_{ts}"
    os.makedirs(run_dir, exist_ok=True)
    
    # 初始化
    portfolio = SeedPortfolioManager(pool_size=100, subset_k=args.subset_k, max_age=args.max_seed_age)
    model = PendulumNNPolicy()
    base_vec = get_weights_vector(model)
    pop = [mutate(base_vec, args.sigma) for _ in range(args.population)]
    
    history_metrics = []
    best_global_raw = -np.inf
    
    print(f"🚀 Training V4 Started (No hidden layer). Logs: {run_dir}")

    for gen in range(1, args.generations + 1):
        subset = portfolio.get_active_subset()
        
        jobs = []
        for i in range(args.population):
            for j, seed in enumerate(subset):
                jobs.append((i, j, pop[i], seed, args.rpc_port, args.max_steps))
        
        # 准备两个矩阵：一个存Raw(用于Log)，一个存Shaped(用于Fitness)
        raw_reward_matrix = np.zeros((args.population, args.subset_k))
        shaped_reward_matrix = np.zeros((args.population, args.subset_k))
        
        with mp.Pool(args.processes) as pool:
            # 接收4个返回值
            for p_idx, s_idx, raw_r, shaped_r in tqdm(pool.imap_unordered(evaluate_individual, jobs), 
                                                     total=len(jobs), desc=f"Gen {gen}", leave=False):
                raw_reward_matrix[p_idx, s_idx] = raw_r
                shaped_reward_matrix[p_idx, s_idx] = shaped_r
        
        # 1. 计算指标 (全部基于 Raw Reward，保证 Log 真实性)
        raw_avgs = raw_reward_matrix.mean(axis=1)
        gen_max_raw = np.max(raw_avgs)
        gen_avg_raw = np.mean(raw_avgs)
        
        # 2. 计算 Fitness Sharing (基于 Shaped Reward，用于进化引导)
        fit_scores = calculate_fitness_sharing(shaped_reward_matrix)
        best_fit = np.max(fit_scores)
        
        print(f"🏆 Gen {gen} | Fit(Shaped): {best_fit:.2f} | MaxRaw: {gen_max_raw:.1f} | AvgRaw: {gen_avg_raw:.1f}")
        
        # --- LOGGING ---
        save_detailed_logs(run_dir, gen, raw_reward_matrix, fit_scores, subset)
        
        history_metrics.append({
            "gen": gen, "best_fitness": best_fit, 
            "max_raw_reward": gen_max_raw, "avg_raw_reward": gen_avg_raw
        })
        
        if gen % 1 == 0:
            df = pd.DataFrame(history_metrics)
            df.to_csv(os.path.join(run_dir, "metrics_summary.csv"), index=False)
            save_plots(run_dir, df) # 生成3张图
            
        if gen % 5 == 0:
            save_checkpoint(run_dir, gen, pop, portfolio)
            
        if gen_max_raw > best_global_raw:
            best_global_raw = gen_max_raw
            np.savez(os.path.join(run_dir, "best_model_running.npz"), weights=pop[np.argmax(raw_avgs)])

        # --- EVOLUTION (基于 Fitness Scores) ---
        # 这里的 fit_scores 包含了 bonus，所以进化方向会被引导去直立
        portfolio.update_and_refresh(raw_reward_matrix, args.seed_refresh_rate) # 种子更新依然基于真实难度
        
        sorted_idx = np.argsort(fit_scores)
        elite_cnt = max(2, int(args.population * 0.2))
        elites = [pop[i] for i in sorted_idx[-elite_cnt:]]
        
        # 强制保留第一名精英 (不进行变异)
        new_pop = [elites[-1]] 
        
        while len(new_pop) < args.population:
            p1, p2 = random.sample(elites, 2)
            c1, c2 = uniform_crossover(p1, p2)
            new_pop.append(mutate(c1, args.sigma))
            if len(new_pop) < args.population:
                new_pop.append(mutate(c2, args.sigma))
        pop = new_pop

    # --- FINAL EXAM ---
    print("\n🏁 Final Robustness Check (100 Seeds)...")
    final_jobs = []
    for i in range(args.population):
        for j, seed in enumerate(portfolio.master_pool):
            final_jobs.append((i, j, pop[i], seed, args.rpc_port, args.max_steps))
            
    final_raw_matrix = np.zeros((args.population, len(portfolio.master_pool)))
    with mp.Pool(args.processes) as pool:
        for p_idx, s_idx, raw_r, _ in tqdm(pool.imap_unordered(evaluate_individual, final_jobs), total=len(final_jobs)):
            final_raw_matrix[p_idx, s_idx] = raw_r
            
    final_avgs = final_raw_matrix.mean(axis=1)
    best_idx = np.argmax(final_avgs)
    
    print(f"🌟 Best Raw Score: {final_avgs[best_idx]:.2f}")
    np.savez(os.path.join(run_dir, "best_model_final.npz"), weights=pop[best_idx])
    
    plot_final_violin(run_dir, final_raw_matrix) # 生成第4张图
    print(f"💾 Done. Results in {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-port", type=int, default=6000)
    parser.add_argument("--population", type=int, default=100) # 建议增加种群
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=0.1) # 建议设为 0.05 或 0.1
    parser.add_argument("--subset-k", type=int, default=5)
    parser.add_argument("--seed-refresh-rate", type=float, default=0.4)
    parser.add_argument("--max-seed-age", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_global_seed(args.seed)
    run_ga(args)