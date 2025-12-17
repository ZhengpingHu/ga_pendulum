#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Universal Visualization Script for Pendulum-v1
# Supports both Linear (No Hidden) and MLP (Hidden Layer) policies via CLI args.

import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

# ==========================================
# 1. 动态模型结构定义
# ==========================================
class PendulumNNPolicy(nn.Module):
    def __init__(self, hidden_size=0):
        super().__init__()
        
        # 输入维度固定为 3: [cos(theta), sin(theta), theta_dot]
        if hidden_size > 0:
            # --- 模式 A: 有隐藏层 (你刚刚修改的版本) ---
            print(f"🧠 Building Neural Network with Hidden Layer (Size: {hidden_size})")
            # Input(3) -> Linear -> ReLU -> Linear -> Tanh
            self.net = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Tanh()
            )
        else:
            # --- 模式 B: 无隐藏层 (线性策略) ---
            print(f"🧠 Building Linear Policy (No Hidden Layer)")
            # Input(3) -> Linear -> Tanh
            self.net = nn.Sequential(
                nn.Linear(3, 1),
                nn.Tanh()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pendulum 动作范围 [-2, 2]
        return self.net(x) * 2.0

# ==========================================
# 2. 辅助函数：加载权重
# ==========================================
def set_weights_vector(m: nn.Module, vec: np.ndarray):
    """将 numpy 数组形式的权重加载到 PyTorch 模型中"""
    # 参数量检查
    expected_num = sum(p.numel() for p in m.parameters())
    if vec.size != expected_num:
        print(f"\n[CRITICAL ERROR] Weight Mismatch!")
        print(f"  > Model expects {expected_num} parameters.")
        print(f"  > Loaded file has {vec.size} parameters.")
        print(f"  > Fix: Check your --hidden-size argument.")
        print(f"    - For No Hidden Layer: use --hidden-size 0")
        print(f"    - For Hidden Layer (16): use --hidden-size 16")
        raise RuntimeError("Parameter size mismatch.")

    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

# ==========================================
# 3. 可视化主循环
# ==========================================
def run_visualization(args):
    if not os.path.exists(args.model_path):
        print(f"[Error] Model file not found: {args.model_path}")
        return

    print(f"📂 Loading model from: {args.model_path}")
    try:
        data = np.load(args.model_path)
        weights = data['weights']
        print(f"✅ Weights loaded. Shape: {weights.shape}")
    except Exception as e:
        print(f"[Error] Failed to load .npz file: {e}")
        return

    # 实例化动态模型
    try:
        model = PendulumNNPolicy(hidden_size=args.hidden_size)
        set_weights_vector(model, weights)
    except RuntimeError:
        return 

    model.eval()

    # 创建环境
    env = gym.make("Pendulum-v1", render_mode="human")

    print(f"\n🎥 Starting visualization for {args.episodes} random episodes...")
    print("press Ctrl+C in terminal to stop.")
    time.sleep(1)

    for ep in range(1, args.episodes + 1):
        seed = random.randint(0, 999999)
        obs, _ = env.reset(seed=seed)

        total_reward = 0.0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            # 转换为 Tensor
            s_tensor = torch.tensor(obs, dtype=torch.float32)
            
            with torch.no_grad():
                # 获取动作 (模型内部已经做了 * 2.0 处理)
                action_val = model(s_tensor).item()
                act = [action_val]

            obs, reward, done, truncated, _ = env.step(act)
            total_reward += reward
            steps += 1

            if args.fps > 0:
                time.sleep(1.0 / args.fps)

        # Pendulum 即使是完美策略，分数通常也是负的（越接近0越好）
        # 一般来说 > -200 就算是很稳了
        status = "EXCELLENT 🌟" if total_reward > -200 else ("GOOD 👍" if total_reward > -500 else "Average")
        
        print(f"🎬 Episode {ep}/{args.episodes} | Seed: {seed:<6} | Steps: {steps:<3} | Reward: {total_reward:.2f} | {status}")
        time.sleep(0.5)

    env.close()
    print("\n✨ Visualization finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Pendulum model (Linear or MLP).")
    parser.add_argument("model_path", type=str, help="Path to .npz file (e.g., runs_pendulum/xxx/best_model.npz)")
    
    # 网络结构参数
    parser.add_argument("--hidden-size", type=int, default=0, 
                        help="Size of hidden layer. Set 0 for Linear Policy, 16 for Hidden Layer (default: 0).")
    
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--fps", type=int, default=60, help="Playback speed limit")

    args = parser.parse_args()
    run_visualization(args)