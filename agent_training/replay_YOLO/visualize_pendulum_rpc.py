#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# RPC Visualization Script for Pendulum (Real YOLO Loop)
# Requires: gymnasium, torch, numpy, opencv-python

import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from multiprocessing.connection import Client

# 尝试导入 OpenCV 用于画面显示
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ Warning: 'opencv-python' not found. Visualization window will not show (only logs).")
    print("   Install it via: pip install opencv-python")

# ==========================================
# 1. 网络结构 (必须匹配 YOLO 训练时的 4D 输入)
# ==========================================
class PendulumNNPolicy(nn.Module):
    def __init__(self, hidden_size=0):
        super().__init__()
        
        # YOLO Server 返回的是 4维: [x, y, vx, vy]
        if hidden_size > 0:
            print(f"🧠 Loading MLP Policy (Input: 4 -> Hidden: {hidden_size} -> Output: 1)")
            self.net = nn.Sequential(
                nn.Linear(4, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Tanh()
            )
        else:
            print(f"🧠 Loading Linear Policy (Input: 4 -> Output: 1)")
            self.net = nn.Sequential(
                nn.Linear(4, 1),
                nn.Tanh()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * 2.0

# ==========================================
# 2. RPC 通信模块 (复用训练代码)
# ==========================================
class RPCClient:
    def __init__(self, host, port, authkey=b"pendulum-rpc"):
        self.address = (host, port)
        try:
            self.conn = Client(self.address, authkey=authkey)
            print(f"✅ Connected to YOLO Server at {host}:{port}")
        except ConnectionRefusedError:
            print(f"❌ Connection Failed! Is the server running on {port}?")
            raise

    def reset(self):
        self.conn.send(("reset", None))
        self.conn.recv()

    def infer(self, frame_bgr):
        # 发送 BGR 格式图像 (OpenCV 格式)
        self.conn.send(("infer", frame_bgr))
        ok, res = self.conn.recv()
        return res if ok else None

    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

# ==========================================
# 3. 辅助函数
# ==========================================
def set_weights_vector(m: nn.Module, vec: np.ndarray):
    expected = sum(p.numel() for p in m.parameters())
    if vec.size != expected:
        print(f"\n[CRITICAL ERROR] Weight Mismatch!")
        print(f"  > Model expects {expected} params (Input=4).")
        print(f"  > Loaded file has {vec.size} params.")
        print(f"  > Check --hidden-size or verify if model was trained with YOLO(4D).")
        return False
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n
    return True

# ==========================================
# 4. 可视化主循环
# ==========================================
def run_rpc_visualization(args):
    # 1. 加载模型
    if not os.path.exists(args.model_path):
        print(f"File not found: {args.model_path}")
        return

    data = np.load(args.model_path)
    weights = data['weights']
    
    model = PendulumNNPolicy(hidden_size=args.hidden_size)
    if not set_weights_vector(model, weights):
        return
    model.eval()

    # 2. 连接服务器
    try:
        rpc = RPCClient(args.host, args.port, authkey=args.authkey.encode('utf-8'))
    except:
        return

    # 3. 创建环境 (必须是 rgb_array 以获取图像传给 Server)
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    print(f"\n🎥 Starting Real-Loop Replay ({args.episodes} episodes)...")
    print("Pre-computation check: If window doesn't appear, check opencv install.")

    for ep in range(1, args.episodes + 1):
        seed = random.randint(0, 99999)
        obs, _ = env.reset(seed=seed)
        
        # 通知 Server 重置 (清除历史轨迹缓存)
        rpc.reset()
        
        last_state = np.zeros(4, dtype=np.float32)
        total_reward = 0.0
        steps = 0
        
        # 本地显示窗口初始化
        window_name = f"Replay Ep {ep}"
        
        while True:
            # A. 获取画面
            frame_rgb = env.render() # Gym 返回 RGB
            if frame_rgb is None: break
            
            # B. 转换颜色 (Gym RGB -> OpenCV BGR)
            frame_bgr = frame_rgb[..., ::-1].copy()
            
            # C. 显示画面 (给人类看)
            if HAS_CV2:
                # 在画面上打印一点信息
                display_frame = frame_bgr.copy()
                cv2.putText(display_frame, f"R: {total_reward:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("YOLO Client View", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): # 按 q 退出
                    env.close()
                    rpc.close()
                    return

            # D. 发送给 Server 获取识别结果 (真正的 YOLO 识别！)
            state = rpc.infer(frame_bgr)
            
            # 处理丢帧/识别失败的情况
            if state is None:
                state = last_state # 保持上一帧状态 (模拟真实环境中的鲁棒性处理)
            else:
                last_state = state

            # E. 神经网络推理
            s_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_val = model(s_tensor).item()
            
            # F. 环境交互
            _, reward, done, truncated, _ = env.step([action_val])
            total_reward += reward
            steps += 1
            
            # G. 控制速度
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
            
            if done or truncated or steps >= 200:
                break
        
        print(f"🎬 Episode {ep} | Seed: {seed} | Reward: {total_reward:.2f}")
        time.sleep(0.5)

    if HAS_CV2:
        cv2.destroyAllWindows()
    env.close()
    rpc.close()
    print("✨ Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to .npz file (trained with YOLO/4D)")
    parser.add_argument("--hidden-size", type=int, default=0, help="0 for Linear, 16 for Hidden")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--authkey", type=str, default="pendulum-rpc")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=0, help="0 for real-time (fastest), 30/60 to slow down")
    
    args = parser.parse_args()
    run_rpc_visualization(args)