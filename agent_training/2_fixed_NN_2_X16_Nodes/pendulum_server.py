#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from threading import Thread
from multiprocessing.connection import Listener
from typing import Optional

# 限制 CPU 占用
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Import ultralytics failed: {e}")

print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# --- 1. 可复现性设置 ---
def set_server_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print(f"🔒 Server Seed set to: {seed}")

# --- 2. 状态估计器 ---
class PendulumStateEstimator:
    def __init__(self, model_path: str, device: str = "cuda:0", img_size: int = 640):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        print(f"[Estimator] Loading YOLO from {model_path} ...")
        self.model = YOLO(model_path).to(self.device)
        self.prev_pos: Optional[np.ndarray] = None

    def clone(self):
        new_obj = PendulumStateEstimator.__new__(PendulumStateEstimator)
        new_obj.device = self.device
        new_obj.img_size = self.img_size
        new_obj.model = self.model
        new_obj.prev_pos = None
        return new_obj

    def begin_episode(self):
        self.prev_pos = None

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        results = self.model.predict(frame_bgr, verbose=False, device=self.device, conf=0.5)
        if not results or len(results) == 0: return None
        r = results[0]
        if r.keypoints is None or r.keypoints.xy.shape[1] < 2: return None

        kpts = r.keypoints.xy[0].cpu().numpy()
        pivot_x, pivot_y = kpts[0]
        tip_x, tip_y = kpts[1]
        
        scale = self.img_size / 2.0 
        norm_x = (tip_x - pivot_x) / scale
        norm_y = (tip_y - pivot_y) / scale
        
        current_pos = np.array([norm_x, norm_y], dtype=np.float32)
        
        if self.prev_pos is None:
            velocity = np.zeros(2, dtype=np.float32)
        else:
            velocity = current_pos - self.prev_pos
            
        self.prev_pos = current_pos
        return np.concatenate([current_pos, velocity])

# --- 3. RPC Server (静默版) ---
class InferenceServer:
    def __init__(self, est: PendulumStateEstimator, host="127.0.0.1", port=6000, authkey=b"pendulum-rpc"):
        self.master_est = est
        self.address = (host, port)
        self.authkey = authkey

    def _handle_client(self, conn):
        session = self.master_est.clone()
        try:
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError):
                    break # 正常断开，不报错
                except Exception as e:
                    print(f"[RPC] Recv Error: {e}")
                    break

                if msg[0] == "reset":
                    session.begin_episode()
                    conn.send((True, "ok"))
                elif msg[0] == "infer":
                    res = session.process_frame(msg[1])
                    try:
                        conn.send((True, res) if res is not None else (False, "fail"))
                    except (ConnectionResetError, BrokenPipeError):
                        break
        finally:
            conn.close()

    def serve_forever(self):
        l = Listener(self.address, authkey=self.authkey)
        print(f"[RPC] Pendulum Server listening on {self.address}")
        while True:
            Thread(target=self._handle_client, args=(l.accept(),), daemon=True).start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./best.pt")
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()
    
    set_server_seed(42) # 固定随机性
    est = PendulumStateEstimator(args.model)
    InferenceServer(est, port=args.port).serve_forever()