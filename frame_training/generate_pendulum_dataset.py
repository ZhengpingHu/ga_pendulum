import gymnasium as gym
import numpy as np
import cv2
import os
import math
import random
import shutil

# --- 配置参数 ---
OUTPUT_DIR = "pendulum_dataset"
NUM_SAMPLES = 3000       # 总样本数
VAL_SPLIT = 0.1         # 验证集比例
IMG_SIZE = 640          # YOLO 训练尺寸
VERIFY_COUNT = 10       # 生成后随机抽查多少张

# 路径配置
DIRS = {
    "images_train": os.path.join(OUTPUT_DIR, "images", "train"),
    "images_val": os.path.join(OUTPUT_DIR, "images", "val"),
    "labels_train": os.path.join(OUTPUT_DIR, "labels", "train"),
    "labels_val": os.path.join(OUTPUT_DIR, "labels", "val"),
}

# Pendulum 物理与渲染参数
# Gym Pendulum 的视窗范围通常是 x=[-2.2, 2.2], y=[-2.2, 2.2]
VIEWPORT_W = 2.2 * 2
VIEWPORT_H = 2.2 * 2
ROD_LENGTH = 1.0  # 物理长度

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        user_input = input(f"目录 '{OUTPUT_DIR}' 已存在。是否删除并重新生成? (y/n): ")
        if user_input.lower() == 'y':
            shutil.rmtree(OUTPUT_DIR)
        else:
            print("已取消操作。")
            exit()
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    print("目录准备就绪。")

def get_pendulum_keypoints(state, img_w, img_h):
    """
    [修正版] 计算 Pendulum 的 Pivot 和 Tip 的像素坐标。
    修复了 X 轴左右颠倒的问题。
    """
    cos_theta = state[0]
    sin_theta = state[1]
    
    # 1. 物理坐标 (World Coords)
    pivot_wx, pivot_wy = 0.0, 0.0
    
    # Tip 位置修正：
    # Gym Pendulum 中：
    # theta = 0 -> Up (x=0, y=1)
    # theta = pi/2 -> Left (x=-1, y=0)
    # 而 sin(pi/2) = 1, 所以 x 必须是 -sin(theta)
    tip_wx = -ROD_LENGTH * sin_theta  # <--- 修正了这里的符号 (加了负号)
    tip_wy = ROD_LENGTH * cos_theta
    
    # 2. 映射到像素坐标 (Pixel Coords)
    scale_x = img_w / VIEWPORT_W
    scale_y = img_h / VIEWPORT_H
    
    def world_to_pixel(wx, wy):
        # 这里的 2.2 是视窗半径 (viewport size / 2)
        px = (wx + 2.2) * scale_x
        py = (2.2 - wy) * scale_y 
        return px, py
    
    kp_pivot = world_to_pixel(pivot_wx, pivot_wy)
    kp_tip = world_to_pixel(tip_wx, tip_wy)
    
    return [kp_pivot, kp_tip]

def generate_data():
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset()
    
    generated_files = []
    print(f"正在生成 {NUM_SAMPLES} 条数据...")
    
    for i in range(NUM_SAMPLES):
        action = env.action_space.sample()
        state, _, _, _, _ = env.step(action)
        
        frame = env.render()
        orig_h, orig_w, _ = frame.shape
        
        # 1. 图像缩放
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
        
        # 2. 计算原始关键点
        kps_orig = get_pendulum_keypoints(state, orig_w, orig_h)
        
        # 3. 缩放到目标尺寸
        scale_x = IMG_SIZE / orig_w
        scale_y = IMG_SIZE / orig_h
        
        kps_resized = []
        all_x, all_y = [], []
        for kx, ky in kps_orig:
            nx = kx * scale_x
            ny = ky * scale_y
            kps_resized.append((nx, ny))
            all_x.append(nx)
            all_y.append(ny)
            
        # 4. 生成 YOLO 标签
        # Class 0: Pendulum
        # Format: <cls> <cx> <cy> <w> <h> <px1> <py1> <v1> <px2> <py2> <v2>
        
        # Bounding Box: 包含 Pivot 和 Tip 的矩形，加一点 Padding
        padding = 30
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        bbox_x1 = max(0, min_x - padding)
        bbox_y1 = max(0, min_y - padding)
        bbox_x2 = min(IMG_SIZE, max_x + padding)
        bbox_y2 = min(IMG_SIZE, max_y + padding)
        
        norm_cx = ((bbox_x1 + bbox_x2)/2) / IMG_SIZE
        norm_cy = ((bbox_y1 + bbox_y2)/2) / IMG_SIZE
        norm_w = (bbox_x2 - bbox_x1) / IMG_SIZE
        norm_h = (bbox_y2 - bbox_y1) / IMG_SIZE
        
        kpts_str = ""
        for kx, ky in kps_resized:
            kpts_str += f"{kx/IMG_SIZE:.6f} {ky/IMG_SIZE:.6f} 2.000000 "
            
        label_line = f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {kpts_str.strip()}"
        
        # 5. 保存
        is_val = random.random() < VAL_SPLIT
        subset = "val" if is_val else "train"
        
        filename = f"{i:06d}"
        img_path = os.path.join(DIRS[f"images_{subset}"], f"{filename}.jpg")
        txt_path = os.path.join(DIRS[f"labels_{subset}"], f"{filename}.txt")
        
        cv2.imwrite(img_path, frame_resized)
        with open(txt_path, "w") as f:
            f.write(label_line)
            
        generated_files.append((img_path, txt_path))
        
        if (i+1) % 100 == 0:
            print(f"Progress: {i+1}/{NUM_SAMPLES}")
            
    env.close()
    return generated_files

def verify_data(file_list):
    print("\n--- 验证模式 ---")
    print(f"随机显示 {VERIFY_COUNT} 张图片。")
    print("绿色点 = Pivot (固定轴), 红色点 = Tip (杆顶)")
    print("请确认红点是否精确位于杆子末端！")
    print("按任意键下一张，按 'q' 退出。")
    
    samples = random.sample(file_list, min(len(file_list), VERIFY_COUNT))
    
    for img_path, txt_path in samples:
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        with open(txt_path, "r") as f:
            data = list(map(float, f.readline().split()))
            
        # data: cls, cx, cy, bw, bh, px1, py1, v1, px2, py2, v2
        kpts = data[5:]
        
        # 画框
        cx, cy, bw, bh = data[1:5]
        x1 = int((cx - bw/2)*w)
        y1 = int((cy - bh/2)*h)
        x2 = int((cx + bw/2)*w)
        y2 = int((cy + bh/2)*h)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 1)
        
        # 画关键点
        # 0: Pivot (Green), 1: Tip (Red)
        colors = [(0, 255, 0), (0, 0, 255)]
        
        pts_coords = []
        for i in range(2):
            kx = kpts[i*3]
            ky = kpts[i*3+1]
            px, py = int(kx*w), int(ky*h)
            pts_coords.append((px, py))
            
            cv2.circle(img, (px, py), 5, colors[i], -1)
            cv2.putText(img, "Pivot" if i==0 else "Tip", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
            
        if len(pts_coords) >= 2:
            cv2.line(img, pts_coords[0], pts_coords[1], (255, 255, 255), 1)

        cv2.imshow("Pendulum Verification", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    setup_directories()
    files = generate_data()
    if files:
        verify_data(files)