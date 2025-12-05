import gymnasium as gym
import cv2
import math
import time
import numpy as np
from ultralytics import YOLO

# --- 配置 ---
# 请修改为你实际训练的模型路径
MODEL_PATH = "./best.pt" 
IMG_SIZE = 640   # 必须与训练尺寸一致
CONF_THRESHOLD = 0.5

# Pendulum 物理参数
ROD_LENGTH = 1.0
VIEWPORT_W = 4.4 # [-2.2, 2.2] -> width 4.4
VIEWPORT_H = 4.4

def get_ground_truth_keypoints(state, img_w, img_h):
    """
    计算 Ground Truth (真值) 坐标
    state: [cos(theta), sin(theta), theta_dot]
    """
    cos_theta = state[0]
    sin_theta = state[1]
    
    # 1. 物理坐标 (World Coords)
    pivot_wx, pivot_wy = 0.0, 0.0
    
    # Tip 位置修正 (注意这里的负号，对应 Gym 的坐标系)
    tip_wx = -ROD_LENGTH * sin_theta
    tip_wy = ROD_LENGTH * cos_theta
    
    # 2. 映射到像素坐标 (Pixel Coords)
    # 假设视窗中心对应图片中心
    scale_x = img_w / VIEWPORT_W
    scale_y = img_h / VIEWPORT_H
    
    def world_to_pixel(wx, wy):
        # 2.2 是视窗半径
        px = (wx + 2.2) * scale_x
        py = (2.2 - wy) * scale_y 
        return px, py

    kp_pivot = world_to_pixel(pivot_wx, pivot_wy)
    kp_tip = world_to_pixel(tip_wx, tip_wy)
    
    return [kp_pivot, kp_tip]

def run_dynamic_verification():
    # 加载模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"错误: 无法加载模型。请检查路径是否正确。\n{e}")
        return

    # 初始化环境
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    state, _ = env.reset()
    
    print("\n--- 动态验证开始 ---")
    print("绿色圈 = 真值 (Ground Truth)")
    print("红色点 = 预测 (YOLO Prediction)")
    print("按 'q' 键退出")

    prev_time = time.time()
    
    while True:
        # 1. 随机动作 (让摆动起来)
        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            state, _ = env.reset()
        
        # 2. 获取图像
        frame = env.render()
        
        # 转换格式: RGB -> BGR (OpenCV)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        orig_h, orig_w, _ = frame_bgr.shape
        
        # 缩放用于显示和推理
        img_display = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
        
        # 3. 计算 Ground Truth (真值)
        gt_kps_orig = get_ground_truth_keypoints(state, IMG_SIZE, IMG_SIZE)
        gt_points = gt_kps_orig # 已经是 640x640 坐标了

        # 4. YOLO 推理 (预测值)
        results = model.predict(img_display, verbose=False, conf=CONF_THRESHOLD)
        
        pred_points = []
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            for kx, ky in kpts:
                pred_points.append((kx, ky))
        
        # 5. 绘制可视化 & 计算 Diff
        point_names = ["Pivot", "Tip"]
        total_error = 0
        
        # 如果未检测到
        if not pred_points:
            cv2.putText(img_display, "LOST TRACKING!", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            # 确保点数匹配 (以防万一)
            loop_count = min(len(gt_points), len(pred_points))
            
            for i in range(loop_count):
                gx, gy = gt_points[i]
                px, py = pred_points[i]
                
                # 计算误差
                diff = math.sqrt((gx - px)**2 + (gy - py)**2)
                total_error += diff
                
                # A. 绘制真值 (绿圈)
                cv2.circle(img_display, (int(gx), int(gy)), 8, (0, 255, 0), 2)
                
                # B. 绘制预测值 (红点)
                cv2.circle(img_display, (int(px), int(py)), 4, (0, 0, 255), -1)
                
                # C. 绘制误差连线 (白线，仅当误差明显时)
                if diff > 1.5: 
                    cv2.line(img_display, (int(gx), int(gy)), (int(px), int(py)), (255, 255, 255), 1)
                
                # D. 显示单个点的误差数值
                # Pivot 误差通常极小，我们主要看 Tip
                text_info = f"{point_names[i]}: {diff:.1f}px"
                y_pos = 30 + i * 30
                cv2.putText(img_display, text_info, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # 显示平均误差
            avg_error = total_error / loop_count
            # 颜色指示状态：绿色<3px, 黄色<10px, 红色>10px
            if avg_error < 3.0: color = (0, 255, 0)
            elif avg_error < 10.0: color = (0, 255, 255)
            else: color = (0, 0, 255)
            
            cv2.putText(img_display, f"Avg Err: {avg_error:.2f} px", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 计算并显示 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(img_display, f"FPS: {fps:.0f}", (IMG_SIZE - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 6. 显示画面
        cv2.imshow("Pendulum YOLO Verification", img_display)
        
        # 等待 1ms，如果有按键输入且是 'q' 则退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dynamic_verification()