from ultralytics import YOLO

def train():
    # 加载预训练的 YOLOv11n-pose 模型
    model = YOLO('yolo11n-pose.pt') 

    results = model.train(
        data='pendulum-pose.yaml', # 指定刚才创建的配置文件
        epochs=50,                 # 训练轮数
        imgsz=640,                 # 图片大小 (与生成数据时保持一致)
        batch=16,                  # 批次大小
        project='pendulum_runs',   # 结果保存目录
        name='train_v1',           # 实验名称
        device=0,                  # GPU索引 (如果没有GPU改用 'cpu')
        plots=True                 # 自动画图
    )
    
    print("训练完成！模型已保存。")

if __name__ == '__main__':
    train()