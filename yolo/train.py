import torch
from ultralytics import YOLO
if __name__ == "__main__":
# 初始化YOLOv8模型
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从官方配置构建并加载预训练权重

# 开始训练
    results = model.train(
    data='D:/resource-ai/Bottle-identification/yolo/data.yaml',
    epochs=200,
    batch=256,
    imgsz=320,
    device='0',
    workers=2,
    optimizer='Adam',
    lr0=0.001,
    name='model',
    project='models',
    fraction=1  # 4000 / 13689 (精确计算后的样本比例)
    )

    print(f'训练完成，最佳模型保存在：{results.save_dir}')