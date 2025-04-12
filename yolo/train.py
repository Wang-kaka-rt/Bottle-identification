from ultralytics import YOLO

# 初始化YOLOv8模型
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从官方配置构建并加载预训练权重

# 开始训练
results = model.train(
    data='/Users/wangzhenyu/Desktop/Project-AI/Opencv/yolo/data.yaml',
    epochs=20,
    batch=8,
    imgsz=320,
    device='mps',
    workers=2,
    optimizer='Adam',
    lr0=0.001,
    name='model',
    project='models',
    fraction=0.074  # 4000 / 13689 (精确计算后的样本比例)
)

print(f'训练完成，最佳模型保存在：{results.save_dir}')