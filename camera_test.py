import cv2
from ultralytics import YOLO

# 初始化摄像头
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查连接")
except Exception as e:
    print(f"摄像头初始化失败: {e}")
    exit()

# 加载YOLOv8模型
model = YOLO('best.pt')

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO目标检测
    results = model(frame)
    
    # 绘制检测结果
    annotated_frame = results[0].plot()
    
    # 显示处理结果
    cv2.imshow('YOLOv8实时检测', annotated_frame)
    
    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()