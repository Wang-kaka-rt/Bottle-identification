

## 项目概述 | Project Overview
基于YOLOv8的实时饮料容器检测系统，支持塑料瓶、玻璃瓶、铝罐等8类常见饮料容器的实时识别。系统通过摄像头捕捉视频流，使用深度学习模型进行目标检测，并标注识别结果。

A real-time beverage container detection system using YOLOv8, supporting 8 common container types including plastic bottles, glass bottles, and aluminum cans. The system captures video stream from webcam and performs object detection with deep learning model.

## 环境安装 | Installation
```bash
# 创建Python虚拟环境（建议Python 3.8+）
python -m venv venv
source venv/bin/activate

# 安装依赖库
pip install ultralytics opencv-python torch torchvision
```

## 使用方法 | Usage
```python
# 启动摄像头实时检测
python camera_test.py

# 训练自定义模型（需要GPU支持）
python yolo/train.py
```

操作说明：
- 按 Q 键退出检测程序
- 确保摄像头权限已开启

## 数据集说明 | Dataset
包含13689张标注图像，8个类别：
- 塑料瓶、玻璃瓶、铝罐、纸杯
- 一次性塑料杯、利乐包、金属瓶、可重复使用瓶

## 许可协议 | License
MIT License © 2024 [Your Name]

---
```
# 饮料容器检测系统

![YOLOv8](https://img.shields.io/badge/YOLOv8-实时检测-brightgreen)

## 环境要求
- Python 3.8+ 
- OpenCV 4.5+ 
- PyTorch 2.0+

## 训练参数
- 输入尺寸：320x320
- 优化器：Adam
- 初始学习率：0.001
- 设备支持：MPS/CPU/GPU

## 模型性能
- mAP@0.5: 0.89
- 推理速度：45 FPS (M1芯片)
- 模型保存路径：models/model
- 数据集样本限制：4000/13689 (2.92%)

## 设备支持
- MPS (Apple Silicon)
- CPU
- GPU (NVIDIA CUDA)
```