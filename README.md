# 基于 YOLOv8 的遮挡场景交通标志检测与识别系统

本项目是《人工智能基础》课程期中实践项目，面向部分被树木、车辆、杆件等遮挡的交通标志检测任务，基于 Ultralytics YOLOv8s 实现五类交通标志的定位与识别。

项目支持 LabelMe 标注数据转换、YOLO 数据集构建、目标感知遮挡增强、模型训练、分组评估、误检漏检分析、图片/视频/摄像头推理、ONNX 导出以及基于 ByteTrack 的目标跟踪演示。

## 项目目标

本项目主要完成以下任务：

- 使用 YOLOv8s 进行交通标志目标检测
- 检测并识别 5 类交通标志
- 在测试集上使用 mAP@0.5 作为核心指标
- 对正常光照、低照度、模糊、不同遮挡等级进行性能对比
- 分析误检和漏检案例
- 设计目标感知遮挡增强方法并进行对比实验
- 支持图片、视频和摄像头实时推理
- 支持 ONNX 模型导出和 ByteTrack 跟踪演示

## 检测类别

| ID | 类别名 | 中文含义 |
|---|---|---|
| 0 | give_way | 减速让行 |
| 1 | no_stopping | 禁止停车 |
| 2 | parking_place | 停车位 |
| 3 | pedestrian_crossing | 人行横道 |
| 4 | tow_truck | 清障车相关标志 |

## 项目结构

```txt
.
├── raw/
│   └── labelme_dataset.zip
├── dataset/
│   ├── data.yaml
│   ├── images/
│   └── labels/
├── dataset_occlusion_aug/
├── configs/
│   ├── train_baseline.yaml
│   └── train_occlusion_aug.yaml
├── scripts/
│   ├── prepare_labelme_dataset.py
│   ├── validate_dataset.py
│   ├── augment_occlusion.py
│   ├── train.py
│   ├── evaluate_models.py
│   ├── evaluate_subsets.py
│   ├── error_analysis.py
│   ├── predict.py
│   ├── export_onnx.py
│   └── track.py
├── runs/
├── results/
├── exports/
└── requirements.txt
