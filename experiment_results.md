# 实验结果汇总

## 模型与权重

| 模型 | 训练轮数 | 最佳权重 |
| --- | ---: | --- |
| Baseline | 200 | `runs/detect/runs/occluded_sign/baseline_yolov8s_e200_full/weights/best.pt` |
| Occlusion Aug | 200 | `runs/detect/runs/occluded_sign/occlusion_aug_yolov8s_e200_full/weights/best.pt` |

## 真实测试集

测试集：75 张真实遮挡图片、78 个实例。

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.98495 | 0.94750 | 0.96879 | 0.75787 |
| Occlusion Aug | 0.95401 | 0.93784 | 0.95212 | 0.75830 |

CSV：`results/model_comparison.csv`

## 条件测试

| 条件 | 图片 | Baseline mAP@0.5 | Aug mAP@0.5 | Baseline mAP@0.5:0.95 | Aug mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| real | 75 | 0.96879 | 0.95212 | 0.75787 | 0.75830 |
| day | 75 | 0.95745 | 0.95105 | 0.74889 | 0.74777 |
| night | 75 | 0.96973 | 0.95165 | 0.76978 | 0.74191 |
| blur | 75 | 0.96380 | 0.95405 | 0.75196 | 0.75998 |
| occluded | 225 | 0.91127 | **0.94810** | 0.59998 | **0.74315** |

## 追加遮挡等级

| 等级 | Baseline P | Aug P | Baseline R | Aug R | Baseline mAP50 | Aug mAP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| light | 0.89728 | 0.96776 | 0.88418 | 0.92554 | 0.93041 | 0.94158 |
| medium | 0.87312 | 0.96375 | 0.91858 | 0.93902 | 0.93376 | 0.95604 |
| heavy | 0.79445 | 0.92883 | 0.82051 | 0.93887 | 0.87893 | 0.95127 |
| real_occluded | 0.97442 | 0.97417 | 0.95437 | 0.92968 | 0.96385 | 0.95233 |

核心增益：

- light：mAP@0.5 +1.12pp，Recall +4.14pp。
- medium：mAP@0.5 +2.23pp，Recall +2.04pp。
- heavy：mAP@0.5 +7.23pp，Recall +11.84pp。

## 错误统计

| 模型 | FP | FN |
| --- | ---: | ---: |
| Baseline | 2 | 5 |
| Occlusion Aug | 4 | 6 |

Baseline 错误类别：

| 类别 | FP | FN |
| --- | ---: | ---: |
| parking_place | 0 | 1 |
| pedestrian_crossing | 2 | 4 |

## 推理速度

| 模型 | 设备 | 运行次数 | ms/image | FPS |
| --- | --- | ---: | ---: | ---: |
| Baseline | RTX 4090 | 200 | 12.35 | 80.95 |
| Occlusion Aug | RTX 4090 | 200 | 11.80 | 84.72 |
| Baseline | CPU | 50 | 88.46 | 11.30 |
| Occlusion Aug | CPU | 50 | 93.22 | 10.73 |

## 图表

- `results/figures/model_comparison.png`
- `results/figures/condition_comparison.png`
- `results/figures/occlusion_comparison.png`
- `results/figures/condition_examples.jpg`

所有数字均由 `results/*.csv` 读取，不是手工估计值。

## 附加题结果

### ONNX

- 模型：`exports/best.onnx`
- 大小：42.7MB
- Opset：17
- ONNX 官方结构检查：通过
- PyTorch 与 ONNX Runtime：均检测到 1 个 class 0 目标
- 置信度：0.961216 / 0.961367，绝对差约 0.00015

明细：`results/onnx_verification.csv`

### ByteTrack

- 脚本：`scripts/track.py`
- 跟踪器：Ultralytics `bytetrack.yaml`
- 输出：`results/tracking/traffic_sign_bytetrack.mp4`
- 支持视频、网络流和摄像头输入
