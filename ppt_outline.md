# 答辩 PPT 逐页提纲

正式 PPT 已生成：

```text
docs/答辩PPT_遮挡交通标志检测.pptx
```

## 第 1 页：标题

基于 YOLOv8 的遮挡场景交通标志检测与识别系统

副标题：五类真实遮挡标注、目标区域遮挡增强、分条件评估、实时检测。

## 第 2 页：任务要求与完成情况

- 检测并识别被树木、车辆等遮挡的交通标志。
- 使用 Ultralytics YOLOv8s，类别数为 5。
- Baseline 和改进模型均完整训练 200 轮。
- 完成真实图、正常光照、夜间、模糊和遮挡等级对比。
- 完成误检漏检分析、图片/视频/摄像头推理和 FPS 测试。

## 第 3 页：数据集

- 用户 LabelMe 标注数据，499 张唯一图片、515 个框。
- 五类：give_way、no_stopping、parking_place、
  pedestrian_crossing、tow_truck。
- 划分：349/75/75 张，源图互斥。
- 校验：0 个缺失标签、0 个越界框。

## 第 4 页：方法与改进

- Baseline：YOLOv8s + 349 张原始训练图。
- 改进：每张训练图生成 1 张目标感知遮挡图，训练集扩充到 698 张。
- 遮挡样式：foliage、branch、vehicle。
- 强度：light 10%-20%，medium 20%-45%，heavy 45%-65%。

## 第 5 页：实验设置

| 设置 | 值 |
| --- | --- |
| 输入尺寸 | 640×640 |
| Epochs | 200 |
| Batch size | 32 |
| 随机种子 | 42 |
| GPU | RTX 4090 24GB |
| 优化器 | Ultralytics Auto / AdamW |

## 第 6 页：真实测试集结果

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 98.50% | 94.75% | 96.88% | 75.79% |
| +遮挡增强 | 95.40% | 93.78% | 95.21% | 75.83% |

讲解：Baseline 超过课程 75% 目标。增强模型在普通真实测试集略降，说明增强
引入了场景偏置，不能声称全面优于 Baseline。

## 第 7 页：环境条件

展示 `results/figures/condition_comparison.png`。

- real 为真实原图。
- day、night、blur 为合成条件。
- Baseline 在 real/night/blur 上 mAP@0.5 均超过 96%。
- 增强模型在追加遮挡组明显更稳定。

## 第 8 页：遮挡等级

展示 `results/figures/occlusion_comparison.png`。

| 等级 | Baseline | 增强模型 | 增益 |
| --- | ---: | ---: | ---: |
| light | 93.04% | 94.16% | +1.12pp |
| medium | 93.38% | 95.60% | +2.23pp |
| heavy | 87.89% | 95.13% | +7.23pp |

重点讲：重度遮挡 Recall 提升 11.84 个百分点，改进对 3D 组重点场景有效。

## 第 9 页：训练曲线

并排展示两组 `results.png`、混淆矩阵或 PR 曲线。说明两组均完成 200 轮，
不存在提前停止或只训练几十轮的情况。

## 第 10 页：误检漏检

- Baseline：2 个 FP、5 个 FN。
- 增强模型：4 个 FP、6 个 FN。
- 典型问题集中在 pedestrian_crossing 复合标志、倾斜透视、远距离小目标和
  检测框定位偏差。
- 增强提高了合成遮挡鲁棒性，但在真实原图上增加了少量误检漏检。

## 第 11 页：实时 Demo

| 模型 | RTX 4090 | CPU |
| --- | ---: | ---: |
| Baseline | 80.95 FPS | 11.30 FPS |
| 增强模型 | 84.72 FPS | 10.73 FPS |

现场可播放 `results/demo/traffic_sign_demo.mp4`，也可运行摄像头：

```bash
python scripts/predict.py --model MODEL.pt --source 0
```

## 第 12 页：附加功能

- ONNX：成功导出 42.7MB 模型，官方结构检查通过。
- PyTorch 与 ONNX Runtime 检测类别、数量一致，置信度差约 0.00015。
- ByteTrack：新增视频/摄像头跟踪脚本，输出 Track ID 演示视频。
- 说明：当前跟踪视频由测试图片序列生成，只验证功能链路，不报告跟踪精度。

## 第 13 页：总结与不足

- 完成五类遮挡交通标志检测，核心 mAP 达标。
- 遮挡增强显著改善中重度追加遮挡场景。
- 代价是普通真实测试集 Precision、Recall 和 mAP@0.5 小幅下降。
- 当前真实数据规模小，且没有像素级遮挡比例标注。
- 后续增加真实夜间/车辆遮挡数据，并尝试小目标检测层与 TensorRT。
