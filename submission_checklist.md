# 课程提交清单

## 必须项

| 课程要求 | 状态 | 对应文件 |
| --- | --- | --- |
| Ultralytics YOLOv5/YOLOv8 | 已完成 | YOLOv8s，`scripts/train.py` |
| 至少 5 类交通标志 | 已完成 | `configs/classes.yaml`、`dataset/data.yaml` |
| 遮挡场景检测 | 已完成 | 真实遮挡 LabelMe 数据 |
| 数据集说明 | 已完成 | `docs/dataset_guide.md` |
| data.yaml | 已完成 | `dataset/data.yaml` |
| 训练脚本/命令 | 已完成 | `scripts/train.py`、`configs/train_*.yaml` |
| 两组完整训练 | 已完成 | Baseline 200 轮、增强模型 200 轮 |
| 训练结果图 | 已完成 | 两组训练目录的 `results.png`、PR/混淆矩阵 |
| 测试集 mAP@0.5 | 已完成 | Baseline 96.88%，`results/model_comparison.csv` |
| 晴朗/正常光照对比 | 已完成 | 合成 day 组，75 张 |
| 夜间/低照度对比 | 已完成 | 合成 night 组，75 张 |
| 模糊对比 | 已完成 | 合成 blur 组，75 张 |
| 遮挡等级分析 | 已完成 | light/medium/heavy，各 75 张 |
| 改进策略 | 已完成 | 目标感知树叶/树枝/车辆遮挡增强 |
| 改进效果验证 | 已完成 | heavy mAP +7.23pp，Recall +11.84pp |
| 误检漏检分析 | 已完成 | 两组 `results/error_cases_*` |
| 图片推理 | 已完成 | `scripts/predict.py` |
| 视频推理 | 已完成 | 同一脚本支持视频 |
| 摄像头实时推理 | 已完成 | `--source 0` |
| FPS | 已完成 | GPU 80.95/84.72，CPU 11.30/10.73 |
| Demo | 已完成 | `results/demo/traffic_sign_demo.mp4` |
| README | 已完成 | `README.md` |
| 项目报告 | 已完成 | `docs/report.md`、正式 Word 版 |
| 答辩 PPT | 已完成 | `docs/答辩PPT_遮挡交通标志检测.pptx` |
| PPT 讲稿 | 已完成 | `docs/ppt_outline.md` |
| 源代码 | 已完成 | `scripts/` |

## 数据真实性说明

- real 组为原始真实测试图片。
- day、night、blur、light、medium、heavy 为合成测试条件。
- light/medium/heavy 是在真实遮挡基础上追加的遮挡，不冒充人工标注的真实遮挡率。
- 合成测试图未进入训练集，避免测试泄漏。

## 加分项

| 可选项 | 状态 | 说明 |
| --- | --- | --- |
| ONNX | 已完成 | `exports/best.onnx`，结构及推理一致性验证通过 |
| TensorRT/NCNN | 未做 | 非必需 |
| ByteTrack 跟踪 | 已完成 | `scripts/track.py`、跟踪演示 MP4 |

## 提交前建议

1. 打开 PPT 检查本机中文字体显示。
2. 打开 Word 后右键目录并选择“更新域”。
3. 播放检测 Demo 和 ByteTrack 跟踪 Demo。
4. 报告中保留“增强模型普通测试略降、重遮挡显著提升”的真实结论。
5. 提交时包含两个 `best.pt` 和 `exports/best.onnx`。
