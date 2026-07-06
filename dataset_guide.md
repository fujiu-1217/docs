# 数据集说明

## 1. 数据来源

本项目没有使用 LISA 或 TT100K 作为最终训练数据。最终数据来自：

```text
raw/labelme_dataset.zip
```

压缩包内含用户使用 LabelMe 标注的 500 份 JSON 和对应道路图片，标注类型为
矩形检测框。图片中的交通标志存在树叶、树干、车辆、标志重叠和远距离小目标
等真实遮挡情况。

## 2. 类别

| ID | LabelMe 原名 | YOLO 类别名 | 中文含义 | 实例数 |
| ---: | --- | --- | --- | ---: |
| 0 | give way | give_way | 减速让行 | 103 |
| 1 | no stopping | no_stopping | 禁止停车 | 102 |
| 2 | parking place | parking_place | 停车位 | 103 |
| 3 | pedestrian crossing | pedestrian_crossing | 人行横道 | 106 |
| 4 | tow truck | tow_truck | 清障车相关标志 | 101 |

类别数为 5，满足课程“至少 5 类交通标志”的要求。

## 3. 转换结果

转换脚本：

```bash
python scripts/prepare_labelme_dataset.py \
  --source raw/labelme_dataset.zip \
  --output dataset \
  --overwrite
```

脚本完成以下操作：

1. 直接读取 ZIP 内的 LabelMe JSON 和图片，不要求手动解压。
2. 将矩形框转换为 YOLO 归一化格式。
3. 按五个类别目录进行 70%/15%/15% 分层划分。
4. 使用图片内容哈希检查重复文件。
5. 将浮点误差造成的边界轻微越界裁剪到合法范围。
6. 生成 `data.yaml`、`metadata.csv` 和 `summary.yaml`。

原始 500 张图片中有 1 张内容完全重复，因此保留 499 张唯一图片。最终统计：

| 划分 | 图片数 | 检测框数 |
| --- | ---: | ---: |
| train | 349 | 360 |
| val | 75 | 77 |
| test | 75 | 78 |
| 合计 | 499 | 515 |

YOLO 标签格式：

```text
class_id x_center y_center width height
```

全部坐标均归一化到 `[0, 1]`。

## 4. 数据目录

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
├── metadata.csv
└── summary.yaml
```

数据校验命令：

```bash
python scripts/validate_dataset.py \
  --data dataset/data.yaml \
  --metadata dataset/metadata.csv \
  --out results/dataset_validation.yaml
```

当前校验结果为 `valid: true`，没有缺失标签、孤立标签、非法类别或越界框。

## 5. 遮挡增强训练集

`scripts/augment_occlusion.py` 只增强训练集，在目标框内部模拟：

- foliage：树叶色块；
- branch：树枝线条；
- vehicle：车辆颜色遮挡块。

遮挡比例定义：

| 等级 | 在目标框内追加的遮挡比例 |
| --- | --- |
| light | 10%-20% |
| medium | 20%-45% |
| heavy | 45%-65% |

生成命令：

```bash
python scripts/augment_occlusion.py \
  --src dataset \
  --dst dataset_occlusion_aug \
  --copies 1 \
  --seed 42 \
  --overwrite
```

生成 349 张增强训练图，其中 light 98 张、medium 126 张、heavy 125 张。
原图和增强图合计 698 张；验证集与测试集保持原样，保证模型对比公平。

## 6. 条件测试集

`dataset_conditions` 由 75 张真实测试图片生成，每张包含 7 个版本：

| 组别 | 数量 | 是否合成 | 处理 |
| --- | ---: | --- | --- |
| real | 75 | 否 | 原始真实遮挡图 |
| day | 75 | 是 | CLAHE 与轻微亮度校正 |
| night | 75 | 是 | HSV 亮度缩放至 32% |
| blur | 75 | 是 | 11×11 高斯模糊 |
| light | 75 | 是 | 追加轻度目标遮挡 |
| medium | 75 | 是 | 追加中度目标遮挡 |
| heavy | 75 | 是 | 追加重度目标遮挡 |

总计 525 张图片、546 个框。源测试图与训练图互斥，生成版本只用于评估，
没有进入训练过程。

## 7. 遮挡分级解释

原始 LabelMe 标注只提供检测框，没有遮挡区域掩码，因此无法从原数据精确计算
真实遮挡百分比。本项目将原始图统一标记为 `real_occluded`，而 light、medium、
heavy 指在真实遮挡基础上追加的合成遮挡强度。

该设计可以公平比较模型对逐渐缺失目标信息的鲁棒性，但不能替代人工标注的
真实无遮挡/轻度/中度/重度基准。报告和 PPT 中均明确披露这一限制。
