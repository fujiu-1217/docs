# 基于 YOLOv8 的遮挡场景交通标志检测与识别系统

## 摘要

交通标志是道路交通环境中传递限速、禁令、警告与指示信息的重要载体。受树木、
车辆、杆件、附属标志以及复杂背景影响，交通标志在实际图像中经常仅有部分区域
可见，容易造成检测框定位偏移、类别混淆和漏检。针对课程 3D 组“部分被树木、
车辆遮挡的交通标志检测”任务，本文设计并实现了一套基于 Ultralytics YOLOv8s
的五类交通标志检测与识别系统。

项目使用用户提供并通过 LabelMe 完成矩形框标注的真实道路图片，建立
give_way、no_stopping、parking_place、pedestrian_crossing 和 tow_truck
五类检测数据。经图片内容去重、标签转换和质量校验后，最终得到 499 张唯一图片
和 515 个目标实例，并按 70%/15%/15% 的比例分层划分训练集、验证集和测试集。
为提高模型对局部信息缺失的适应能力，本文提出目标感知遮挡增强方法：根据交通
标志真实标注框，将模拟树叶、树枝和车辆的遮挡块直接作用于目标区域，并按照
light、medium 和 heavy 三个等级控制附加遮挡比例。

Baseline 与遮挡增强模型均采用相同的 YOLOv8s 结构、640×640 输入尺寸、
Batch Size 32 和 200 个训练轮次。真实测试集上，Baseline 的 Precision、
Recall、mAP@0.5 和 mAP@0.5:0.95 分别为 98.50%、94.75%、96.88% 和
75.79%，其中 mAP@0.5 高于课程要求的 75%。遮挡增强模型在真实原图上的
mAP@0.5 为 95.21%，略低于 Baseline；但在重度附加遮挡组上，mAP@0.5
由 87.89% 提升至 95.13%，Recall 由 82.05% 提升至 93.89%。实验表明，
目标感知遮挡增强以少量普通场景性能为代价，显著改善了中重度遮挡情况下的
检测鲁棒性。

系统同时实现了图片、视频、摄像头推理，完成了晴天/正常光照、低照度、模糊和
不同遮挡等级的分组评估，以及误检漏检案例导出和 GPU/CPU 速度测试。附加功能
包括 ONNX 模型导出与一致性验证，以及基于 ByteTrack 的目标跟踪视频生成。

**关键词：** 交通标志检测；YOLOv8；目标检测；遮挡增强；误检漏检；ONNX；
ByteTrack

## Abstract

Traffic signs are essential visual elements for conveying regulatory, warning, and guidance
information in road environments. In practical scenes, signs are frequently occluded by
foliage, vehicles, poles, or adjacent objects, which may lead to localization errors,
misclassification, and missed detections. This project develops a five-class occluded traffic
sign detection and recognition system based on Ultralytics YOLOv8s.

A user-annotated LabelMe dataset is converted into the YOLO detection format. After duplicate
removal and label validation, the dataset contains 499 unique images and 515 object instances.
A target-aware occlusion augmentation strategy is introduced to place simulated foliage,
branches, and vehicle-like blocks directly over annotated traffic sign regions. Both the
baseline and augmented models are trained for 200 epochs under identical hyperparameters.

On the real test set, the baseline model achieves 96.88% mAP@0.5, exceeding the required
75% target. Although the augmented model obtains a slightly lower mAP@0.5 of 95.21% on
real images, it improves heavy-occlusion mAP@0.5 from 87.89% to 95.13% and recall from
82.05% to 93.89%. The results demonstrate that target-aware occlusion augmentation
substantially improves robustness under severe occlusion while introducing a small trade-off
on ordinary real images. The system further supports image, video, and camera inference,
condition-based evaluation, error analysis, ONNX export, and ByteTrack-based tracking.

**Keywords:** traffic sign detection; YOLOv8; occlusion augmentation; object detection;
ONNX; ByteTrack

## 1 绪论

### 1.1 研究背景

交通标志检测是辅助驾驶、自动驾驶、道路巡检和智能交通系统中的基础视觉任务。
与图像分类不同，目标检测不仅要判断图像中是否存在某类交通标志，还要同时输出
目标的位置和类别。在无遮挡、目标尺寸较大、光照充足的条件下，交通标志通常
具有规则的几何形状和鲜明的颜色，检测难度相对较低；但在真实道路场景中，
树叶和树枝会覆盖标志边缘，车辆和行人可能遮挡标志下半部分，杆件和附属牌会
造成重叠，远距离拍摄还会使目标仅占据很少的像素。

遮挡会使模型可利用的视觉证据减少。轻度遮挡通常只影响局部边缘，对类别判断
影响有限；中度遮挡可能破坏图案、文字或几何轮廓；重度遮挡则可能使模型只看到
目标的一小部分，导致置信度下降或直接漏检。因此，仅报告普通测试集的总体
mAP 无法充分反映遮挡场景中的实际能力，需要额外建立不同遮挡等级和环境条件
的测试分组，并分析模型性能随困难程度变化的规律。

### 1.2 课程任务要求

本项目对应课程 3D 组，核心要求如下：

1. 使用 Ultralytics 开源 YOLOv5 或 YOLOv8 模型。
2. 完成至少 5 类交通标志的定位与识别。
3. 使用 mAP@0.5 作为核心指标，测试集结果不低于 75%。
4. 对正常光照、夜间/低照度、模糊和遮挡条件进行性能对比。
5. 对无明显附加遮挡、轻度、中度和重度遮挡进行分组分析。
6. 分析误检和漏检案例。
7. 提出至少一种改进策略并通过对比实验验证。
8. 支持图片、视频和摄像头推理，并报告实时检测速度。

### 1.3 研究内容与主要贡献

围绕上述任务，本文完成了以下工作：

1. 建立可复现的数据转换流程，将 LabelMe JSON 矩形标注直接转换为 YOLO
   格式，并实现图片哈希去重、边界框裁剪和数据完整性校验。
2. 训练 YOLOv8s Baseline，完成 200 轮完整训练，并在独立真实测试集上达到
   96.88% mAP@0.5。
3. 设计目标感知遮挡增强方法，使模拟遮挡稳定覆盖交通标志，而不是随机落在
   无关背景区域。
4. 构建正常光照、低照度、模糊和轻/中/重附加遮挡条件测试集，分别评价模型。
5. 实现误检漏检可视化、GPU/CPU 基准测试、演示视频、ONNX 导出和 ByteTrack
   跟踪功能，形成完整的训练、评估和部署流程。

## 2 相关理论与技术基础

### 2.1 目标检测任务

目标检测模型的输出通常包含边界框坐标、类别概率和目标置信度。对于每个真实
目标，预测框需要同时满足类别正确和交并比达到阈值，才记为一次正确检测。
交并比 IoU 定义为：

$$
IoU = \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}
$$

其中，$B_{pred}$ 表示预测框，$B_{gt}$ 表示真实标注框。IoU 越高，说明预测框
与真实框的重合程度越高。

### 2.2 YOLOv8s 检测模型

YOLO 系列将目标检测建模为单阶段端到端预测问题，可以在一次前向传播中完成
目标定位和类别判断。本文选择 YOLOv8s 作为主模型，主要原因是其模型规模适中，
相比 nano 版本具有更强的特征表达能力，同时仍能在消费级 GPU 上实现实时推理。

YOLOv8s 的检测网络可概括为三个部分：

1. **Backbone：** 使用卷积、C2f 和 SPPF 模块提取多尺度视觉特征。浅层特征
   保留边缘和颜色信息，深层特征表达更强的语义。
2. **Neck：** 通过自顶向下和自底向上的特征融合，将不同尺度的信息进行组合，
   有利于同时检测大目标和远距离小目标。
3. **Decoupled Head：** 将类别预测和边界框回归分开处理，减少两个任务之间
   的相互干扰。

YOLOv8 采用 Anchor-Free 预测方式，不需要预先聚类锚框尺寸。边界框回归由
定位损失和 Distribution Focal Loss 共同优化，分类分支负责学习五类交通标志
的类别概率。推理阶段通过非极大值抑制去除大量重叠候选框。

### 2.3 评价指标

Precision 表示所有预测为目标的结果中，真正正确的比例：

$$
Precision = \frac{TP}{TP + FP}
$$

Recall 表示所有真实目标中被模型成功检测的比例：

$$
Recall = \frac{TP}{TP + FN}
$$

其中，TP、FP 和 FN 分别表示正确检测、误检和漏检。对每个类别绘制
Precision-Recall 曲线并计算曲线下面积，可得到 AP。五类 AP 的平均值为 mAP。
本文重点报告 IoU 阈值为 0.5 时的 mAP@0.5，同时报告从 0.5 到 0.95、步长
0.05 的平均结果 mAP@0.5:0.95，用于衡量更严格的定位质量。

### 2.4 遮挡增强的作用

随机擦除和 Cutout 等方法通过遮盖局部像素，迫使模型减少对单一局部特征的
依赖。然而，完全随机的遮挡可能落在天空、道路或建筑等背景区域，对目标本身
没有产生有效扰动。本文利用真实标注框约束遮挡位置，使增强直接破坏交通标志
的颜色、轮廓和内部图案，从而更有针对性地模拟遮挡。

## 3 数据集构建与质量控制

### 3.1 数据来源

最终数据来自 `raw/labelme_dataset.zip`。压缩包中包含五个类别目录、道路图片
和对应 LabelMe JSON 文件。标注对象为交通标志的矩形检测框，原始图片中可见
树叶、树干、车辆、附属牌、倾斜透视、夜间光照和小目标等实际困难。

![五类交通标志样例](../results/figures/class_samples.jpg)

*图 1 五类交通标志样例。部分目标本身已受到树叶、杆件或附属标志遮挡。*

### 3.2 类别与样本统计

| 类别 ID | 类别名 | 中文含义 | 训练实例 | 验证实例 | 测试实例 | 合计 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | give_way | 减速让行 | 72 | 16 | 15 | 103 |
| 1 | no_stopping | 禁止停车 | 72 | 15 | 15 | 102 |
| 2 | parking_place | 停车位 | 71 | 16 | 16 | 103 |
| 3 | pedestrian_crossing | 人行横道 | 74 | 15 | 17 | 106 |
| 4 | tow_truck | 清障车相关标志 | 71 | 15 | 15 | 101 |
| 合计 | - | - | 360 | 77 | 78 | 515 |

![类别实例分布](../results/figures/class_distribution.png)

*图 2 五类交通标志实例数量。各类数量均约为 100，类别分布较为均衡。*

原始数据包含 500 份 JSON。转换脚本使用图片内容哈希检查重复文件，发现 1 张
内容完全相同但文件名不同的图片，因此保留 499 张唯一图片。最终划分如下：

| 数据划分 | 图片数 | 实例数 | 占唯一图片比例 |
| --- | ---: | ---: | ---: |
| train | 349 | 360 | 69.94% |
| val | 75 | 77 | 15.03% |
| test | 75 | 78 | 15.03% |

### 3.3 LabelMe 到 YOLO 格式转换

LabelMe 矩形标注以两个对角点表示。设原图宽高为 $W$ 和 $H$，矩形左上角、
右下角分别为 $(x_1,y_1)$ 和 $(x_2,y_2)$，则 YOLO 格式坐标为：

$$
x_c = \frac{x_1+x_2}{2W},\quad
y_c = \frac{y_1+y_2}{2H},\quad
w = \frac{x_2-x_1}{W},\quad
h = \frac{y_2-y_1}{H}
$$

转换后每个标签文件使用如下格式：

```text
class_id x_center y_center width height
```

脚本按图片划分数据，避免同一图片中的多个目标被拆分到不同集合。固定随机种子
42，并在五个类别目录内进行分层抽样，使训练、验证和测试集中类别数量接近。

### 3.4 数据质量校验

`scripts/validate_dataset.py` 对以下问题进行自动检查：

- 图片是否可读取；
- 是否存在缺失标签或孤立标签；
- 标签列数是否为 5；
- 类别 ID 是否位于 0 至 4；
- 归一化坐标是否位于 0 至 1；
- 边界框宽高是否大于 0；
- 元数据中的图片是否真实存在。

转换过程中仅发现 1 个由浮点坐标产生的轻微越界框，脚本将其裁剪到合法范围。
原始数据集、遮挡增强数据集和条件测试集的最终校验结果均为 `valid: true`，
没有缺失标签、非法类别和越界框。

### 3.5 数据真实性与实验边界

原始 499 张图片是真实道路场景图片，但 LabelMe 标注只提供交通标志检测框，
没有提供遮挡区域掩码。因此无法准确计算原图中“被遮挡了百分之多少”。本文将
原始图片统一记为 `real_occluded`，并将后续 light、medium 和 heavy 定义为
在真实遮挡基础上额外施加的合成遮挡强度。该定义用于比较模型对逐渐缺失目标
信息的敏感程度，不等同于人工标注的绝对真实遮挡率。

## 4 系统总体设计

### 4.1 系统流程

系统由数据准备、模型训练、模型评估、推理演示和扩展部署五个模块构成：

```text
LabelMe ZIP
  -> 去重与标注转换
  -> 原始 YOLO 数据集
  -> Baseline 训练
  -> 目标感知遮挡增强
  -> 改进模型训练
  -> 真实测试集与条件测试集评估
  -> 错误案例、FPS、图片/视频/摄像头 Demo
  -> ONNX 导出与 ByteTrack 跟踪
```

数据转换、训练、评估和推理均通过独立脚本完成，各模块输入输出明确，便于重新
运行单个实验，而不需要手工修改代码。

### 4.2 功能模块

| 模块 | 主要脚本 | 功能 |
| --- | --- | --- |
| 数据转换 | `prepare_labelme_dataset.py` | LabelMe ZIP 转 YOLO |
| 数据校验 | `validate_dataset.py` | 检查图片、标签和元数据 |
| 遮挡增强 | `augment_occlusion.py` | 生成目标感知遮挡训练图 |
| 训练 | `train.py` | 根据 YAML 配置训练模型 |
| 总体评估 | `evaluate_models.py` | 同一测试集比较多个模型 |
| 分组评估 | `evaluate_subsets.py` | 按环境或遮挡等级评估 |
| 错误分析 | `error_analysis.py` | 导出 FP/FN 图片和统计 |
| 推理 | `predict.py` | 图片、视频、URL、摄像头检测 |
| ONNX | `export_onnx.py` | 导出并验证 ONNX |
| 跟踪 | `track.py` | ByteTrack 视频/摄像头跟踪 |

## 5 目标感知遮挡增强方法

### 5.1 设计动机

如果直接在整张图像上随机放置遮挡块，大量遮挡会落在背景中，模型仍然看到完整
交通标志，训练信号较弱。目标感知方法首先读取标注框，再随机选择一个目标，
按照目标框宽高和设定比例生成遮挡矩形，确保遮挡与交通标志相交。

### 5.2 遮挡样式

| 样式 | 视觉形式 | 模拟场景 |
| --- | --- | --- |
| foliage | 深浅绿色椭圆和色块 | 树叶覆盖 |
| branch | 多条深色线段 | 树枝穿过标志 |
| vehicle | 灰色、深色实体块 | 车辆或实体遮挡 |

遮挡样式随机选择，并对透明度、颜色和位置进行扰动。增强后仍保留原目标的完整
检测框标签，训练目标是使模型根据剩余可见区域恢复原目标位置和类别。

### 5.3 遮挡等级

| 等级 | 目标框内附加遮挡比例 | 训练图片数 |
| --- | --- | ---: |
| light | 10%-20% | 98 |
| medium | 20%-45% | 126 |
| heavy | 45%-65% | 125 |

每张原始训练图生成 1 张遮挡增强图，共新增 349 张图片，训练集从 349 张扩充到
698 张。验证集和测试集不参与训练增强，避免对比实验受到数据泄漏影响。

### 5.4 条件测试集

为了分析不同退化因素，在 75 张真实测试图基础上生成以下版本：

| 条件 | 图片数 | 处理方法 | 是否合成 |
| --- | ---: | --- | --- |
| real | 75 | 不处理 | 否 |
| day | 75 | CLAHE 与轻微亮度校正 | 是 |
| night | 75 | HSV 亮度缩放至 32% | 是 |
| blur | 75 | 11×11 高斯模糊 | 是 |
| light | 75 | 追加轻度遮挡 | 是 |
| medium | 75 | 追加中度遮挡 | 是 |
| heavy | 75 | 追加重度遮挡 | 是 |

![环境与遮挡条件样例](../results/figures/condition_examples.jpg)

*图 3 同一真实测试图生成的正常光照、低照度、模糊和不同附加遮挡版本。*

合成版本只用于最终评估，没有进入训练集。所有版本沿用原图检测框，以便将模型
差异归因于图像条件变化。

## 6 实验设计与实现环境

### 6.1 硬件与软件环境

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 |
| Python | 3.10.18 |
| 深度学习框架 | PyTorch 2.5.1 |
| CUDA | 11.8 |
| Ultralytics | 8.4.14 |
| GPU | NVIDIA GeForce RTX 4090 24GB |
| CPU | Intel Core i7-13700KF |

### 6.2 训练参数

| 参数 | Baseline | 遮挡增强模型 |
| --- | --- | --- |
| 模型 | YOLOv8s | YOLOv8s |
| 预训练权重 | yolov8s.pt | yolov8s.pt |
| 输入尺寸 | 640×640 | 640×640 |
| Epochs | 200 | 200 |
| Batch Size | 32 | 32 |
| Workers | 8 | 8 |
| 随机种子 | 42 | 42 |
| 优化器 | Auto，实际 AdamW | Auto，实际 AdamW |
| 训练图片 | 349 | 698 |
| 验证图片 | 75 | 75 |
| 测试图片 | 75 | 75 |

两组实验只改变训练数据，模型结构、预训练权重、输入尺寸、训练轮数、批量大小、
验证集和测试集均保持一致。该控制变量设计使结果能够反映遮挡增强本身的影响。

### 6.3 训练完成性

两个训练目录的 `results.csv` 均包含表头和 200 行 epoch 记录，共 201 行，
说明训练完整执行到第 200 轮，没有因早停提前结束。最佳模型由验证过程自动
选择，最终比较统一使用各自的 `best.pt`，而不是最后一轮权重。

![训练过程对比](../results/figures/training_curves.jpg)

*图 4 Baseline 与遮挡增强模型的训练损失和验证指标曲线。*

从曲线可以看出，训练前期分类损失快速下降，后期边界框和 DFL 损失逐渐收敛。
增强模型的训练样本更多、难度更高，但验证 mAP 仍稳定在较高水平，没有出现
训练过程发散。

## 7 实验结果与分析

### 7.1 真实测试集总体结果

真实测试集包含 75 张图片和 78 个实例。两组模型的独立测试结果如下：

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | **98.50%** | **94.75%** | **96.88%** | 75.79% |
| +遮挡增强 | 95.40% | 93.78% | 95.21% | **75.83%** |

![真实测试集总体指标](../results/figures/model_comparison.png)

*图 5 两组模型在真实测试集上的总体指标。*

Baseline 的 mAP@0.5 高于课程最低目标 21.88 个百分点，说明系统已经具备可靠
的五类检测能力。增强模型的 Precision、Recall 和 mAP@0.5 分别下降 3.09、
0.97 和 1.67 个百分点，但 mAP@0.5:0.95 略升 0.04 个百分点，整体定位质量
基本持平。

该结果说明遮挡增强不是对所有场景都产生正收益。增强训练增加了大量局部信息
缺失样本，使模型倾向于根据残缺特征进行判断，在普通真实图上可能产生更多低质
候选框；但这种训练偏置有利于模型应对严重遮挡，具体效果需要结合遮挡分组分析。

### 7.2 各类别检测结果

| 类别 | Baseline mAP@0.5 | 增强模型 mAP@0.5 | 主要观察 |
| --- | ---: | ---: | --- |
| give_way | 99.50% | 99.50% | 三角形轮廓和红色边缘明显 |
| no_stopping | 99.50% | 99.50% | 圆形与红蓝配色辨识度高 |
| parking_place | 95.40% | 95.30% | 夜间和透视变化下仍较稳定 |
| pedestrian_crossing | 90.50% | 82.30% | 复合标志、小目标和框偏移较多 |
| tow_truck | 99.50% | 99.50% | 类别外观较固定 |

pedestrian_crossing 是两组模型中最困难的类别。一方面，该类别在测试集中有
17 个实例，包含倾斜、远距离和复合标志；另一方面，标志内部行人图案较小，
附加箭头或其他标牌容易使模型产生多个局部候选框。

### 7.3 混淆矩阵分析

![归一化混淆矩阵](../results/figures/confusion_matrices.jpg)

*图 6 Baseline 与遮挡增强模型的归一化混淆矩阵。*

多数类别在对角线上具有较高响应，说明类别判别总体准确。错误主要集中于
pedestrian_crossing 与背景之间，表现为真实目标被记为背景，即漏检；增强模型
在真实测试集上的背景误检略有增加，与其 Precision 小幅下降的现象一致。

### 7.4 环境条件对比

| 条件 | Baseline P | Baseline R | Baseline mAP@0.5 | 增强 mAP@0.5 |
| --- | ---: | ---: | ---: | ---: |
| real | 98.50% | 94.75% | 96.88% | 95.21% |
| day | 99.19% | 92.81% | 95.74% | 95.11% |
| night | 99.16% | 94.37% | 96.97% | 95.17% |
| blur | 96.14% | 94.88% | 96.38% | 95.40% |
| occluded | 87.60% | 87.66% | 91.13% | **94.81%** |

![不同环境条件对比](../results/figures/condition_comparison.png)

*图 7 正常光照、低照度、模糊、真实原图和追加遮挡条件下的 mAP@0.5。*

当前 night 组通过统一降低亮度生成，虽然画面整体变暗，但标志的主要颜色和
轮廓仍保留，因此没有出现大幅性能下降。blur 组采用 11×11 高斯模糊，对大尺寸
标志影响有限。相比之下，追加遮挡直接覆盖目标关键区域，对 Baseline 的影响
最大，其 mAP@0.5 降至 91.13%，mAP@0.5:0.95 降至 60.00%。

遮挡增强模型在综合 occluded 组上的 mAP@0.5 为 94.81%，比 Baseline 提高
3.68 个百分点；mAP@0.5:0.95 从 60.00% 提高到 74.32%，说明增强不仅减少
漏检，也明显改善了遮挡条件下的边界框定位质量。

### 7.5 遮挡等级对比

| 组别 | Baseline mAP@0.5 | 增强 mAP@0.5 | mAP 增益 | Baseline Recall | 增强 Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| light | 93.04% | 94.16% | +1.12pp | 88.42% | 92.55% |
| medium | 93.38% | 95.60% | +2.23pp | 91.86% | 93.90% |
| heavy | 87.89% | **95.13%** | **+7.23pp** | 82.05% | **93.89%** |
| real_occluded | 96.38% | 95.23% | -1.15pp | 95.44% | 92.97% |

![遮挡等级对比](../results/figures/occlusion_comparison.png)

*图 8 不同附加遮挡强度下两组模型的 mAP@0.5。*

随着遮挡加重，Baseline 的性能总体下降，heavy 组下降最明显。增强模型在
light、medium 和 heavy 三组均获得提升，且遮挡越重，收益越大。heavy 组
mAP@0.5 提升 7.23 个百分点，Recall 提升 11.84 个百分点，
mAP@0.5:0.95 提升 16.07 个百分点。这一结果与方法设计目标一致：模型通过
训练中反复观察残缺目标，学会利用剩余边缘、颜色和局部图案完成识别。

real_occluded 组包含 real、day、night 和 blur 四类未追加遮挡版本。增强模型
在该组 mAP@0.5 下降 1.15 个百分点，进一步表明该方法存在普通场景性能与重度
遮挡鲁棒性之间的权衡。

## 8 误检与漏检分析

### 8.1 分析方法

`scripts/error_analysis.py` 使用置信度阈值 0.25 和 IoU 阈值 0.5，将预测框
与同类别真实框进行一对一匹配。未匹配预测框记为 FP，未匹配真实框记为 FN，
并自动保存带颜色标注的错误图片。

| 模型 | False Positive | False Negative |
| --- | ---: | ---: |
| Baseline | 2 | 5 |
| +遮挡增强 | 4 | 6 |

Baseline 的 5 个 FN 中，1 个属于 parking_place，4 个属于
pedestrian_crossing；2 个 FP 均属于 pedestrian_crossing。增强模型在真实
测试集上新增少量 FP 和 FN，与其总体 Precision、Recall 略低的结果一致。

![典型误检与漏检案例](../results/figures/error_examples.jpg)

*图 9 典型错误案例。蓝框表示未匹配真实框，红框表示未匹配预测框。*

### 8.2 主要错误原因

1. **目标尺寸较小。** 远距离标志在 640×640 输入中只占较少像素，内部图案
   和边缘信息不足，容易被背景特征淹没。
2. **透视与倾斜。** 部分标志拍摄角度较大，外观从规则矩形变为明显梯形，
   预测框与人工标注框可能因定位差异而低于 0.5 IoU。
3. **复合标志。** 人行横道标志上方存在附加箭头或相邻标牌时，模型可能将
   不同局部区域分别生成候选框，造成重复检测或框范围不一致。
4. **遮挡与背景叠加。** 树干、杆件和相似颜色背景同时出现时，目标边缘不完整，
   容易降低置信度。
5. **增强分布偏差。** 合成遮挡形态相对简单，不能完全覆盖真实树叶纹理、车辆
   轮廓和复杂光照，增强模型在真实原图上仍可能增加误检。

### 8.3 改进方向

针对上述问题，可进一步提高输入分辨率、增加小目标检测层、使用真实遮挡物
Copy-Paste、对复合标志统一标注规范，并增加真实夜间和车辆遮挡样本。对于
部署阶段，还可以根据实际应用调整置信度阈值，减少低置信度误检。

## 9 实时检测与系统演示

### 9.1 推理功能

统一推理脚本支持：

- 单张图片或图片目录；
- 本地视频；
- 网络视频流；
- 摄像头编号，例如 `--source 0`。

推理输出包含边界框、类别名和置信度。已生成 75 张测试图片检测结果和一段
37 秒演示视频。

### 9.2 速度测试

速度测试包含模型调用、图像预处理和后处理，输入尺寸为 640。GPU 每个模型运行
200 次，CPU 每个模型运行 50 次。

| 模型 | 设备 | 单张平均耗时 | FPS |
| --- | --- | ---: | ---: |
| Baseline | RTX 4090 | 12.35 ms | 80.95 |
| +遮挡增强 | RTX 4090 | 11.80 ms | 84.72 |
| Baseline | CPU | 88.46 ms | 11.30 |
| +遮挡增强 | CPU | 93.22 ms | 10.73 |

两组模型结构相同，参数量一致，速度差异主要来源于测量波动。两者在 RTX 4090
上均达到 80 FPS 以上，能够满足实时检测需要。CPU 约为 11 FPS，适合低帧率
演示，但若要达到 30 FPS，可进一步使用 ONNX/OpenVINO/TensorRT 等部署后端。

![检测结果示例](../results/demo/final_test/labelme_0004.jpg)

*图 10 增强模型在真实测试图片上的检测结果。*

## 10 附加功能

### 10.1 ONNX 模型导出与验证

为提高模型的跨平台部署能力，本文将遮挡增强模型导出为固定输入尺寸 640×640、
opset 17 的 ONNX 模型：

```text
exports/best.onnx
```

导出模型大小为 42.7MB。使用 ONNX 官方检查器验证通过，模型 IR Version 为
8，共包含 231 个计算节点。随后在同一张测试图片上分别使用 PyTorch 和
ONNX Runtime CPU 后端运行 30 次：

| 后端 | 检测数 | 类别 ID | 平均置信度 | ms/image | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 1 | 0 | 0.961216 | 168.89 | 5.92 |
| ONNX Runtime | 1 | 0 | 0.961367 | 176.88 | 5.65 |

两种后端均检测到同一个 give_way 目标，置信度绝对差仅约 0.00015，说明导出前后
预测结果一致。本次速度测试通过 Ultralytics 高层接口运行，包含预处理、后处理
和 Python 调用开销，仅用于功能验证，不代表经过线程、图优化或批处理调优后的
ONNX 极限性能。

### 10.2 ByteTrack 目标跟踪

在逐帧检测基础上，项目新增 `scripts/track.py`，调用 Ultralytics 内置
ByteTrack 配置，对视频或摄像头目标进行数据关联并显示 Track ID。跟踪演示
视频保存为：

```text
results/tracking/traffic_sign_bytetrack.mp4
```

运行方式：

```bash
python scripts/track.py \
  --model runs/detect/runs/occluded_sign/occlusion_aug_yolov8s_e200_full/weights/best.pt \
  --source input.mp4 \
  --tracker bytetrack.yaml
```

当前演示输入由测试图片序列生成，主要用于验证检测与跟踪代码链路、输出格式和
Track ID 显示功能。由于相邻画面并非同一路段连续采集，本文不报告 MOTA、IDF1
等多目标跟踪指标，也不将该演示解释为真实连续道路跟踪性能。

![ByteTrack 跟踪结果](../results/figures/tracking_example.png)

*图 11 ByteTrack 演示结果，检测框上方显示 Track ID、类别和置信度。*

## 11 可复现性与工程文件

项目主要产物如下：

| 内容 | 路径 |
| --- | --- |
| 原始数据压缩包 | `raw/labelme_dataset.zip` |
| YOLO 数据配置 | `dataset/data.yaml` |
| 数据集说明 | `docs/dataset_guide.md` |
| Baseline 最佳权重 | `runs/detect/runs/occluded_sign/baseline_yolov8s_e200_full/weights/best.pt` |
| 增强模型最佳权重 | `runs/detect/runs/occluded_sign/occlusion_aug_yolov8s_e200_full/weights/best.pt` |
| ONNX 模型 | `exports/best.onnx` |
| 测试指标 | `results/model_comparison.csv` |
| 分组指标 | `results/*_condition_metrics.csv`、`results/*_occlusion_metrics.csv` |
| 错误案例 | `results/error_cases_baseline/`、`results/error_cases_augmented/` |
| 检测 Demo | `results/demo/traffic_sign_demo.mp4` |
| 跟踪 Demo | `results/tracking/traffic_sign_bytetrack.mp4` |
| 答辩 PPT | `docs/答辩PPT_遮挡交通标志检测.pptx` |

主要复现命令：

```bash
conda activate torch_env
pip install -r requirements.txt

python scripts/prepare_labelme_dataset.py \
  --source raw/labelme_dataset.zip \
  --output dataset \
  --overwrite

python scripts/augment_occlusion.py \
  --src dataset \
  --dst dataset_occlusion_aug \
  --copies 1 \
  --overwrite

python scripts/train.py --config configs/train_baseline.yaml
python scripts/train.py --config configs/train_occlusion_aug.yaml
```

## 12 结论与展望

本文完成了一套面向遮挡道路场景的五类交通标志检测与识别系统。通过 LabelMe
数据转换、分层划分和自动校验，建立了 499 张图片、515 个目标的 YOLO 数据集。
YOLOv8s Baseline 在真实测试集上取得 96.88% mAP@0.5，满足并明显超过课程
75% 的指标要求。

针对遮挡任务，本文提出目标感知遮挡增强，将树叶、树枝和车辆风格遮挡直接作用
于真实交通标志框。增强模型在普通真实测试集上的 mAP@0.5 下降 1.67 个百分点，
说明增强方法存在一定分布偏移；但其在中度和重度附加遮挡组中分别提升 2.23 和
7.23 个百分点，重度遮挡 Recall 提升 11.84 个百分点。综合来看，该方法成功
增强了模型在课程重点场景中的鲁棒性，但不能简单表述为所有条件下均优于
Baseline。

项目还完成了不同环境条件评估、误检漏检分析、GPU/CPU 实时性测试、ONNX 导出
和 ByteTrack 跟踪功能，形成从数据处理到模型部署的完整工程流程。

后续工作可以从以下方向展开：

1. 扩充真实夜间、雨雪、逆光和车辆遮挡数据，减少对合成条件的依赖。
2. 对真实遮挡区域进行像素级标注，建立更严格的无/轻/中/重遮挡基准。
3. 提高输入分辨率或增加小目标检测层，改善远距离标志漏检。
4. 使用真实树叶、车辆和行人实例进行 Copy-Paste，提高遮挡纹理真实性。
5. 采用 TensorRT、OpenVINO 或 NCNN 进行端侧部署和速度优化。
6. 使用连续道路视频评价 ByteTrack 的 MOTA、IDF1 和 ID Switch 等指标。

## 参考文献

[1] Redmon J, Divvala S, Girshick R, Farhadi A. You Only Look Once: Unified,
Real-Time Object Detection. Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, 2016.

[2] Redmon J, Farhadi A. YOLO9000: Better, Faster, Stronger. Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition, 2017.

[3] Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of
Object Detection. arXiv preprint arXiv:2004.10934, 2020.

[4] Ultralytics. Ultralytics YOLO Documentation and Open-Source Repository.
https://docs.ultralytics.com/

[5] Zhong Z, Zheng L, Kang G, Li S, Yang Y. Random Erasing Data Augmentation.
Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

[6] DeVries T, Taylor G W. Improved Regularization of Convolutional Neural Networks
with Cutout. arXiv preprint arXiv:1708.04552, 2017.

[7] Zhang Y, Sun P, Jiang Y, et al. ByteTrack: Multi-Object Tracking by Associating
Every Detection Box. European Conference on Computer Vision, 2022.

## 附录 A 关键结果文件

```text
results/model_comparison.csv
results/baseline_condition_metrics.csv
results/augmented_condition_metrics.csv
results/baseline_occlusion_metrics.csv
results/augmented_occlusion_metrics.csv
results/onnx_verification.csv
results/error_cases_baseline/summary.csv
results/error_cases_augmented/summary.csv
```

## 附录 B 推理与扩展命令

图片、视频或摄像头检测：

```bash
python scripts/predict.py --model MODEL.pt --source IMAGE_OR_VIDEO
python scripts/predict.py --model MODEL.pt --source 0
```

ONNX 导出与验证：

```bash
python scripts/export_onnx.py \
  --model runs/detect/runs/occluded_sign/occlusion_aug_yolov8s_e200_full/weights/best.pt \
  --out-dir exports
```

ByteTrack 跟踪：

```bash
python scripts/track.py \
  --model runs/detect/runs/occluded_sign/occlusion_aug_yolov8s_e200_full/weights/best.pt \
  --source input.mp4
```
