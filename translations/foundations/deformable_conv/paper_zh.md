# 可变形卷积网络（Deformable Convolutional Networks）

**作者：** Jifeng Dai\*、Haozhi Qi\*$^{\dag}$、Yuwen Xiong\*$^{\dag}$、Yi Li\*$^{\dag}$、Guodong Zhang\*$^{\dag}$、Han Hu、Yichen Wei
（\* 同等贡献；$\dag$ 在微软亚洲研究院实习期间完成）
**机构：** Microsoft Research Asia
**会议：** ICCV 2017
**代码：** https://github.com/msracver/Deformable-ConvNets
**原文 TeX：** [`arxiv/foundations/deformable_conv/extracted/egpaper_for_arxiv.tex`](../../../arxiv/foundations/deformable_conv/extracted/egpaper_for_arxiv.tex)

---

## 摘要

卷积神经网络（CNN）因其构建模块中的**固定几何结构**，在建模几何变换方面存在固有局限。本文提出两个新模块以增强 CNN 的变换建模能力：**可变形卷积（deformable convolution）** 与 **可变形 RoI 池化（deformable RoI pooling）**。二者的共同思想是：在模块的空间采样位置上增加额外偏移，并从目标任务中**学习**这些偏移，**无需额外监督**。新模块可即插即用地替换现有 CNN 中的对应普通模块，并通过标准反向传播端到端训练，从而得到**可变形卷积网络（deformable ConvNets）**。大量实验验证了方法的有效性。本文首次表明：在深度 CNN 中学习稠密空间变换，对目标检测、语义分割等复杂视觉任务是有效的。

---

## 1. 引言

视觉识别的一个核心挑战是如何容纳几何变化，或建模物体在尺度、姿态、视角与部件形变上的几何变换。一般有两条路：

1. **构造含充分变化的训练集**：通常通过对已有样本做仿射等增强实现。可从数据中学到鲁棒表示，但代价是训练昂贵、模型参数复杂。
2. **使用变换不变的特征与算法**：涵盖 SIFT、基于滑窗的检测范式等经典技术。

上述两条路有两个共同缺陷：

- 几何变换被假定为**固定且已知**——用这种先验做数据增强与特征/算法设计，难以泛化到含**未知变换**的新任务。
- 即便变换已知，对手工设计不变特征与算法而言，过于复杂的变换也可能**很难甚至不可行**。

近年来 CNN 在分类、分割、检测上取得巨大成功，但仍共享上述缺陷。其几何变换建模能力主要来自**大量数据增强、大模型容量**，以及少量简单手工模块（如 max-pooling 提供小范围平移不变性）。

简言之，CNN 对**大范围、未知变换**的建模能力有先天不足。根源在于模块的固定几何结构：

- 卷积单元在**固定位置**采样输入特征图；
- 池化层以**固定比例**降低分辨率；
- RoI 池化将候选框划分为**固定空间 bin** 等。

内部缺少处理几何变换的机制，会带来明显问题。例如：同一 CNN 层中所有激活单元的**感受野大小相同**——对编码空间位置语义的高层尤其不合理，因为不同位置可能对应不同尺度/形变的物体；又如，尽管检测进展迅速，方法仍依赖基于**原始边界框**的特征提取，对非刚性物体明显次优。

![图1：标准卷积与可变形卷积的采样位置示意。(a) 规则网格；(b) 带偏移的变形采样；(c)(d) 为 (b) 的特例，说明可变形卷积可概括尺度、各向异性长宽比、旋转等变换。](../../../arxiv/foundations/deformable_conv/extracted/figures/DC_concept_v4a.png)

本文引入两个显著增强 CNN 几何变换建模能力的新模块：

1. **可变形卷积**：在标准卷积的规则网格采样位置上增加二维偏移，使采样网格可自由形变。偏移由前一层特征图经额外卷积层学习，因而形变以**局部、稠密、自适应**的方式依赖输入。
2. **可变形 RoI 池化**：在常规 RoI 池化的各 bin 位置上增加偏移；偏移由特征与 RoI 共同学习，使不同形状物体的部件定位更自适应。

两模块均轻量，仅为学习偏移增加少量参数与计算；可直接替换普通对应模块，用标准 BP 端到端训练。所得网络称为 **deformable ConvNets**。

方法与 STN、可变形部件模型（DPM）等高层次思想相近：均含内部变换参数并由数据学习。关键差异在于：可变形 ConvNets 以**简单、高效、深层、端到端**的方式处理**稠密**空间变换（详见相关工作讨论）。

---

## 2. 可变形卷积网络

CNN 中特征图与卷积是三维的；两模块均在二维空间域操作，通道维上操作相同。下文以二维叙述，推广到三维直接。

### 2.1 可变形卷积

二维卷积两步：① 在输入特征图 $\mathbf{x}$ 上用规则网格 $\mathcal{R}$ 采样；② 用权重 $\mathbf{w}$ 对采样值加权求和。$\mathcal{R}$ 定义感受野大小与膨胀。例如

$$
\mathcal{R}=\{(-1,-1),(-1,0),\ldots,(0,1),(1,1)\}
$$

为膨胀率 1 的 $3\times3$ 核。对输出特征图上每个位置 $\mathbf{p}_0$：

$$
\mathbf{y}(\mathbf{p}_0)=\sum_{\mathbf{p}_n\in\mathcal{R}}\mathbf{w}(\mathbf{p}_n)\cdot \mathbf{x}(\mathbf{p}_0+\mathbf{p}_n).
$$

可变形卷积为 $\mathcal{R}$ 增加偏移 $\{\Delta\mathbf{p}_n|n=1,\ldots,N\}$（$N=|\mathcal{R}|$）：

$$
\mathbf{y}(\mathbf{p}_0)=\sum_{\mathbf{p}_n\in\mathcal{R}}\mathbf{w}(\mathbf{p}_n)\cdot \mathbf{x}(\mathbf{p}_0+\mathbf{p}_n+\Delta\mathbf{p}_n).
$$

偏移通常为分数坐标，用**双线性插值**实现：

$$
\mathbf{x}(\mathbf{p})=\sum_{\mathbf{q}} G(\mathbf{q},\mathbf{p})\cdot \mathbf{x}(\mathbf{q}),
$$

其中 $G(\mathbf{q},\mathbf{p})=g(q_x,p_x)\cdot g(q_y,p_y)$，$g(a,b)=\max(0,1-|a-b|)$。$G$ 仅对少数 $\mathbf{q}$ 非零，计算高效。

偏移由**同一输入特征图**上并联的卷积层预测：核尺寸与膨胀率与当前层一致；输出偏移场与输入同分辨率，通道数为 $2N$（$N$ 个二维偏移）。训练时，生成输出特征的卷积核与偏移**同时学习**；偏移梯度经双线性插值回传（附录给出细节）。

![图2：<span class=](../../../arxiv/foundations/deformable_conv/extracted/figures/deform_conv_layer_v7.png)3\times3 可变形卷积结构示意。" />

### 2.2 可变形 RoI 池化

RoI 池化将任意大小矩形区域转为固定尺寸特征，是基于区域提议的检测方法的标准组件。

**标准 RoI 池化：** 给定特征图与大小为 $w\times h$、左上角为 $\mathbf{p}_0$ 的 RoI，划分为 $k\times k$ 个 bin，输出 $k\times k$ 特征图。第 $(i,j)$ 个 bin：

$$
\mathbf{y}(i,j)=\sum_{\mathbf{p}\in\mathrm{bin}(i,j)}\mathbf{x}(\mathbf{p}_0+\mathbf{p})/n_{ij}.
$$

**可变形 RoI 池化：** 为各 bin 增加偏移 $\{\Delta\mathbf{p}_{ij}\}$：

$$
\mathbf{y}(i,j)=\sum_{\mathbf{p}\in\mathrm{bin}(i,j)}\mathbf{x}(\mathbf{p}_0+\mathbf{p}+\Delta\mathbf{p}_{ij})/n_{ij}.
$$

同样用双线性插值。偏移获取方式：先做标准 RoI 池化，再经全连接层得到**归一化偏移** $\Delta\widehat{\mathbf{p}}_{ij}$，再与 RoI 宽高相乘：

$$
\Delta\mathbf{p}_{ij}=\gamma\cdot\Delta\widehat{\mathbf{p}}_{ij}\circ(w,h),
$$

其中 $\gamma=0.1$（经验设定）。归一化使偏移学习对 RoI 尺寸不变。

**可变形位置敏感（PS）RoI 池化：** 对应 R-FCN。先用卷积得到各类的 $k^2$ 张 score map，再在对应 score map 上池化；偏移学习亦遵循全卷积精神——先卷积得到全分辨率偏移场，再对每个 RoI（及每类）做 PS RoI 池化得到归一化偏移，再按上式还原。

![图3：<span class=](../../../arxiv/foundations/deformable_conv/extracted/figures/deform_pool_layer_v6.png)3\times33\times3 可变形 RoI 池化。" />

![图4：<span class=](../../../arxiv/foundations/deformable_conv/extracted/figures/deform_pspool_layer_v4.png)3\times33\times3 可变形 PS RoI 池化。" />

### 2.3 可变形 ConvNets

两模块与普通版本输入输出一致，可直接替换。训练时，偏移分支的卷积/全连接层**零初始化**；其学习率设为已有层的 $\beta$ 倍（默认 $\beta=1$；Faster R-CNN 中 fc 偏移层 $\beta=0.01$）。通过双线性插值 BP 训练。

**特征提取中的可变形卷积：** 采用 ImageNet 预训练的 ResNet-101 与 Aligned-Inception-ResNet。去掉平均池化与分类 fc，末尾加随机初始化的 $1\times1$ 卷积将通道压至 1024；末块有效步长由 32 改为 16（stride $2\to1$，同时将该块中核大小 $>1$ 的卷积膨胀率 $1\to2$）。可选地，对最后若干层 $3\times3$ 卷积使用可变形卷积；实验表明 **3 层**是较好折中（见表 1）。

**任务头：**

- **DeepLab**：在特征图上加 $1\times1$ 卷积得到 $(C+1)$ 张逐像素得分图 + softmax。
- **Category-Aware RPN**：类似 Faster R-CNN 的 RPN，但分类器改为 $(C+1)$ 类。
- **Faster R-CNN**：RPN 接在 conv4 顶；RoI 池化放在最后（简化设计），其上两个 1024 维 fc + 分类/回归。可选替换为可变形 RoI 池化。
- **R-FCN**：可选将其 RoI 池化换为可变形 PS RoI 池化。

---

## 3. 理解可变形 ConvNets

核心思想：在卷积与 RoI 池化的空间采样位置上增加偏移，并从目标任务学习偏移。

**堆叠可变形卷积**时，复合形变效果显著（图 5）：标准卷积在整张特征图上感受野与采样位置固定；可变形卷积则按物体尺度与形状自适应调整。图 6、表 2 给出更多可视化与定量证据。

**可变形 RoI 池化**效果类似（图 7）：规则网格不再成立，各 part 偏离 bin 移向附近前景，定位能力增强，尤其对非刚性物体。

![图5：两层标准卷积 vs 可变形卷积的感受野对比。](../../../arxiv/foundations/deformable_conv/extracted/figures/standard_conv_receptive_field_v6.png)

![图6：背景 / 小物体 / 大物体上三层可变形滤波的采样点可视化（每图 <span class=](../../../arxiv/foundations/deformable_conv/extracted/figures/demo_of_deform_conv_v2.png)9^3=7299^3=729 个红点）。" />

![图7：可变形 PS RoI 池化中各 part 偏移以覆盖非刚性物体。](../../../arxiv/foundations/deformable_conv/extracted/figures/demo_of_deform_pool_v4.png)

### 3.1 相关工作语境

**STN：** 首次在深度学习框架中学习空间变换；用**全局参数化变换**（如仿射）对特征图做 warp，计算贵、参数难学，主要在小规模分类上成功。可变形卷积的偏移学习可视为**极轻量**的空间变换器，但**不做全局参数化 warp**，而是**局部、稠密采样**，再加权求和生成新特征；易嵌入任意 CNN，适用于检测/分割等（半）稠密预测——这些任务对 STN 很难甚至不可行。

**Active Convolution（同期）：** 同样为卷积采样加偏移并端到端学习，在分类上有效。与可变形卷积的关键差异：① 偏移在**所有空间位置共享**；② 偏移是**静态模型参数**（按任务/训练学死）。可变形卷积的偏移是**动态模型输出**，随图像、随位置变化，适合（半）稠密预测。

**有效感受野：** 理论感受野内并非所有像素贡献相等，有效感受野远小于理论值，且随层数以 $\sqrt{\cdot}$ 变慢增长——高层有效感受野可能仍不够大，部分解释了空洞卷积的广泛使用，也说明需要**自适应感受野学习**。可变形卷积正具备此能力。

**空洞卷积（atrous / dilated）：** 通过增大采样间距扩大感受野且不增参数与计算。可变形卷积是其**泛化**（图 1c）；表 3 给出大量对比。

**DPM：** 可变形 RoI 池化与之类似（学习部件空间形变以提升分类分），但更简单（不考虑部件间空间关系）。DPM 浅且训练非端到端；可变形 ConvNets 深层、端到端，堆叠后形变建模更强。

**DeepID-Net：** 含形变约束池化，精神相近但更复杂、高度工程化，基于 RCNN，难以端到端接到 Faster R-CNN / R-FCN。

**RoI 池化中的空间布局：** 空间金字塔池化用手工多尺度区域。可变形 RoI 池化是 CNN 中**首个端到端学习池化区域**的方法（当前同尺寸，扩展到多尺寸直接）。

**变换不变特征及其学习：** SIFT、ORB 及大量 CNN 不变性工作通常假定变换**先验已知**，用手工艺或可学参数编码该先验，无法处理新任务中的未知变换。可变形模块可概括多种变换（图 1），不变性从目标任务学习。

**Dynamic Filter：** 滤波器权重也依赖输入、随样本变化；但学的是**权重**而非**采样位置**。本文学采样位置。

**低层滤波器组合：** Steerable Filters 等通过组合基滤波器得到新变换；有工作用高斯导数加权组合正则化滤波器空间。与本文相关在于多尺度组合可产生复杂核，但可变形卷积学的是**采样位置**而非滤波器权重。

---

## 4. 实验

### 4.1 设置与实现

**语义分割：** PASCAL VOC（mIoU@V）、Cityscapes（mIoU@C）。短边分别缩到 360 / 1024；SGD，8 GPU，VOC / Cityscapes 各 30k / 45k 迭代；学习率前 2/3 为 $10^{-3}$，后 1/3 为 $10^{-4}$。

**目标检测：** VOC（mAP@0.5 / @0.7）、COCO（mAP@[0.5:0.95] 与 mAP@0.5）。短边 600；VOC 上为便于消融，Faster R-CNN / R-FCN 使用预训练固定 RPN 提议且不共享特征；COCO 上联合训练并共享特征。

### 4.2 消融实验

**可变形卷积层数（表 1）：** 使用层数增加则精度稳步提升；DeepLab 在 3 层饱和，其他任务 6 层略优。后续默认特征提取用 **3 层**。

| usage of deformable convolution (# layers) | DeepLab mIoU@V / @C   | class-aware RPN mAP@0.5 / @0.7 | Faster R-CNN mAP@0.5 / @0.7 | R-FCN mAP@0.5 / @0.7  |
| ------------------------------------------ | --------------------- | ------------------------------ | --------------------------- | --------------------- |
| none (0, baseline)                         | 69.7 / 70.4           | 68.0 / 44.9                    | 78.1 / 62.1                 | 80.0 / 61.8           |
| res5c (1)                                  | 73.9 / 73.5           | 73.5 / 54.4                    | 78.6 / 63.8                 | 80.6 / 63.0           |
| res5b,c (2)                                | 74.8 / 74.4           | 74.3 / 56.3                    | 78.5 / 63.3                 | 81.0 / 63.8           |
| **res5a,b,c (3, default)**           | **75.2 / 75.2** | 74.5 / 57.2                    | 78.6 / 63.3                 | 81.4 / 64.7           |
| res5 & res4b22–b20 (6)                    | 74.8 / 75.1           | **74.6 / 57.7**          | **78.7 / 64.0**       | **81.5 / 65.4** |

**有效膨胀（表 2）：** 定义为滤波中相邻采样点距离的均值，粗略度量感受野大小。按 GT 框与滤波中心将采样分为 small / medium / large / background：

| layer | small         | medium        | large         | background    |
| ----- | ------------- | ------------- | ------------- | ------------- |
| res5c | $5.3\pm3.3$ | $5.8\pm3.5$ | $8.4\pm4.5$ | $6.2\pm3.0$ |
| res5b | $2.5\pm1.3$ | $3.1\pm1.5$ | $5.1\pm2.5$ | $3.2\pm1.2$ |
| res5a | $2.2\pm1.2$ | $2.9\pm1.3$ | $4.2\pm1.6$ | $3.1\pm1.1$ |

结论：① 感受野大小与物体尺度相关，形变从内容有效学到；② 背景需要介于中大物体之间的较大感受野。

**对比空洞卷积（表 3）：**

| deformation modules                     | DeepLab mIoU@V / @C   | class-aware RPN mAP@0.5 / @0.7 | Faster R-CNN mAP@0.5 / @0.7 | R-FCN mAP@0.5 / @0.7  |
| --------------------------------------- | --------------------- | ------------------------------ | --------------------------- | --------------------- |
| atrous (2,2,2) default                  | 69.7 / 70.4           | 68.0 / 44.9                    | 78.1 / 62.1                 | 80.0 / 61.8           |
| atrous (4,4,4)                          | 73.1 / 71.9           | 72.8 / 53.1                    | 78.6 / 63.1                 | 80.5 / 63.0           |
| atrous (6,6,6)                          | 73.6 / 72.7           | 73.6 / 55.2                    | 78.5 / 62.3                 | 80.2 / 63.5           |
| atrous (8,8,8)                          | 73.2 / 72.4           | 73.2 / 55.1                    | 77.8 / 61.8                 | 80.3 / 63.2           |
| **deformable convolution**        | **75.3 / 75.2** | **74.5 / 57.2**          | 78.6 / 63.3                 | 81.4 / 64.7           |
| deformable RoI pooling                  | N.A.                  | N.A.                           | 78.3 / 66.6                 | 81.2 / 65.0           |
| **deformable conv & RoI pooling** | N.A.                  | N.A.                           | **79.3 / 66.9**       | **82.6 / 68.5** |

观察：① 增大固定膨胀率普遍涨点 → 默认感受野偏小；② 最优膨胀率因任务而异（DeepLab 偏 6，Faster R-CNN 偏 4）；③ **可变形卷积全面最优**。可变形 RoI 单独使用在严格 **mAP@0.7** 上提升明显；两者联用提升更大。

**复杂度与运行时（表 4）：** 参数与计算开销很小（如 DeepLab 46.0M→46.1M），说明涨点来自几何建模能力而非堆参数。

### 4.3 COCO 目标检测

| method          | backbone                 | M  | B  | mAP@[0.5:0.95] | mAP@0.5        | small          | mid            | large          |
| --------------- | ------------------------ | -- | -- | -------------- | -------------- | -------------- | -------------- | -------------- |
| class-aware RPN | ResNet-101               |    |    | 23.2           | 42.6           | 6.9            | 27.1           | 35.1           |
| **Ours**  | ResNet-101               |    |    | **25.8** | **45.9** | **7.2**  | **28.3** | **40.7** |
| Faster R-CNN    | ResNet-101               |    |    | 29.4           | 48.0           | 9.0            | 30.5           | 47.1           |
| **Ours**  | ResNet-101               |    |    | **33.1** | **50.3** | **11.6** | **34.9** | **51.2** |
| R-FCN           | ResNet-101               |    |    | 30.8           | 52.6           | 11.8           | 33.9           | 44.8           |
| **Ours**  | ResNet-101               |    |    | **34.5** | **55.0** | **14.0** | **37.7** | **50.3** |
| Faster R-CNN    | Aligned-Inception-ResNet |    |    | 30.8           | 49.6           | 9.6            | 32.5           | 49.0           |
| **Ours**  | Aligned-Inception-ResNet |    |    | **34.1** | **51.1** | **12.2** | **36.5** | **52.4** |
| R-FCN           | Aligned-Inception-ResNet |    |    | 32.9           | 54.5           | 12.5           | 36.3           | 48.3           |
| **Ours**  | Aligned-Inception-ResNet |    |    | **36.1** | **56.7** | **14.8** | **39.8** | **52.2** |
| R-FCN           | Aligned-Inception-ResNet | ✓ |    | 34.5           | 55.0           | 16.8           | 37.3           | 48.3           |
| **Ours**  | Aligned-Inception-ResNet | ✓ |    | 37.1           | 57.3           | 18.8           | 39.7           | 52.3           |
| R-FCN           | Aligned-Inception-ResNet | ✓ | ✓ | 35.5           | 55.6           | 17.8           | 38.4           | 49.3           |
| **Ours**  | Aligned-Inception-ResNet | ✓ | ✓ | **37.5** | **58.0** | **19.4** | **40.1** | **52.5** |

（M = multi-scale testing；B = iterative bounding box average。）

ResNet-101 上，可变形版 class-aware RPN / Faster R-CNN / R-FCN 的 mAP@[0.5:0.95] 相对提升约 **11% / 13% / 12%**。换更强骨干后增益仍成立；多尺度 + 框平均后可变形 R-FCN 达 **37.5**，且增益与这些 tricks **互补**。

---

## 5. 结论

本文提出可变形 ConvNets：一种简单、高效、深层、端到端的稠密空间变换建模方案。首次表明：在 CNN 中学习稠密空间变换，对目标检测与语义分割等复杂视觉任务是可行且有效的。

---

## 附录 A. 反向传播（摘要）

对偏移 $\Delta\mathbf{p}_n$ 的梯度经双线性核 $G$ 对偏移的偏导回传：

$$
\frac{\partial\mathbf{y}(\mathbf{p}_0)}{\partial\Delta\mathbf{p}_n}
=\sum_{\mathbf{p}_n\in\mathcal{R}}
\left[
\mathbf{w}(\mathbf{p}_n)\cdot
\sum_{\mathbf{q}}
\frac{\partial G(\mathbf{q},\mathbf{p}_0+\mathbf{p}_n+\Delta\mathbf{p}_n)}{\partial\Delta\mathbf{p}_n}
\mathbf{x}(\mathbf{q})
\right].
$$

可变形 RoI 池化对 $\Delta\mathbf{p}_{ij}$ 的梯度形式类似；再经 $\Delta\mathbf{p}_{ij}=\gamma\cdot\Delta\widehat{\mathbf{p}}_{ij}\circ(w,h)$ 得到归一化偏移的梯度。

---

## 附录 B. Aligned-Inception-ResNet（摘要）

原始 Inception-ResNet 多层 valid 卷积/池化会导致特征与图像位置**不对齐**，不利于稠密预测。Aligned-Inception-ResNet 通过适当 padding 消除对齐问题，并采用更简单的重复 IRB 模块。ImageNet-1K 上：ResNet-101 top-1 err 23.6%；Aligned-Inception-ResNet 22.1%（参数 64.3M）。

---

## 术语对照（便于汇报）

| 英文                            | 中文                               |
| ------------------------------- | ---------------------------------- |
| deformable convolution          | 可变形卷积                         |
| deformable RoI pooling          | 可变形兴趣区域池化                 |
| offset                          | 偏移                               |
| bilinear interpolation          | 双线性插值                         |
| atrous / dilated convolution    | 空洞卷积                           |
| receptive field                 | 感受野                             |
| effective dilation              | 有效膨胀（相邻采样点平均距离）     |
| Spatial Transform Network (STN) | 空间变换网络                       |
| Active Convolution              | 主动卷积（静态、位置共享偏移）     |
| Dynamic Filter                  | 动态滤波器（动权重、不动采样位置） |

---

## 一句话总结（个人理解）

> **标准卷积的采样网格是死的；可变形卷积让每个采样点长出可学习的**脚**，从检测/分割任务里端到端学会按内容弯折感受野——几乎不增参数，却把固定几何变成输入依赖的几何。**
