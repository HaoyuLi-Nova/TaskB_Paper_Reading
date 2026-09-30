# Free-Form Motion Control：视频生成中相机与物体 6D 姿态的自由形式控制

**作者：** Xincheng Shuai$^{1}$、Henghui Ding$^{1}$、Zhenyuan Qin$^{1}$、Hao Luo$^{2,3}$、Xingjun Ma$^{1}$、Dacheng Tao$^{4}$

**机构：** $^{1}$复旦大学；$^{2}$阿里巴巴集团达摩院（DAMO Academy）；$^{3}$湖畔实验室（Hupan Lab）；$^{4}$南洋理工大学

**出处：** ICCV 2025（arXiv:2501.01425）

**arXiv：** [2501.01425](https://arxiv.org/abs/2501.01425)

**项目页：** https://henghuiding.com/SynFMC/

**原文 TeX：** [`arxiv/geometry_control/free_form_motion_control/extracted/main.tex`](../../../arxiv/geometry_control/free_form_motion_control/extracted/main.tex)

---

## 摘要

在生成视频中控制动态物体与相机的运动，是一项有意义但颇具挑战的任务。由于缺乏带完整 6D 姿态标注的数据集，现有文本到视频（text-to-video, T2V）方法无法以 3D-aware 的方式同时控制相机与物体运动，从而限制了对生成内容的可控性。为解决这一问题并推动该方向研究，我们引入 **Syn**thetic Dataset for **F**ree-Form **M**otion **C**ontrol（**SynFMC**）。所提出的 SynFMC 覆盖多样的物体与环境类别，并按特定规则涵盖多种运动模式，以模拟常见且复杂的真实场景。完整的 6D 姿态信息有助于模型学习把视频中来自物体与相机的运动效应解耦。为提供精确的 3D-aware 运动控制，我们进一步提出在 SynFMC 上训练的方法 **Free-Form Motion Control（FMC）**。FMC 可独立或同时控制物体与相机的 6D 姿态，并生成高保真视频。此外，它可与多种个性化文本到图像（text-to-image, T2I）模型兼容，以支持不同内容风格。大量实验表明，所提出的 FMC 在多种场景下优于先前方法。

---

![图 1. 所提出的 Synthetic Dataset for Free-Form Motion Control（SynFMC）中，基于规则的视频生成管线。本例生成含三个物体的合成视频：（1）选择环境资产及其匹配的物体资产作为场景元素；（2）随机选择物体与相机的运动类型以生成轨迹；（3）中心区域为用于渲染的 3D 动画序列。末行展示渲染视频与标注。](../../../arxiv/geometry_control/free_form_motion_control/extracted/teaser_compressed.png)

---

## 1. 引言

视频生成中的运动动态控制日益受到关注（CameraCtrl、MotionCtrl、HumanVid、AnimateDiff、VMC、STDF、MotionClone、DreamMotion、MotionInversion、MotionMaster 等），因为它能带来更好的定制能力，并在许多应用中至关重要。例如在电影制作中，导演会精心编排演员与相机的运动。因此，对视频中物体与相机运动的精确控制可提供创作灵活性。

尽管近期进展显著，T2V 生成中的运动控制仍有挑战。一个关键局限是 **缺乏带完整 6D 姿态标注的高质量数据集**。
在物体运动控制方面（MotionCtrl、AnyI2V、VideoComposer、DragNUWA、Tora、Trailblazer、FreeTraj、Motion-Zero、MotionBooth、Image-Conductor、Motion-I2V），运动主要被标注为图像空间中的轨迹。这种标注缺少 3D 本质，并把物体动态与相机动态缠在一起。例如，一条向右的轨迹既可以表示相机静止、物体右移，也可以表示物体静止、相机左移。近期 $\mathrm{360}^{\circ}$-Motion 合成数据集（3DTrajMaster）提供了物体的 6D 姿态，但限于静态相机设定，运动多样性也有限。
此外，现有常用于学习相机运动的数据集（Direct-a-Video、RealEstate-10K、MVImageNet）主要关注物体动态极少的场景。
一些以人为中心的合成数据集（HumanVid、BEDLAM、SynBody、WHAC）在全局坐标系中同时提供人体与相机运动的真值，但运动多样性与类别多样性仍然有限。

另一局限是 **缺少能独立或联合控制物体与相机 6D 姿态的方法**。例如，Motion-Zero 一类方法可以动画化物体，但没有 3D-aware 控制（如朝向）（PEEKABOO、MotionBooth、Trailblazer）；CameraCtrl 一类方法则只关注相机运动（Training-free-CC、CamCo、VD3D）。MotionCtrl 以两阶段分别训练物体控制与相机控制模块。然而，由于没有同时包含二者完整 6D 姿态标注的视频数据，它难以在同一场景中实现真实、同步的物体与相机控制。

![图 2. 本方法 FMC 在所提出的 SynFMC 上训练后生成的示例视频，展示其与不同个性化 T2I 模型（Stable Diffusion、Toonyou、RealisticVision）的适配能力。](../../../arxiv/geometry_control/free_form_motion_control/extracted/personalized_compressed.png)

为应对上述局限，需要一份同时包含物体与相机完整 6D 姿态标注的数据集。然而获取此类数据困难，通常需要专用设备与专业知识。本文引入 **Syn**thetic dataset for **F**ree-Form **M**otion **C**ontrol（**SynFMC**）。该数据集强调质量与多样性，包含丰富的动画物体资产与跨类别环境资产。进一步地，我们实现基于规则的生成算法，为物体与相机创建轨迹，如图 1 所示。该算法覆盖基本模式，并模拟具有挑战性的电影镜头，见图 5。相较近期只能构造不可控轨迹的工作（3DTrajMaster、HumanVid），我们的基于规则算法支持可定制的物体与相机运动，并覆盖多样模式。为增强真实感，物体的关键属性（如生存环境、速度类型、尺寸类型）由多模态大语言模型（MLLM，InternVL）与人工标注共同完成，以便生成合理轨迹。SynFMC 还提供详细标注，包括物体与相机的 6D 姿态、实例分割图、深度图，以及内容与运动的完整描述，可支持广泛研究方向。

为进一步验证所提出 SynFMC 的有效性，并支持 T2V 生成中的 3D-aware 控制，我们提出 **Free-Form Motion Control（FMC）**。FMC 主要包括两个组件：**Camera Motion Controller（CMC）** 与 **Object Motion Controller（OMC）**。与先前方法（DragNUWA、MotionCtrl）不同，本方法在 SynFMC 上训练，解耦全局（相机）与局部（物体）动态，并操控相机与物体的 6D 姿态。此外，我们采用 Domain LoRA，防止模型拟合合成数据的渲染风格。如图 2 与图 6 所示，FMC 能有效缓解域差距，并适配多种个性化 T2I 模型，在多样风格下生成高保真结果。FMC 还提供灵活的运动控制用户接口：用户可通过绘制 3D 曲线输入物体与相机轨迹，或为二者指定运动类型（详见第 3.3 节），再由基于规则的算法据此生成轨迹。

**表 1. 所提出 SynFMC 与现有数据集的比较。** 物体/相机运动模式列仅适用于合成数据集。除物体类别丰富外，SynFMC 在运动模式多样性与可控性上更优，并提供相机与物体的完整姿态标注。实现中我们仅使用 26K 子集作为训练数据。

| Dataset | Clips | Source | Category | Object Motion Pattern | Camera Motion Pattern | Camera Pose Annotation | Object Motion Annotation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RealEstate10K | 65K | Real | Real Estate | - | - | Fitting | ✗ |
| MVImgNet | 220K | Real | **Common** | - | - | Fitting | ✗ |
| VideoHD | 75K | Real | **Common** | - | - | ✗ | Optical Flow |
| MotionCtrl | **240K** | Real | **Common** | - | - | ✗ | Optical Flow |
| HumanVid-Real | 20K | Real | Human | - | - | Fitting | 2D Human Pose |
| BEDLAM | 10K | Synthetic | Human | Limited/Uncontrollable | Static | **Ground Truth** | 3D Human Pose |
| SynBody | 27K | Synthetic | Human | Limited/Uncontrollable | Static | **Ground Truth** | 3D Human Pose |
| HumanVid-Syn | 100K | Synthetic | Human | Limited/Uncontrollable | **Diverse**/Uncontrollable | **Ground Truth** | 3D Human Pose |
| $\mathrm{360}^{\circ}$-Motion | 54K | Synthetic | Animal & Human | Limited/Uncontrollable | Static | ✗ | **Object Pose** |
| **SynFMC（ours）** | 62K | Synthetic | **Common** | **Diverse/Controllable** | **Diverse/Controllable** | **Ground Truth** | **Object Pose** |

总结而言，本文主要贡献如下：

- 据我们所知，**SynFMC** 是首个同时为相机与物体提供 6D 姿态标注的数据集。其多样场景与复杂运动模式，为模型学习多物体与相机动态提供了有价值的资源。

- **FMC** 可独立或同时操控相机与物体的 6D 姿态，并在多样场景中取得高质量结果。

- 大量实验表明，在 SynFMC 上训练的 FMC 相较 state-of-the-art 方法能生成更高质量的视频。

---

## 2. 相关工作

**带运动标注的数据集。** 多数数据集（MotionCtrl、DragNUWA）聚焦于图像空间中的操作。然而，相机运动与物体运动在该空间中是耦合的，同时运动范围也受限。另一方面，仅有少数真实数据集（Direct-a-Video、RealEstate-10K、MVImageNet）提供相机姿态，且主要关注没有动态物体的静态场景。一些合成数据集（SynBody、BEDLAM、WHAC、HumanVid）为物体与相机同时提供姿态标注，但其 3D 资产以人为中心，类别多样性有限。更多讨论见第 3.1 节。

**运动控制方法。** 多数现有工作（CameraCtrl、PEEKABOO、CamCo、Tora、3DTrajMaster）只能控制物体运动或视角变化二者之一。对于同时支持操控相机与物体的方法，Direct-a-Video 只能模拟基本运动；MotionCtrl 中的同时控制往往效果不佳，其论文亦有指出。更多讨论见第 4 节。

---

## 3. SynFMC Dataset

### 3.1 与现有数据集的比较

目前仍缺少同时包含物体与相机 6D 姿态的数据集（RealEstate-10K、MVImageNet、HumanVid）。如表 1 所示，仅有少数真实数据集（RealEstate-10K、MVImageNet）提供估计的相机姿态，且由于估计方法（COLMAP、DROID-SLAM、ParticleSfM）表现不佳，它们主要限于没有动态物体的场景。部分方法（DragNUWA、MotionCtrl）使用野外视频，并用光流模型（ParticleSfM）推断图像空间中的物体轨迹，但这会把物体运动与相机运动纠缠在一起，同时缺少朝向等 3D 信息。合成数据集便于获得姿态信息（BEDLAM、WHAC、HumanVid、SynBody），但多数聚焦人体，且限于小幅度运动与简单相机运动模式，从而限制了学习复杂动态的能力。

为促进模型以 3D-aware 方式学习运动控制，有必要构建一份同时包含完整物体与相机姿态的新数据集。这在真实世界中极具挑战。首先，拍摄包含复杂、不规则物体与相机运动的视频极为困难，通常需要专用设备与专业知识。其次，获得准确姿态估计同样困难。能够捕获相机或物体 6D 姿态的设备昂贵且不易操作。一些研究（RealEstate-10K、MotionCtrl）试图用估计模型得到相机姿态，但现有算法（ParticleSfM、DROID-SLAM）耗时长，且常难以处理含动态物体的单目视频。此外，对一般物体推断 6D 姿态仍然困难。为应对这些局限，我们引入 **SynFMC**：一份用 Unreal Engine 生成的合成数据集，包含多样运动模式的动画以及完整标注。

- **与 $\mathrm{360}^{\circ}$-Motion Dataset（3DTrajMaster）的区别。** **（1）** 我们同时处理静态与动态相机，能实现比 3DTrajMaster 静态设定更复杂的镜头。**（2）** 我们的基于规则算法支持多样的水平与非水平物体运动，而 3DTrajMaster 由 GPT 导出的轨迹仅为水平运动。**（3）** 我们的环境与物体资产超出 3DTrajMaster 的地面场景。

- **与 HumanVid-Syn Dataset 的区别。** **（1）** HumanVid 的物体轨迹依赖预定义 3D 运动资产（SMPL-X / skeleton），而我们的基于规则算法能产生多样模式，例如图 3 中的原地运动、水平与非水平运动。**（2）** HumanVid 在人体前方半圆柱内随机采样相机位置；我们通过解耦相机运动（图 4）实现细粒度控制，支持可控且多样的运动。**（3）** HumanVid 聚焦单人动画合成；我们支持多物体场景，从而生成更丰富的动态。

### 3.2 SynFMC Dataset 概览

SynFMC 含 62K 视频，分为四组：15K **static single-object**、15K **static multi-object**、16K **dynamic single-object**、16K **dynamic multi-object**。
**Static** 表示物体在世界空间中位置固定，相机仍可运动。为追求多样性，数据集包含跨类别的常见物体（人、动物、植物、载具等），以及街道、草地、天空、海洋等广泛环境。此外，SynFMC 具有多样且复杂的多物体与相机运动，不仅覆盖基本运动，也包括现实中难以拍摄的镜头。物体资产由人工标注员在 MLLM（InternVL）辅助下标注速度、尺寸等属性，以保证运动仿真合理。

- **视频标注。** 所提出的 SynFMC 提供完整标注，包括物体与相机的姿态信息、实例分割 mask、深度图，以及内容与运动的详细描述，从而拓宽其在多个领域（MOSE、MeViS、CharaConsist）中的适用性。

![图 3. 物体运动类型。stationary point 的轨迹未在图中画出，它是空间中的固定点。](../../../arxiv/geometry_control/free_form_motion_control/extracted/3.1_compressed.png)

### 3.3 数据生成管线

- **资产收集与标注。** 我们收集多样 3D 资产，包括环境与物体。环境资产涵盖五类：**ground**、**near ground**、**sky**、**water surface**、**underwater**。我们从互联网收集高质量全景 HDRI 图像。物体资产选自 Objaverse-LVIS、Objaverse-XL、Mixamo 等带动画的资产，覆盖多样类别。人工标注员过滤低质量资产，并核验/修正 InternVL 查询得到的物体属性（如类别、栖息地、速度、尺寸）。他们还为每个动画标注运动类型并给出描述。

- **物体运动。** 为创建真实运动，我们在每个运动片段中基于 Bézier 曲线设计轨迹，旋转由沿曲线的切向量与法向量导出。图 3 给出运动类型示例。控制点根据物体速度加以约束。

- **相机运动。** 我们将相机运动分解为三类：**viewpoint**、**distance**、**height**，见图 4。（a）**Viewpoint** 控制拍摄物体时的相机朝向：front/back、left/right、top。每个运动片段的起始与结束 viewpoint 随机指定，中间帧插值以保证平滑过渡，从而使相机运动范围覆盖物体四周的全部朝向。（b）**Distance** 控制相机与物体的水平距离：zoom in/out 与 static。（c）**Height** 控制垂直距离：up/down 与 static。为在保持物体可见的同时不强制将其置于画面中心，相机瞄准物体质心附近一个随机偏移点。

![图 4. 相机运动类型。我们将相机运动分解为三个方面。（a）Viewpoint 控制拍摄物体时的相机朝向。①–⑤ 分别表示 front/back、left/right 与 top 视角。（b）Distance 与（c）Height 分别决定相机与物体的水平与垂直距离。⑥–⑨ 分别为 zoom in/out 与 up/down。（b）（c）中省略了 static 类型，表示距离固定。](../../../arxiv/geometry_control/free_form_motion_control/extracted/cam_type_compressed.png)

**表 2. FMC 与其他方法的比较。** FMC 擅长以多样运动模式控制物体与相机的 6D 姿态。

| Methods | Motion Condition | Object Control | Camera Control | Dataset |
| --- | --- | --- | --- | --- |
| AnimateDiff | ✗ | ✗ | Limited Patterns | Dataset for Specific Motion Pattern |
| CameraCtrl | **Camera Pose** | ✗ | **Diverse Patterns** | RealEstate10K |
| VideoComposer | Image Space Trajectory | Entanglement | Entanglement | LAION-400M + WebVid |
| DragNUWA | Image Space Trajectory | Entanglement | Entanglement | WebVid + VideoHD |
| 3DTrajMaster | **Object Pose** | Limited Patterns | ✗ | $\mathrm{360}^{\circ}$-Motion |
| Direct-a-Video | Image Space Trajectory + Camera Type | Entanglement | Limited Patterns | MovieShot |
| MotionCtrl | Image Space Trajectory + **Camera Pose** | Entanglement | **Diverse Patterns** | RealEstate10K + WebVid |
| **FMC（ours）** | **Object Pose + Camera Pose** | **Diverse Patterns** | **Diverse Patterns** | **SynFMC（ours）** |

![图 5. 运动轨迹。水平与非水平运动示例表明，基于规则的算法所生成轨迹具有多样性与复杂性。](../../../arxiv/geometry_control/free_form_motion_control/extracted/curves.png)

- **多物体场景生成。** 选择同一环境、尺寸与速度相当的物体。第一个物体的轨迹按前述方式创建，后续轨迹在前一物体路径上加以合理偏移。相机运动方面，我们在每个运动片段随机跟踪一个物体，轨迹生成方法与前述相同。

- **渲染。** 本阶段根据所选资产以及物体与相机的运动类型创建合成视频。我们将完整运动轨迹划分为许多小片段。对每个片段，随机选择物体动画，并按其标注的运动类型生成轨迹。类似地，不同片段随机组合相机运动类型，使生成视频能像真实场景那样涵盖多样运动模式。这些片段被无缝组合，形成复杂多样的全局轨迹，如图 5 所示。最后，将所选资产以及相机与物体姿态导入 Unreal Engine，得到用于渲染的 3D 动画序列。图 1 给出一个轨迹片段示例。

---

## 4. 所提出的方法

给定长度为 $N$ 的相机姿态 $\mathcal{C}_{RT}=RT_{\mathrm{cam}}^{1:N}$、全局坐标系中 $N_o$ 个物体的姿态 $\mathcal{O}_{RT}=\{RT_{\mathrm{obj}_i}^{1:N}\}_{i=1}^{N_o}$，以及内容描述 $\mathbf{C}_p$，我们的目标是生成能正确呈现真实世界运动的视频。

如表 2 所示，多数方法无法以 3D-aware 方式独立或联合控制物体与相机运动。例如，AnimateDiff 与 CameraCtrl 只支持相机控制。使用图像空间轨迹的方法（VideoComposer、DragNUWA、Direct-a-Video、MotionCtrl）面临运动纠缠问题，也无法控制朝向。尽管 Direct-a-Video 引入若干相机类型并允许显式控制，它仍限于简单运动模式。最接近本文的 MotionCtrl 分别训练两个运动模块，但缺少完整 6D 姿态标注，导致相机与物体运动的同时控制效果不佳。此外，它在训练运动模块时仅使用标准 diffusion loss，进一步阻碍了在视频中解耦相机运动与物体运动。在 SynFMC 上训练的 FMC 引入 Camera Motion Controller（CMC）与 Object Motion Controller（OMC）以应对这些局限。OMC 接收物体的 6D 姿态与粗 mask，以感知其空间位置与朝向，从而在不同视点下得到真实外观。训练目标使 FMC 能解耦视频中物体与相机的运动效应，从而独立或联合地对相机与物体运动做 3D-aware 控制。

- **预备。** **（1）** T2V 扩散模型（AnimateDiff、Imagen-Video、I2VGen-XL、MAGVIT、Make-A-Video、VDM）在训练时向图像序列 $\mathbf{z}_0^{1:N}$ 加入高斯噪声 $\epsilon$，得到时间步 $t$ 的带噪 latent $\mathbf{z}_t^{1:N}$。网络 $\varepsilon_{\theta}$ 随后被训练从当前 latent 推断所注入噪声。**（2）** LoRA（含 AnimateDiff 中的用法）用于学习不同内容风格。我们用它弥合合成视频与真实视频之间的域差距。**（3）** 遵循 CameraCtrl，我们用 Plücker embedding 表示相机姿态，以便几何解释。

![图 6. Domain LoRA。我们采样生成视频的首帧，对比不使用与使用 Domain LoRA 的设定。](../../../arxiv/geometry_control/free_form_motion_control/extracted/lora.png)

![图 7. FMC 的架构。第一阶段从合成视频中随机采样图像，并更新注入的 Domain LoRA 参数。随后学习 CMC 中的模块：它由 Camera Encoder 与 Camera Adapter 两部分组成，其中 Camera Adapter 引入到 temporal modules。最后训练 OMC 中的 Object Encoder。它接收 6D 物体姿态特征，并在对应物体区域内重复。我们使用以质心为中心的 Gaussian blur kernel，从而无需精确 mask。随后将输出与粗 mask 相乘，以调制主分支特征。](../../../arxiv/geometry_control/free_form_motion_control/extracted/network_compressed.png)

### 4.1 Free-Form Motion Control

图 7 给出所提出 FMC 方法的整体架构。我们分三阶段训练。首先，将 Domain LoRA 注入空间块以适配渲染内容，此时 temporal modules 不启用，图像从合成数据中随机采样。图 6 展示了该阶段在弥合域差距上的有效性。随后训练 CMC 以学习相机运动：引入 temporal modules，并加载上一阶段的 LoRA。最后训练 OMC，把物体动态从相机运动中解耦，其余参数冻结。推理时丢弃 LoRA 模块，以保持基座模型质量。

- **Camera Motion Controller（CMC）。** 如图 7 所示，它由两部分组成。Camera Encoder 接收 Plücker embeddings：第一帧使用初始相机姿态（平移置 0），后续帧使用相对相机姿态。初始姿态有助于确定起始时刻的透视。随后，输出经 Camera Adapter 处理，以调制 temporal blocks 中的特征。由于背景动态仅受相机运动影响，本阶段施加 **camera loss** $L_{\mathrm{cam}}$：

$$
\begin{aligned}
L_{\mathrm{cam}} = &\mathbb{E}_{\mathbf{z}_0^{1:N},t,\epsilon,\mathbf{C}_p,\mathcal{C}_{RT}}\Big[
\mathcal{M}_{bg}\bigl\|\varepsilon_{\theta,\theta_c}(\mathbf{z}^{1:N}_t, t, \mathbf{C}_p,\mathcal{C}_{RT})-\epsilon\bigr\|^2 \\
&+ \lambda_c \bigl\|\varepsilon_{\theta,\theta_c}(\mathbf{z}^{1:N}_t, t, \mathbf{C}_p,\mathcal{C}_{RT})-\epsilon\bigr\|^2 \Big],
\end{aligned}
\tag{1}
$$

其中 $\theta_c$ 为 CMC 的参数，$\mathcal{M}_{bg}$ 为背景 mask，$\lambda_c$ 为权重系数。$L_{\mathrm{cam}}$ 通过聚焦背景使相机运动更准确。

- **Object Motion Controller（OMC）。** OMC 的 Object Encoder 接收 6D 物体姿态信息，以调整若干 downsample blocks（Stable Diffusion、MotionCtrl）中空间模块的特征。具体地，每帧相对相机的姿态在对应物体区域内复制，其余位置置 0。随后将姿态特征与前景 mask 拼接后送入 OMC。这样，不同物体的姿态可聚合到单一输入中。此外，我们使用以物体质心为中心的 Gaussian blur kernel，避免用户提供精确 mask。然后，OMC 的输出与粗 mask 相乘，再加到主分支的空间特征上，以免损害背景内容。推理时，kernel 尺寸可根据物体尺寸（由用户指定）及其与相机的距离近似得到。施加 **object loss** $L_{\mathrm{obj}}$，使 OMC 聚焦于物体区域：

$$
\begin{aligned}
L_{\mathrm{obj}} = &\mathbb{E}_{\mathbf{z}_0^{1:N},t,\epsilon,\mathbf{C}_p,\mathcal{C}_{RT},\mathcal{O}_{RT}}\Big[
\mathcal{M}_{fg}\bigl\|\varepsilon_{\theta,\theta_c,\theta_o}(\mathbf{z}^{1:N}_t, t, \mathbf{C}_p,\mathcal{C}_{RT},\mathcal{O}_{RT})-\epsilon\bigr\|^2 \\
&+ \lambda_o \bigl\|\varepsilon_{\theta,\theta_c,\theta_o}(\mathbf{z}^{1:N}_t, t, \mathbf{C}_p,\mathcal{C}_{RT},\mathcal{O}_{RT})-\epsilon\bigr\|^2 \Big],
\end{aligned}
\tag{2}
$$

其中 $\theta_o$ 表示 OMC 的参数，$\mathcal{M}_{fg}$ 为前景 mask，$\lambda_o$ 为权重系数。$L_{\mathrm{obj}}$ 提升动态物体的外观质量。

---

## 5. 实验

- **实现细节。** 所提出的 FMC 基于 AnimateDiff V3，用长度为 16、分辨率为 $256\times 384$ 的视频训练，优化器为 Adam，学习率为 $1\times 10^{-4}$。Domain Adapter 训练 8K 次迭代，batch size 为 128。CMC 与 OMC 各训练 50K 次迭代，batch size 为 8。$\lambda_c$ 与 $\lambda_o$ 分别设为 0.6 与 0.3。实验中我们发现 26K 数据样本已足以让本方法学习 3D-aware 运动控制。

- **评价指标。** 遵循 AnimateDiff 与 MotionCtrl，我们用 FID 评估视觉质量，用 FVD 评估时间连贯性，用 CLIPSIM 衡量与文本的语义相似度。对相机运动，遵循 CameraCtrl 使用 CamTransErr 与 CamRotErr。对物体运动，我们引入 ObjTransErr 与 ObjRotErr。我们首先用深度估计模型 DepthAnythingV2 得到物体质心处的深度，再根据相机姿态确定其全局位置，然后拟合轨迹曲线以得到各时间步的切向量与法向量，从而导出旋转。在已知尺度信息的情况下，我们对平移误差计算施加适当缩放。

![图 8. 对相机运动与物体运动的独立控制。（a）表明所有方法（CameraCtrl、MotionCtrl）都能有效反映相机条件。对物体运动，对比方法（MotionCtrl、Direct-a-Video）无法保持相机静止，如（b）绿框所示（例如第 5 行花朵的移动）。此外，它们对物体朝向（条件中的 3D 坐标轴）的保真度也较低。](../../../arxiv/geometry_control/free_form_motion_control/extracted/indep_comparison_compressed.png)

### 5.1 与 State-of-the-Art 方法的比较

我们首先与先前方法（MotionCtrl、CameraCtrl）比较对相机运动与物体运动的独立控制，再展示 FMC 在同时控制上的更优表现，最后给出跨不同场景的更多例子，以验证 SynFMC 与 FMC 的有效性。为公平起见，我们与基于 U-Net 的方法比较。更多例子见补充材料。

- **相机运动的独立控制。** 本比较选用 MotionCtrl 与 CameraCtrl，因为它们接受显式相机信息。在图 8(a) 中，我们模拟两种相机运动，并将平移缩放到这些方法所需的输入范围。FMC 与对比方法都能有效反映输入条件。表 3 中的 CamTransErr 与 CamRotErr 也表明，FMC 在控制相机方面取得可比结果。

**表 3. 所提出方法 FMC 与 AnimateDiff、CameraCtrl、MotionCtrl 的定量比较。**

| Method | AnimateDiff | CameraCtrl | MotionCtrl | **FMC（ours）** |
| --- | --- | --- | --- | --- |
| **FID**  | 149.61 | 137.96 | **125.52** | 133.42 |
| **FVD**  | 868.97 | **805.25** | 952.31 | 846.51 |
| **CLIPSIM** $\uparrow$ | 29.33 | 29.21 | 26.83 | **31.01** |
| **CamTransErr**  | - | 18.16 | **17.84** | 18.12 |
| **CamRotErr**  | - | **0.94** | 1.11 | 1.03 |
| **ObjTransErr**  | - | - | 80.66 | **42.25** |
| **ObjRotErr**  | - | - | 1.77 | **0.96** |

![图 9. 对相机运动与物体运动的同时控制。MotionCtrl 难以生成真实的物体动态，导致物体离开视野；而我们的 FMC 实现了高质量的同时控制。](../../../arxiv/geometry_control/free_form_motion_control/extracted/simul_comparison_compressed.png)

- **物体运动的独立控制。** 对物体控制，我们与 MotionCtrl 和 Direct-a-Video 比较。对这些方法，我们用相机与物体姿态信息把全局轨迹投影到图像空间。如图 8(b) 所示，对比方法无法保持相机静止（背景出现动态运动），表明图像空间轨迹会把物体运动与相机运动纠缠在一起。例如图 8(b) 第 2 个例子中，Direct-a-Video 因动态相机导致花朵位置变化。本方法通过把静态相机姿态作为约束，有效缓解了该问题。此外，由于 OMC 接收物体的 6D 姿态，FMC 能按输入条件实现高保真的物体朝向。

**表 4. 质量、文本相似度与运动保真度的用户研究。**

| Method | CameraCtrl | MotionCtrl | **FMC（ours）** |
| --- | --- | --- | --- |
| **Quality Score** | 0.88 | 0.89 | **0.91** |
| **Text Similarity Score** | 0.84 | 0.81 | **0.95** |
| **Camera Motion Score** | **0.95** | 0.93 | **0.95** |
| **Object Motion Score** | - | 0.53 | **0.98** |

**表 5. 消融实验的定量结果。**

| Metrics | CamTransErr | CamRotErr | ObjTransErr | ObjRotErr |
| --- | --- | --- | --- | --- |
| MotionCtrl（*w/o* $\mathcal{C}_{RT}$） | 18.24 | 1.08 | 78.82 | 1.65 |
| MotionCtrl（*w/* $\mathcal{C}_{RT}$） | 18.24 | 1.08 | 55.33 | 1.26 |
| **FMC**（*w/o* $L_{\mathrm{cam}}$） | 20.35 | 1.19 | - | - |
| **FMC**（*w/o* $L_{\mathrm{obj}}$） | **18.12** | **1.03** | 46.62 | 1.15 |
| **FMC** | **18.12** | **1.03** | **42.25** | **0.96** |

- **相机与物体运动的同时控制。** 我们探索同时使用相机与物体控制信号。由于 Direct-a-Video 只支持基本相机运动，我们选择 MotionCtrl 作为对比方法。我们随机模拟物体与相机的运动，得到多样轨迹。如图 9 所示，FMC 生成的视频更忠实地对齐指定条件。MotionCtrl 能捕捉相机运动，但难以生成真实的物体动态。这些结果证明 FMC 能有效同时控制相机与物体运动。表 3 中的物体误差指标表明本方法在物体运动控制上更好。此外，如表 4 所示，FMC 在用户研究中得分更高，在质量与运动保真度上优于先前方法。

图 10 给出 4 种不同情形下的视频生成结果：**static single-object**、**dynamic single-object**、**static multi-object**、**dynamic multi-object**。得益于 SynFMC 中运动模式的多样性，FMC 能有效学习一系列多样、高级且复杂的镜头。例如图 10 第 2 行，相机先从正面拍摄人物，再从后方跟随。最后两行展示多物体场景中的表现：物体与相机之间的相对运动与输入条件高度吻合。

### 5.2 消融实验

![图 10. FMC 在不同情形下的同时控制结果。第 2 行的复杂情形表明本方法学到了复杂镜头：相机先从正面拍摄滑雪者，再从后方跟随。](../../../arxiv/geometry_control/free_form_motion_control/extracted/simul_compressed.png)

![图 11. 在 SynFMC 上训练的 MotionCtrl 的同时控制结果：训练物体模块时不使用与使用相机姿态。](../../../arxiv/geometry_control/free_form_motion_control/extracted/ablation1_compressed.png)

- **SynFMC Dataset。** 为验证 SynFMC 中完整相机与物体姿态标注的有效性，并评估数据集泛化能力，我们在其上训练 MotionCtrl，并把标注适配为其输入格式。我们先训练相机模块，再以两种方式优化物体模块：不使用相机姿态（与原 MotionCtrl 一致），以及像 FMC 一样使用相机姿态。如图 11 所示，在优化物体模块时纳入已知相机姿态，能使物体更准确地跟随输入轨迹，降低其离开视野的风险。表 5 中的物体运动误差进一步凸显使用完整相机与物体姿态标注的收益。

- **OMC。** 如图 12 前 2 行以及表 5 中的物体运动误差所示，FMC 在运动精度上优于在 SynFMC 上训练的 MotionCtrl。这一提升来自 OMC 处理 6D 姿态的能力：它能根据朝向以及物体与相机的距离（由粗 mask 尺寸反映）生成更真实的物体外观。MotionCtrl 的物体运动控制模块只能处理没有姿态与距离信息的 2D 图像空间轨迹，从而限制了与输入的对齐精度。

![图 12. 消融实验中不同设定的结果。第一行为在 SynFMC 上训练的 MotionCtrl。](../../../arxiv/geometry_control/free_form_motion_control/extracted/ablation2_compressed.png)

- **训练目标。** 我们做两组实验，评估式 (1) 中 $L_{\mathrm{cam}}$ 与式 (2) 中 $L_{\mathrm{obj}}$ 的影响。首先，我们仅用标准 diffusion loss 训练 CMC。如图 12 第 3 行所示，没有 $L_{\mathrm{cam}}$ 的模型倾向于平移前景物体以达成相似的相对运动，这并不准确匹配输入姿态。表 5 中的相机运动误差凸显了 $L_{\mathrm{cam}}$ 的有效性。此外，不使用 $L_{\mathrm{obj}}$ 训练 OMC 会导致不期望的物体外观，如图 12 第 5 行所示；表 5 中的物体运动误差确认了 $L_{\mathrm{obj}}$ 的收益。

- **与不同个性化 T2I 模型的适配性。** 如图 2 所示，FMC 可适配多种个性化骨干（Toonyou、RealisticVision、Stable Diffusion），表明所提出的数据集 SynFMC 与相应训练策略不会损害模型原有的生成能力。

---

## 6. 结论

本文引入 SynFMC：一份带完整 6D 姿态信息与多样资产的数据集，既提供标准镜头，也提供现实中难以拍摄、轨迹又近似真实场景的复杂镜头。借助 SynFMC，所提出的方法 FMC 能在同一视频中独立或联合地对物体与相机运动做 3D-aware 控制。实验结果证明了 SynFMC 数据集与 FMC 方法的有效性。

**局限性。** 本方法对多物体复杂运动的控制能力仍然有限。需要更好的指标以更准确地评估物体运动。未来希望引入额外输入模态（例如图像），以便为参考主体定制运动视频。

**致谢：** 本项目获国家自然科学基金（NSFC）资助，批准号 62472104。本工作获达摩院创新研究计划支持。Tao 博士的研究部分获 NTU RSR 与 Start Up Grants 支持。本工作还部分获国家自然科学基金资助，批准号 62276067。

---

> **附录说明：** arXiv:2501.01425v3（ICCV 2025 相机就绪稿，10 页）的 TeX 源与 PDF 均不含附录。正文第 5.1 节提到“更多例子见补充材料”，但该 supplementary 未随本源发布，故译本覆盖全部正文六节、五表、十二图及致谢，不另编造附录。
