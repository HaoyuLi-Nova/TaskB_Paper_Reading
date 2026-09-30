# UCPE：面向可控视频生成的统一相机位置编码

**作者：** Cheng Zhang$^{1,2}$、Boying Li$^{1\dagger}$、Meng Wei$^{1}$、Yan-Pei Cao$^{3}$、Camilo Cruz Gambardella$^{1,2}$、Dinh Phung$^{1}$、Jianfei Cai$^{1}$

**机构：** $^{1}$莫纳什大学（Monash University）；$^{2}$Building 4.0 CRC；$^{3}$VAST

**贡献说明：** $\dagger$通讯作者

**出处：** CVPR 2026（arXiv:2512.07237）

**arXiv：** [2512.07237](https://arxiv.org/abs/2512.07237)

**项目页：** https://chengzhag.github.io/publication/ucpe/

**代码：** https://github.com/chengzhag/UCPE

**原文 TeX：** [`arxiv/geometry_control/ucpe/extracted/main.tex`](../../../arxiv/geometry_control/ucpe/extracted/main.tex)

---

## 摘要

Transformer 已成为 3D 感知、视频生成，以及自动驾驶与具身智能 world model 的通用骨干；其中理解相机几何，对于把视觉观测锚定到三维空间至关重要。然而，现有相机编码方法往往依赖简化的针孔（pinhole）假设，难以在真实相机多样的内参与镜头畸变上泛化。

我们提出 **Relative Ray Encoding**：一种几何一致的表示，把完整相机信息统一起来，包括 6-DoF 位姿、内参与镜头畸变。为在多样化可控性需求下评估其能力，我们以相机可控的 text-to-video 生成为测试任务。在该设定中，我们进一步识别出 pitch 与 roll 是 **Absolute Orientation Encoding** 的两个有效分量，从而能够对初始相机朝向做完整控制。二者共同构成 **UCPE（Unified Camera Positional Encoding）**：通过轻量 spatial attention adapter 接入预训练视频 Diffusion Transformer，**可训练参数增量不到 1%**，同时达到 state-of-the-art 的相机可控性与视觉保真度。

为支持系统化训练与评估，我们构建了覆盖广泛相机运动与镜头类型的大规模视频数据集。大量实验验证了 UCPE 在相机可控视频生成中的有效性，并凸显其作为 Transformer 通用相机表示、面向未来多视角、视频与 3D 任务的潜力。代码见 https://github.com/chengzhag/UCPE 。

---

![图 1. 相机可控视频生成总览。给定用户指定的文本提示与相机参数（包括水平视场、畸变 $\xi$，以及可选绝对朝向的相机位姿，编码为 latitude-up map），模型合成与多样相机几何一致的写实视频，展现准确的位姿与镜头可控性以及高视觉保真度。应用覆盖生成式视频内容创作，以及面向自动驾驶与具身智能的 world model。](../../../arxiv/geometry_control/ucpe/extracted/figure/pdf/teaser.png)

---

## 1. 引言

Transformer 已成为新视角合成、3D 重建以及相机可控视频生成等现代架构的基础。这些网络必须推理视觉观测如何由相机几何（例如 pose、intrinsics、投影模型、镜头畸变）形成，从而把像素序列锚定到 3D 空间。为从有限视角最大化视觉覆盖，自动驾驶、机器人感知与 world model 等真实应用常使用 fisheye、catadioptric 以及 equirectangular 投影来获得 360° 全景。然而，尽管 Transformer 骨干持续进步，多数相机条件方法仍依赖简化的针孔假设，忽视这些多样相机几何中高度非线性的投影与强烈镜头畸变。

![图 2. 相机编码方法比较。(a) Direct Parameterization 把相机内参与外参编码为原始参数，缺乏几何可解释性，也难以跨相机类型兼容。(b) Plücker Encoding 把每条射线表示为方向与矩向量对，物理上有根据，但是绝对的、依赖坐标系的描述。(c) Projective Positional Encoding 在投影空间编码相对相机，但仍假设针孔投影，无法建模非线性镜头畸变。(d) 本文的 Relative Ray Encoding 在射线空间重写几何关系，每个 token 对应其自身视线，从而更好泛化位姿，并兼容任意相机镜头。](../../../arxiv/geometry_control/ucpe/extracted/figure/pdf/rays.png)

实践中，这些针孔假设体现在送入网络的相机表示上，如图 2 所示，并在第 3.1 节详述；它们可分为 *absolute* 与 *relative* 编码。Absolute encodings 要么直接输入原始相机参数，要么采用 Plücker encoding，在针孔模型下表示射线。然而，如实验（第 4.2 节）所示，它们对特定世界坐标系配置的依赖会限制跨多样相机几何的泛化。近期的 relative encodings，如 GTA 与 PRoPE，通过向 attention 注入几何或投影感知关系，改善了新视角合成中的多视角一致性。但它们仍限于针孔投影，且不易与预训练视频扩散模型兼容。*因此，现有相机条件方法仍缺少一种显式建模异构相机几何的编码机制，从而在变化的内参、畸变剖面与视点设定上泛化有限。*

为应对这些局限，我们提出 *Relative Ray Encoding*（图 2d），把位置编码从相对相机编码改写为相对射线编码。通过把每个 token 表示为局部射线坐标系中的视线，该编码使 attention 直接在射线空间而非相机层级上运算，从而在异构相机镜头上实现几何一致的条件化。

为在多样化可控性需求下评估这一能力，我们以相机可控的 text-to-video 生成为测试任务。然而，现有相机条件 T2V 方法仅相对第一帧定义相机位姿，使全局旋转坐标系含糊。特别是两个自由度——*pitch* 与 *roll*——无法被唯一确定，因而无法指定或复现初始视角的绝对朝向。为消除该歧义，我们引入 *Absolute Orientation Encoding*（图 1 第三列），把相机锚定到与重力对齐的 “up” 方向。该编码把绝对 pitch 与 roll 角显式提供为 latitude-up map，从而对相机绝对朝向实现精确、可复现的控制。

把这两个分量结合起来，得到 **UCPE（Unified Camera Positional Encoding）**：一个与相机模型无关的框架，把完整相机几何——包括 6-DoF 位姿、内参与畸变——注入 Transformer attention。UCPE 提供统一表述，把异构相机几何收拢到同一形式，从而对镜头类型、视点与朝向做细粒度控制，如图 1 所示。通过轻量 spatial attention adapter 接入基于扩散的视频生成后，UCPE 使现有 Diffusion Transformer 仅需微调不到 1% 的额外参数。

为支持系统化训练与评估，我们进一步构建覆盖多样内参、畸变剖面与相机运动的大规模视频数据集，作为相机可控生成的综合基准。实验表明，UCPE 在多样相机配置上持续提升相机可控性与视觉保真度，在物理相机镜头与 attention 机制之间建立统一桥梁，服务于可控视频生成。

贡献如下：

- 提出 *UCPE*：把完整相机几何（6-DoF 位姿、内参、镜头畸变）编码进 Transformer 的统一框架。UCPE 采用混合表述：以 *Relative Ray Encoding* 做射线空间几何推理与投影非线性，以 *Absolute Orientation Encoding* 提供与重力对齐的参考，从而实现可控的相机朝向。
- 引入轻量 *attention injection method*，以不到 1% 的额外可训练参数把 UCPE 接入视频 Diffusion Transformer，在实现相机感知视频生成的同时保留预训练模型的视觉保真度。
- 构建覆盖多样相机内参、畸变剖面与运动轨迹的 *大规模视频数据集*，作为在变化相机几何下评估相机可控生成的基准。
- 大量实验表明，UCPE 不仅能准确控制镜头与朝向，还提升针孔相机以及多样相机类型上的位姿精度与生成质量。

---

## 2. 相关工作

**Video Diffusion Models.** 基于扩散的生成模型已成为视频合成的有力范式，在写实性与时间连贯性上表现突出。早期工作把图像扩散框架扩展到视频域，加入时间模块以建模运动动态。随后的 UNet-based 与 Transformer-based 架构在 3D 视频潜空间中做时空去噪。随着训练数据与空间分辨率扩大，HunyuanVideo、Seedance 等大规模扩散框架已把视频生成质量推到接近真实影像的水平。为实现可控性，这些模型通常接入文本、视觉或图像空间条件（例如 depth、edges 或 optical flow），以引导运动与外观合成。尽管如此，多数预训练视频扩散框架仍对从根本上决定视觉观测如何形成的底层相机几何保持无关。

**Camera-Controlled Generation.** 近期面向相机可控视频生成的工作，旨在在显式相机控制下合成视频，途径包括微调或免训练适配。一类方法用 3D 表示引导新视角生成，依赖渲染、tracking、optical flow 或 coordinates 等图像空间条件。它们便于视点控制并改善 3D 一致性，但依赖从输入图像或视频得到的估计 depth map、点云、mesh 或 3D Gaussian，从而限制运动多样性，并常在大相机运动或不完美重建下退化。另一方向直接以相机位姿条件化扩散模型，覆盖 video-to-video、image-to-video 与 text-to-video 生成，依赖训练中学到的隐式 3D 先验。这些方法展示了有前景的相机感知生成，但通常把参考系定义在第一帧上，在 T2V 合成中缺少绝对朝向控制（例如 pitch 与 roll）。此外，多数现有方法只操控外参（即 6-DoF 位姿），忽视内参与镜头畸变，从而妨碍跨多样相机配置与投影模型的泛化。*相较之下，本文引入统一、几何一致的表示，联合编码位姿、内参与畸变，从而在广泛投影类型上实现有物理根据、准确的相机感知控制。*

**Camera Encoding.** 在扩散模型中实现细粒度相机控制，根本上取决于相机几何如何被表示与编码。少数工作探索直接相机条件化，把相机参数注入扩散模型，从而在无需显式 3D 重建的情况下实现可控视点过渡。另一些方法通过 ray-map encodings 编码相机位姿，用于 image-to-video 与 text-to-video 生成。Ray-map 表示用对应射线的原点与方向，或用其 Plücker 坐标描述每个像素，使模型能同时纳入内参与外参属性。然而，这些 absolute encodings 依赖预先定义的世界坐标系，使表示依赖于任意坐标选择，并限制跨场景泛化。

受 RoPE 等相对位置编码成功的启发，若干研究提出 *relative camera encodings*，建模视角间的成对几何变换。通过在 attention 层级直接编码相对 SE(3) 变换，这些方法不再需要固定参考系，并已被证明能改善多视角推理与新视角生成。基于这些认识，*本文的 UCPE 超越透视相机，在几何一致的表述中联合编码位姿、内参与畸变。* 该统一表示与 attention adapter 无缝结合，使 Diffusion Transformer 能在不同投影类型上一致地推理相机几何，并实现准确的视点控制。

---

## 3. 方法

### 3.1 预备知识

**Camera as Ray Mapping.** 所有相机模型，无论投影类型，都可纳入把图像坐标映射到空间中三维射线的统一表述。我们不假设特定投影模型，而把相机定义为射线映射函数 $\Phi_{\psi}: (u,v) \mapsto (\mathbf{o}_{u,v}^{\textrm{cam}}, \mathbf{d}_{u,v}^{\textrm{cam}})$，对每个像素 $(u,v)$ 在相机坐标系中产生射线原点 $\mathbf{o}_{u,v}^{\textrm{cam}} \in \mathbb{R}^3$ 与单位方向 $\mathbf{d}_{u,v}^{\textrm{cam}} \in \mathbb{S}^2$。该映射由模型相关的参数集 $\psi$ 参数化，例如焦距与畸变系数，取决于所选投影模型。对 *central* 相机，所有射线共享同一原点（$\mathbf{o}_{u,v}^{\textrm{cam}} = \mathbf{0}$）；而 *non-central* 相机（如 catadioptric 或全景系统）则为像素相关的原点。除非另有说明，后续推导为记号简便假设 central 相机模型。我们以 Unified Camera Model（UCM）为代表性例子（细节见补充材料第 A 节）。

记 $\mathbf{T}^{\textrm{wc}} = \begin{bmatrix} \mathbf{R} & \bm{t} \\ \bm{0}^{\top} & 1 \end{bmatrix} \in \mathrm{SE}(3)$ 为从相机到世界坐标系的相机位姿。施加该变换得到世界空间射线表示：

$$
\bm{d}_{u,v} = \mathbf{R}\, \bm{d}_{u,v}^{\textrm{cam}},
\qquad
\bm{o}_{u,v} = \bm{t}.
\tag{1}
$$

这为推理具有多样投影特性的相机提供了共同的几何基础。

**Absolute Camera Encodings** 在世界坐标系中显式编码每相机参数或每像素射线。最直接的做法是把原始相机参数编码为数值特征向量，如图 2a。更有物理根据的替代是 *Plücker encoding*（图 2b），把每条射线改写为六维向量 $(\bm{d}_{u,v},\, \bm{m}_{u,v}) \in \mathbb{R}^6$，其中 $\bm{m}_{u,v} = \bm{o}_{u,v} \times \bm{d}_{u,v}$ 为射线矩。该表示对每个像素的射线给出紧凑、数值稳定的描述，以物理可解释的形式同时编码相机位姿与镜头。然而，作为 absolute encoding，它本质上仍对所选坐标系敏感，从而限制跨不同场景与视点的泛化。

**Relative Camera Encodings.** 近期工作引入 relative encodings，在 Transformer 的 attention 机制中直接建模成对相机关系，从而对全局坐标选择不变。对每个 token $t \in \{1,\ldots,T\}$ 及其对应相机索引 $i(t) \in \{1,\ldots,N\}$，由 world-to-camera 变换 $\mathbf{T}_{i(t)}^{\textrm{cw}} = (\mathbf{T}_{i(t)}^{\textrm{wc}})^{-1}$ 导出变换矩阵 $\mathbf{D}_{t}$：

$$
\mathbf{D}_{t} = \mathbf{I}_{d/4} \otimes \mathbf{T}_{i(t)}^{\textrm{cw}},
\tag{2}
$$

其中 $\mathbf{I}$ 为单位矩阵，$d$ 为特征维，$\otimes$ 为 Kronecker 积，把 3D 变换复制到各个特征子空间。所得分块对角矩阵 $\mathbf{D} = \operatorname{blkdiag}(\mathbf{D}_1,\ldots,\mathbf{D}_{T})$ 编码每 token 的相机位姿，并通过 token-wise 矩阵–向量乘法 $\odot$ 作用于 token 特征，例如 $\mathbf{D}\odot Q = \operatorname{blkdiag}(\mathbf{D}_1 Q_1, \ldots, \mathbf{D}_{T} Q_{T})$。

*Camera Positional Encoding（CaPE）* 用这类变换矩阵把相对相机位姿注入 self-attention。Attention 运算被修改为：

$$
O = \operatorname{Attn}\big(\mathbf{D}^{\top}\odot Q,\; \mathbf{D}^{-1}\odot K,\; V\big).
\tag{3}
$$

这把每个 query–key 交互 $Q_{t_1}^{\top} K_{t_2}$ 替换为 $Q_{t_1}^{\top} \mathbf{D}_{t_1} \mathbf{D}_{t_2}^{-1} K_{t_2}$，其中 $\mathbf{D}_{t_1}\mathbf{D}_{t_2}^{-1} = \mathbf{I}_{d/4} \otimes \mathbf{T}_{i(t_1)}^{\textrm{cw}} (\mathbf{T}_{i(t_2)}^{\textrm{cw}})^{-1}$，从而使 attention 以相对相机位姿为条件。

*Geometric Transform Attention（GTA）* 在此表述上进一步变换 value 矩阵，从而在特征比较与聚合时都强制几何一致性：

$$
O = \mathbf{D} \odot \operatorname{Attn}\big(\mathbf{D}^{\top}\odot Q,\; \mathbf{D}^{-1}\odot K,\; \mathbf{D}^{-1}\odot V\big).
\tag{4}
$$

基于这一想法，*Projective Positional Encoding（PRoPE）*（图 2c）把相对变换一般化：在内参 $\mathbf{K}_{i}$ 下，用相机投影矩阵 $\mathbf{P}_{i} = \begin{bmatrix} \mathbf{K}_{i} & \bm{0} \\ \bm{0}^{\top} & 1 \end{bmatrix} \mathbf{T}_{i}^{\textrm{cw}}$ 替换 $\mathbf{T}_{i}^{\textrm{cw}}$。该投影表述捕捉完整的相机视锥几何，但限于针孔相机。

### 3.2 Unified Camera Positional Encoding

**Relative Ray Encoding.** 现有 relative camera encodings 假设同一视角内所有图像 token 共享单一、线性的投影函数。该假设把整幅图像当作一个刚体，简化了相机间几何推理，却忽视相机内部投影几何的偏离。因此，这类相机级编码在理想化针孔模型下表现良好，却难以泛化到呈现空间变化投影行为的真实相机，例如镜头畸变与大视场效应。

为容纳非线性投影模型，我们把编码从 *camera-to-camera* 改写为 *ray-to-ray* 变换，以捕捉跨图像 token 的细粒度几何变化。具体地，对每个图像 token $t$，我们构造局部 ray-to-world 矩阵 $\mathbf{T}^{\textrm{wr}}_{t}$，作为 attention 层级特征变换的几何算子。形式化地，每个图像 token $t$ 关联一条由世界坐标系中原点与方向参数化的视线：

$$
\bm{r}_t = \big(\bm{o}_t,\, \bm{d}_t\big),
\quad
\bm{o}_t \in \mathbb{R}^3,
\quad
\bm{d}_t \in \mathbb{S}^2,
\tag{5}
$$

其中 $(\bm{o}_t, \bm{d}_t)$ 按式 (1) 导出。我们通过定义锚定在相机中心的 *local ray coordinate system* 来构造 $\mathbf{T}^{\textrm{wr}}_{t}$，其正交基为 $\mathbf{R}^{\textrm{wr}}_{t} = [\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t]$。如图 2d，我们取每条射线方向 $\mathbf{d}_t$ 为局部 $z$ 轴，并基于相机向下方向 $\mathbf{y}^{\textrm{cam}}_{i(t)}$ 确定另外两个正交轴：

$$
\bm{z}_t = \bm{d}_t,
\quad
\bm{x}_t = \bm{y}^{\textrm{cam}}_{i(t)} \times \bm{z}_t,
\quad
\bm{y}_t = \bm{z}_t \times \bm{x}_t.
\tag{6}
$$

该正交基 $\mathbf{R}^{\textrm{wr}}_{t}$ 连同平移 $\mathbf{t}^{\textrm{wr}}_{t} = \mathbf{o}_t$，构成 ray-to-world 变换：

$$
\mathbf{T}^{\textrm{wr}}_{t}
=
\begin{bmatrix}
\mathbf{R}^{\textrm{wr}}_{t} & \bm{t}^{\textrm{wr}}_{t} \\
\mathbf{0}^{\top} & 1
\end{bmatrix}.
\tag{7}
$$

因此每个 token 特征关联其特定几何变换 $\mathbf{T}^{\textrm{rw}}_{t} = (\mathbf{T}^{\textrm{wr}}_{t})^{-1}$，把世界坐标映射到局部射线，并作为后续 attention 层级特征变换的几何算子。该表述使 attention 在射线空间中推理，而不是在同一相机帧内共享同一位置编码，从而为 UCPE 在帧内与跨帧提供一致、物理可解释的几何基础。

![图 3. Spatial Attention Adapter 总览。Adapter 通过保留预训练先验的轻量分支，把 UCPE 注入预训练 Transformer。它由 world-to-ray 变换 $\mathbf{T}^{\textrm{rw}}$ 与可选 Lat-Up map 构造混合编码，在 attention 内施加它们，再经 zero-initialized 线性层把得到的相机感知 token 融合回去。](../../../arxiv/geometry_control/ucpe/extracted/figure/pdf/pipeline.png)

**Absolute Orientation Encoding.** Relative ray encoding 虽能在异构相机间做几何一致推理，却不提供绝对朝向的概念，使第一帧朝向含糊。然而，多数真实视频是在与重力对齐的 “up” 方向下拍摄的，这自然定义了绝对 pitch 与 roll 角。为纳入该全局朝向，我们采用图 1 所示的 *latitude-up map（Lat-Up map）*：把每条射线的 *latitude angle* 与对应 *up-vector* 拼接。该表示通过天空–地面分离与物体对齐等外观线索捕捉相机旋转，同时也为广角镜头提供一定的畸变感知。

计算 *latitude map* 时，从射线 $t$ 的世界空间方向 $\bm{d}_t = [d_{t,x}, d_{t,y}, d_{t,z}]^{\top}$ 出发，如式 (5)。Latitude map 编码每条射线相对水平面的角仰角：

$$
\mathrm{Lat}_t = \arctan2 \big(-d_{t,y},\, \sqrt{d_{t,x}^{2} + d_{t,z}^{2}}\big),
\tag{8}
$$

其中正值对应向上看的射线。

为计算 *Up map*，我们把每条世界空间射线 $\mathbf{d}_t$ 绕其局部轴 $\bm{k}_t = \bm{d}_t \times \bm{u}^{\textrm{wld}}$ 旋转一个小角度 $\delta$，其中 $\bm{u}^{\textrm{wld}}$ 表示世界上方向。再把旋转后的射线 $\bm{d}^{\textrm{rot}}_{t}$ 投影回图像平面，所得归一化像素位移定义上方向：

$$
\mathrm{Up}_t =
\frac{[\Delta u_t,\; \Delta v_t]}{\|[\Delta u_t,\; \Delta v_t]\|},
\tag{9}
$$

其中 $[\Delta u_t, \Delta v_t]$ 是旋转射线与原始射线之间的投影偏移（细节见补充材料第 B 节）。所得 Lat-Up encoding $[\mathrm{Lat}_t, \mathrm{Up}_t]$ 为每个 token 提供全局朝向上下文，从而在生成时显式控制相机 pitch 与 roll。

因此 UCPE 整合两条互补线索，即局部射线几何与绝对朝向，以提供一致、物理可解释、并能跨相机镜头泛化的编码。

### 3.3 用于 UCPE 注入的 Spatial Attention Adapter

**Spatial Attention with Hybrid Encoding.** 我们按 PRoPE 的做法，把 relative ray encoding 与 RoPE 融合，以同时促进射线空间与图像空间推理。对每个 token $t$，构造分块对角算子：

$$
\mathbf{D}^{\textrm{UCPE}}_{t}
= \operatorname{blkdiag}\!\big(
\mathbf{D}^{\textrm{Ray}}_{t},\;
\mathbf{D}^{\textrm{RoPE}}_{t}
\big),
\tag{10}
$$

其中 $\mathbf{D}^{\textrm{Ray}}_{t} = \mathbf{I}_{d/8} \otimes \mathbf{T}^{\textrm{rw}}_{t} \in \mathbb{R}^{d/2 \times d/2}$ 编码式 (7) 所定义的 world-to-ray 变换，$\mathbf{D}^{\textrm{RoPE}}_{t} \in \mathbb{R}^{d/2 \times d/2}$ 用 RoPE 构造以编码图像空间位置，二者各占一半特征维。我们把该设计共享给 GTA 与 PRoPE 基线，这是它们表现最好的配置。随后定义 $\mathbf{D}^{\textrm{UCPE}} = \operatorname{blkdiag}(\mathbf{D}^{\textrm{UCPE}}_{1},\ldots,\mathbf{D}^{\textrm{UCPE}}_{T})$，并按式 (4) 将其施加于 attention 运算，如图 3。

此外，Spatial Attention 模块把 Lat-Up map 作为可选输入，以提供绝对朝向控制。当该输入存在时，Lat-Up 特征经线性层投影到 token 维，再作为 bias 加到输入 token 上。

**Adapting Transformer with Attention.** 我们采用 Wan 作为预训练 Diffusion Transformer；直观上，注入 UCPE 最合适的位置是其建模空间与时间相关的 self-attention 层。然而，直接用 UCPE 替换现有位置编码（例如 3D RoPE）可能扰动大规模预训练建立的先验。

为保留先验，我们通过与原始 attention 并行的轻量、LoRA 风格 adapter 接入 UCPE。如图 3，每个 DiT block 保留标准 self-attention，同时引入基于所提编码的相机条件分支 $\operatorname{UCPEAttn}(\cdot)$。关键在于：得益于 UCPE 的几何感知设计，该 adapter 只需少量参数即可有效建模相机条件相关。实践中，我们用 $\mathcal{P}_{Q}$、$\mathcal{P}_{K}$、$\mathcal{P}_{V}$ 把输入 token 线性投影到原始维的 $1/C$，并按比例减少 attention head 数，从而有效降低参数量与计算量。$\operatorname{UCPEAttn}(\cdot)$ 的输出再经权重零初始化的线性投影层映射回去，确保初始化时预训练模型未被改变。

该 spatial adapter 使 UCPE 能高效接入 Diffusion Transformer，以极少额外参数提供细粒度相机控制。

---

## 4. 实验

![图 4. 在我们合成数据集上的比较。UCPE 忠实跟随目标轨迹，并产生与 Lat-Up Map 可视化对齐的一致镜头畸变。相较之下，Wan CameraCtrl 出现相机运动偏离，而 ReCamMaster 未能复现预期畸变。颜色与图中高亮效果对应。](../../../arxiv/geometry_control/ucpe/extracted/figure/pdf/panshot.png)

![图 5. 在 RealEstate10K 数据集上的比较。UCPE 生成更锐利、更细致、更好跟随目标相机运动的帧。CameraCtrl 产生严重伪影（左）与糟糕构图（右）；AC3D 保留训练集审美，但出现构图不平衡（左）与动态范围偏低（右）。Wan CameraCtrl 与 ReCamMaster 虽基于同一骨干，却难以保证相机一致性，在针孔设定下导致运动减弱（左）以及不期望的畸变伪影（右）。](../../../arxiv/geometry_control/ucpe/extracted/figure/pdf/re10k.png)

### 4.1 实验设置

**Datasets.** 为在多样相机条件（pose、intrinsics 与 distortion）下训练与评估，我们从 in-the-wild 360° 视频出发，用 Unified Camera Model（UCM）合成约 48k 片段，随机化 $\mathrm{xFoV}$ 与畸变 $\xi$，覆盖 pinhole、wide-angle、fisheye 配置（见补充材料第 A 节）。为评估分布外泛化，我们额外在 RealEstate10K 测试集中随机选取 100 个片段，使用固定 100° $\mathrm{xFoV}$ 的针孔相机，不做任何进一步适配或微调。

**Evaluation Metrics.** 我们从四个方面评估生成视频（细节见补充材料第 C 节）：

- *Video quality：* 用 FID、FVD 评估数据集级保真度，用 CLIP score 评估 text–video 对齐，用 Q-Align 评估面向用户的质量。
- *Relative camera pose control：* 用 ground-truth 畸变把每帧校正到针孔视图，再由 ViPE 估计，报告 translation（TransErr）、rotation（RotErr）以及 motion consistency（CamMC）。
- *Absolute camera orientation：* 经 GeoCalib 估计 pitch 与 roll，并与 ground-truth 视频估计比较，得到绝对朝向误差。
- *Lens control：* 用 GeoCalib 标定径向模型的 FoV 与畸变系数 $k_1$、$k_2$，并相对 ground-truth 标定计算绝对误差。

### 4.2 与先前方法的比较

**Baselines.** 我们将 UCPE 与若干代表性相机条件方法比较：

1. *ReCamMaster* 把相机外参 $[\mathbf{R}, \mathbf{t}] \in \mathrm{SE}(3)$ 注入扩散模型；我们通过拼接归一化 $\mathrm{FoV}$ 与畸变 $\xi$ 将其扩展到镜头控制。
2. *Wan CameraCtrl*（Wan2.1-Fun-V1.1-1.3B-Control）是 CameraCtrl 在 Wan 上的第三方实现，经卷积 adapter 接入 Plücker encoding；我们将其适配到 text-to-video 生成。
3. *AC3D* 通过建在 CogVideoX 上的 ControlNet adapter 施加 Plücker encoding。
4. *CameraCtrl* 把 Plücker encoding 接入 AnimateDiff 的 U-Net 时间 attention 层。

对方法 (1) 与 (2)，我们分别实现两个变体：一是遵循其原始设计，相机位姿相对第一帧定义；二是采用锚定在第一帧的重力对齐坐标系，以显式控制 roll 与 pitch（记为 w/ absolute orientation）。方法 (1) 与 (2) 与 UCPE 共享同一基座，并在我们的数据集上以相同设定微调以便公平比较，唯一例外是 (2) 因全参数优化而使用更小的学习率 $1\mathrm{e}{-5}$（细节见补充材料第 D 节）。对 (3) 与 (4)，我们直接评估作者发布的预训练模型，因此只在其训练数据集 RealEstate10K 上报告结果。

**表 1. 在我们合成数据集上的定量比较。** UCPE 在镜头、朝向与位姿控制上全面优于所有基线，同时以比 ReCamMaster 少 90% 的参数保持强视频质量。在 w/o 与 w/ Absolute Orientation Control 两种设定下，它都给出更低的 pitch、roll 与旋转误差。消融结果（下块）表明，Spatial Attention Adapter 中适中的压缩比（例如 1/8-dim）在可控性与保真度之间取得最佳折中。灰色数字表示因缺少相应控制而不适用的指标。此处 *1/C-dim* 表示 token 投影压缩比，$(d \times n)$ 为每头维数与 attention head 数。最优结果加粗。

| Settings | Method | FoV (°) | $k_1$ | $k_2$ | Pitch (°) | Roll (°) | RotErr (°) | TransErr | CamMC | FVD | FID | CLIP$\uparrow$ | Trainable Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w/o Absolute Orientation Control | ReCamMaster | 10.25 | 0.210 | 0.143 | *10.00* | *7.42* | 10.89 | 31.44 | 37.38 | **555.54** | 69.91 | 24.86 | 354M |
|  | Wan CameraCtrl | 10.05 | 0.222 | 0.142 | *9.35* | *7.31* | 17.04 | 35.09 | 46.10 | 593.10 | 67.83 | 25.05 | 1.5B |
|  | UCPE | **9.62** | **0.174** | **0.120** | *8.03* | *6.64* | **4.29** | **13.46** | **15.94** | 569.31 | **66.22** | **25.11** | 35.5M |
| w/ Absolute Orientation Control | ReCamMaster | 10.04 | 0.183 | 0.136 | 6.62 | 5.29 | 9.23 | 28.95 | 33.88 | 605.83 | 67.07 | 24.84 | 354M |
|  | Wan CameraCtrl | 9.86 | 0.230 | 0.162 | 6.25 | 6.01 | 17.92 | 39.16 | 50.32 | 554.43 | 65.73 | 25.01 | 1.5B |
|  | UCPE | **8.22** | **0.129** | **0.102** | **4.35** | **3.74** | **4.12** | **15.21** | **17.59** | **495.14** | **63.37** | **25.12** | 35.6M |
| w/ Absolute Orientation Control & Ablation Study | 1/2-dim ($128 \times 6$) | 8.39 | 0.170 | 0.110 | 4.11 | 3.93 | 3.69 | **14.03** | 16.06 | 534.44 | 64.88 | 25.05 | 141M |
|  | 1/4-dim ($128 \times 3$) | 8.47 | 0.149 | **0.101** | 3.94 | 4.08 | **3.43** | 14.26 | **16.02** | 512.85 | **62.86** | 25.09 | 71.0M |
|  | 1/8-dim ($192 \times 1$) | **8.22** | **0.129** | 0.102 | 4.35 | 3.74 | 4.12 | 15.21 | 17.59 | 495.14 | 63.37 | **25.12** | 35.6M |
|  | 1/12-dim ($128 \times 1$) | 8.96 | 0.151 | 0.107 | **3.91** | 3.89 | 5.13 | 14.54 | 17.84 | **487.54** | 62.98 | 25.06 | 23.8M |
|  | Pre-Attn | 8.47 | 0.145 | 0.108 | 4.26 | 3.96 | 4.03 | 15.77 | 17.85 | 502.73 | 63.62 | 25.07 | 35.6M |
|  | Post-Attn | 8.91 | 0.147 | 0.116 | 3.95 | 3.92 | 4.68 | 17.47 | 20.00 | 515.32 | 64.65 | 25.00 | 35.6M |
|  | PRoPE | 8.84 | 0.151 | 0.113 | 4.18 | 3.70 | 5.35 | 17.52 | 20.58 | 516.59 | 65.00 | 25.03 | 35.6M |
|  | GTA | 8.80 | 0.157 | 0.117 | 4.21 | **3.69** | 5.27 | 17.07 | 20.14 | 497.19 | 64.91 | 25.04 | 35.6M |

斜体数字对应原文灰色单元格：因缺少绝对朝向控制，Pitch / Roll 指标不适用，仅作参考。消融块内加粗为该块最优。

**表 2. 在 RealEstate10K 上的定量比较。** UCPE 未经微调即可良好泛化，取得最低的旋转、平移与运动误差，并给出高于在 RealEstate10K 上训练的模型（CameraCtrl 与 AC3D）的 Q-Align 分数。

| Method | RotErr (°) | TransErr | CamMC | Image Quality$\uparrow$ | Image Aesthetic$\uparrow$ | Video Quality$\uparrow$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ReCamMaster | 1.10 | 5.64 | 6.15 | 0.9492 | 0.5185 | 0.9720 |
| Wan CameraCtrl | 2.22 | 7.42 | 8.67 | **0.9822** | **0.5691** | **0.9885** |
| CameraCtrl | 1.17 | 3.96 | 4.59 | 0.6877 | 0.3306 | 0.7338 |
| AC3D | 0.62 | 2.11 | 2.43 | 0.7699 | 0.3651 | 0.8211 |
| UCPE | **0.56** | **1.25** | **1.58** | 0.9480 | 0.4686 | 0.9694 |

**Quantitative Results.** 表 1 给出合成数据集上的定量比较，表 2 给出 RealEstate10K 上的结果。在我们的数据集上，我们先与 ReCamMaster 和 Wan CameraCtrl 在其原始设定（w/o Absolute Orientation Control）下比较，再在所提出的重力对齐坐标系（w/ Absolute Orientation Control）下比较。UCPE 在两种配置下都持续更好，展现更优的相机可控性与视频质量，同时相对基座 7.3B 参数仅增加 35.5M 或 0.5% 参数，比 ReCamMaster 少 90%。Lat-Up map 也提供外观线索，从而改善镜头控制（UCPE w/ vs. w/o Absolute Orientation）。

在 RealEstate10K 上，尽管未在该数据集上微调，UCPE 在 Relative Camera Pose Control 上仍优于所有基线，显示出对未见轨迹与文本提示的强泛化。需注意：UCPE 用详细场景描述训练（见图 4），而 RealEstate10K 使用短提示（见图 5）。尽管如此，UCPE 的 Q-Align 分数高于 CameraCtrl 与 AC3D，表明更好的视频质量，并在更少参数下与 ReCamMaster、Wan CameraCtrl 表现相当。

**Qualitative Results.** 我们在图 4 可视化合成数据集上的生成样本，在图 5 可视化 RealEstate10K 上的样本。在我们的数据集上，UCPE 忠实跟随指定相机轨迹，并产生与目标相机参数（由 Lat-Up Map 可视化）对齐的一致镜头畸变效果。相较之下，Wan CameraCtrl 出现明显的相机运动偏离，而 ReCamMaster 未能复现预期镜头畸变。在 RealEstate10K 上，UCPE 生成更锐利、更细致、更好遵循相机运动的视频；CameraCtrl 则出现严重伪影（左）与糟糕帧构图（右）。AC3D 因在该数据集上训练而保持 RealEstate10K 的整体审美，但出现构图不平衡（左）与动态范围偏低（过曝窗户，右）。尽管 Wan CameraCtrl 与 ReCamMaster 在与我们相同的基座与数据集上微调，它们仍难以保证相机一致性，导致相机运动减弱（左），并在针孔设定下出现不期望的畸变伪影（右）。

### 4.3 消融实验

我们在合成数据集上消融 UCPE 与 Spatial Attention Adapter 的核心设计，结果汇总于表 1 下块。

**Compression Ratio of Token Projection.** 如第 3.3 节所述，我们在 Spatial Attention Adapter 中施加压缩比为 $1/C$ 的 token 投影，以平衡效率与表示容量。我们测试四种比例，分别对应不同的每头维数与 attention head 数。更高压缩比改善视频质量指标，同时保持稳定的相机可控性。我们采用 $1/8$ 作为默认设定，它在性能与效率之间提供最佳折中。

**Injection Position of the Attention Adapter.** 我们比较 Diffusion Transformer 内的三种配置：插入到 self-attention 之前（Pre-Attn）、之后（Post-Attn），以及默认的并行设计（1/8-dim）。Pre-Attn 与 Post-Attn 变体在相机控制与视频质量上都明显变差，证实并行架构的有效性。

**Comparison with Other Relative Camera Encodings.** 为评估 relative ray encoding 的贡献，我们用先前表述（包括 PRoPE 与 GTA）替换它，同时保留 Lat-Up Map 以提供朝向与镜头线索。相较 UCPE（1/8-dim），两种替代在相机镜头与相对位姿控制上更弱，视频质量也更低，尽管它们借助 Lat-Up Map 达到相近的绝对朝向表现。我们将其归因于它们对非线性镜头投影的建模能力有限：两种方法对所有 token 使用单一相机编码，无法捕捉畸变变化。该局限对 PRoPE 尤其明显：它在若干指标上甚至弱于 GTA，凸显 UCPE 的 relative ray encoding 作为跨多样相机镜头的统一表示的优势。

---

## 5. 结论

我们提出 UCPE：通过 relative ray encoding 与绝对朝向线索，联合建模位姿、内参与镜头畸变。经轻量 spatial attention adapter 接入后，它使视频 Diffusion Transformer 以不到 1% 的额外参数实现准确的相机控制。在我们覆盖多样相机的数据集上的实验，显示出镜头可控性、朝向精度与位姿保真度的明确提升。总体而言，UCPE 为多样相机提供统一表示，并有潜力作为跨多视角、视频与 3D 任务的通用编码。

**致谢：** 本研究由 Building 4.0 CRC 支持，并部分得到澳大利亚研究理事会 Grant DP260100218 的支持。

---

## 附录

本补充材料组织如下：

- 第 A 节给出相机多样数据集构建的额外细节。
- 第 B 节阐述绝对朝向编码的推导。
- 第 C 节汇总实验所用评估指标。
- 第 D 节给出我们管线与基线的实现细节。
- 第 E 节讨论稳健性、泛化能力、局限性与未来工作。
- 第 F 节包含详细演示视频说明。

### A. 数据集构建

正文引入 UCPE：联合建模 6-DoF 位姿、内参与镜头畸变的统一相机编码。要在相机可控视频生成设定中训练并评估这种表示，需要带有多样运动、FoV 与畸变的大规模数据。然而，采集此类真实数据极为昂贵，因为需要多种镜头、硬件配置与大量人工。

为克服这些限制，我们以 360° 全景影像作为灵活的探索空间来合成相机多样视频，再把视频投影到任意虚拟相机。

我们设计 *三阶段数据合成管线*，包括：

1. *360° 全景视频整理*，以提取高质量、重力对齐的探索片段；
2. *真实旋转仿真*，从具有相似平移的透视视频迁移相机运动；
3. *相机多样视频合成*，在变化的内参、畸变与增强旋转下进行。

正文第 4.1 节给出概览，此处给出完整细节。

#### A.1 坐标约定

在本补充材料中，我们对所有投影、射线映射与位姿相关计算采用一致的 3D 坐标约定。坐标系定义如下：

- $x$ 轴指向图像右侧；
- $y$ 轴指向下方（遵循图像坐标）；
- $z$ 轴指向前方，与 ERP 相机的观看方向对齐。

在该约定下，世界上方向为

$$
\bm{u}^{\mathrm{wld}} = [0,\,-1,\,0]^{\top},
\tag{A.1}
$$

并在 ERP 投影、UCM 射线映射（第 A 节）以及 Up map 的构造（第 B 节）中一致使用。

#### A.2 360° 全景视频整理

UCPE 的一个关键收益是对绝对 pitch 与 yaw 的可控性，从而消除常规相机可控 T2V 生成中的歧义。这要求所有训练视频共享重力感知的世界坐标系，而这很难从 in-the-wild 影像中获得。

幸运的是，许多商用 360° 相机借助标定镜头与机载 IMU，产出稳定、重力对齐的 equirectangular 全景。这保证了一致的 up 向量，使定义在全景球面内的虚拟相机能够继承绝对 pitch 与 yaw。

我们从 Wallingford 等人的大规模 4K 分辨率 360° 视频语料出发，包含 24.1k 条场景与运动多样的 YouTube 视频。这些原始视频存在镜头切换、水印、低质量片段、非等距柱状投影或不完美稳定等问题。因此我们构建多阶段管线，过滤低质量片段，得到适合相机多样视频合成的高质量、重力对齐全景。

**Clip Segmentation.** 视频生成模型需要时间上一致的片段。我们首先用基于阈值的转场检测器去除硬切与黑场淡入淡出，但该方法常漏掉微妙转场（例如交叉淡化）。随后对每个初步片段运行全景 visual SLAM。经验上，SLAM 无法在场景转场间跟踪特征，从而产生 “tracking-lost” 信号。通过监测这些信号，我们把初步片段切成场景更一致的精细片段，得到大约 400k 个各 10 秒的片段。

**Camera Pose and Rotation Score Extraction.** 我们在 localization 模式下再跑一遍 SLAM，以获得更准确的位姿，得到大约 300k 个具有有效估计的片段。如前所述，要启用绝对朝向监督，需要去掉未稳定或未与重力对齐的片段。由于稳定的 360° 视频通常重力一致，我们检查估计位姿的旋转分量，并丢弃出现大漂移的片段。为量化这一点，我们计算旋转分数，定义为每帧相对第一帧的最大相对旋转：

$$
\alpha_{\max}
=
\max_{i>0}
\arccos\!\left(
\frac{\mathrm{Tr}\!\left(\mathbf{R}_0^{\top} \mathbf{R}_i\right)-1}{2}
\right),
\tag{A.2}
$$

其中 $\mathbf{R}_0$ 为第一帧旋转，$\mathbf{R}_i$ 为第 $i$ 帧旋转。漂移过大的片段稍后在第 A.3 节被移除，以确保一致的重力感知朝向。

**Quality Filtering.** 原始 360° 视频包含多种伪影，我们用三种互补过滤策略处理：

- **Low-quality filtering：** 用 Q-Align 评估图像质量、审美与视频质量，并把这些分数平均作为第 A.3 节候选选择的质量指标。
- **Watermark filtering：** 对每帧施加水印检测器，平均分数作为候选选择的水印指标。
- **vLLM-based filtering：** 自定义 vLLM 过滤器（见图 A.1）识别并移除非 equirectangular、低质量，或含叠加、缺边及其他伪影的片段。vLLM 模型还把每个片段分配到最相关的 POI（Point of Interest）类别，供后续语义平衡使用。

**图 A.1. 用于全景视频过滤与场景分类的 prompt。**

```text
You are a video understanding assistant specialized in analyzing panoramic ERP-format videos.
Given one frame of a panoramic video, your tasks are:

1. Filtering: Identify if the video should be filtered out.
Output boolean flags for the following conditions (true if the issue exists, false otherwise):

- non_ERP_format: The video is not in ERP (Equirectangular Projection) panoramic format. For example, if the video looks like a flat perspective, fisheye, cube-map, or any projection other than ERP, set this to true.

- has_subtitle_or_watermark: The video contains text overlays, subtitles, logos, or watermarks. Look carefully for visible text at the bottom, center, or corners of the video. If such elements are present and not part of the real scene, set this to true.

- edge_missing: The top or bottom edges of the ERP panorama are cut off, blacked out, cropped, or covered by logos/watermarks, so the full 360 vertical coverage is missing or obstructed. If you cannot clearly see the poles (sky/ground) or if the edges are hidden by overlays, set this to true.

- has_overlay: The frame contains artificial overlays, such as embedded UI elements, pop-up graphics, stickers, video-in-video inserts, menus, or other synthetic elements that are not part of the natural scene. If you see signs of AR/VR interface, streaming UI, or added images, set this to true.

- low_quality: The video is of poor visual quality, such as being blurry, noisy, heavily pixelated, very low resolution, or distorted in a way that prevents recognizing the scene. If the content is hard to interpret due to quality issues, set this to true.

- unnatural_content: The video contains cartoons, animations, CGI, synthetic 3D renderings, or game engine graphics rather than real-world panoramic footage. If the content is not realistic, set this to true.

2. POI Categorization: From the provided list of categories, select one or more most relevant labels that best describe the scene.
   Only use the given categories, do not invent new ones.

---

poi_category list (choose only from below):

Restaurant, Coffee-Shop, Bars-and-Pubs, Residential-area, Hotels-Motels, Vaccation-Rentals, Hospitals-Clinics, Pharmacies, Dentists, School-Universities, Library, Supermarkets, Shopping-Malls, Clothing-Stores, Shoe-Stores, Bookstores, Flowerstore, Furniture-Stores, Electorical-Store, Pet-Store, Toy-Shop, Airports, Train-Stations, Bus-Stops, Gas-Station, Car-Rental-Agencies, Theaters, Concert-Halls, Sports-Stadiums, Parks-and-Recreation-Areas, Museums, Art-Galleries, Zoos-Aquariums, Botanical-Gardens, Landmarks, Cultural-Centers, Post-Offices, Police-Stations, Courthouses, CityHalls, Banks-ATMs, Events-Conferences-halls, Beaches, Hiking-Trails, Campgrounds, Lakes, Mountains, Forest-Mountains, Farms, Street-View, Square, Business-Centers, Tech-Companies, Co-working-Spaces, Gyms-and-Fitness-Centers, Sports-Clubs, Swimming-Pools, Tennis-Courts, Auto-Repair-Shops, Car-Washes, Parking-Lots, Churches, Mosques, Temples, Graveyards.

---

Output strictly in JSON format as follows:

{
  "filter": {
    "non_ERP_format": false,
    "has_subtitle_or_watermark": false,
    "edge_missing": false,
    "has_overlay": false,
    "low_quality": false,
    "unnatural_content": false
  },
  "poi_category": ["Mountains"]
}
```

**Trajectory Scale Normalization.** 单目 SLAM 只能在任意尺度上恢复相机轨迹，导致同一单位平移在浅景中显得更快、在深景中显得更慢。为确保片段间运动一致，我们估计每片段的几何尺度并据此归一化所有轨迹。具体地，我们用 PanoFlow 计算稠密光流，经广义对极几何恢复每像素深度，再取每帧有效深度的第 25 百分位数，并在片段上取中位数，得到近平面统计量。我们用该值缩放每条轨迹，使中位近平面深度变为 1，从而在所有训练片段上得到一致的表观运动。

#### A.3 仿真真实相机旋转

整理后的全景片段作为带有相机平移的探索空间，但确定真实的相机旋转仍具挑战。合成旋转（例如匀速扫过）显得不自然，也无法反映与平移相关的真实 panning、tilting 或 rolling 模式。

我们的关键认识是：具有相似平移运动的视频往往呈现相似的旋转动态。因此我们采用基于匹配的方法，分两步：

1. 从透视视频提取真实旋转轨迹；
2. 基于平移相似度与片段质量，把这些旋转匹配并迁移到全景片段。

**Candidate Rotation Extraction.** 我们使用 CameraBench：一个含 1k 运动多样电影视频、并带文本描述相机运动的数据集。我们去掉描述中含 “zoom” 的片段以保持内参固定，并用 ViPE 提取相机位姿，得到大约 300 条独特轨迹。在某些困难场景中，ViPE 可能产生不稳定或抖动的位姿。为去除这些结果，我们计算每片段的最大旋转速度并丢弃最高的 20%。随后经 GeoCalib 把所有轨迹对齐到重力感知世界坐标系：把第一帧 up 向量对齐到世界上轴。该步骤确保所有候选旋转可直接施加到重力对齐的全景片段上，以产生自然的绝对朝向。

**Trajectory Matching and Rotation Transfer.** 对每个全景片段，我们寻找平移相似的 CameraBench 轨迹，并把它们的旋转迁移到虚拟相机。我们首先经受垂直轴旋转约束的 Umeyama fitting，把每条候选轨迹对齐到全景片段。然后保留 RMSE 最低的前 30 个候选以供进一步选择。

原始 360° 视频在 POI 类别上呈现严重长尾（见图 A.2(a)），可能是因为 360° 相机广泛用于山地户外运动与街景等特定场景。因此我们在候选选择中施加一系列多样性约束，以确保语义平衡与运动多样性。具体地，每个候选用质量分数、水印分数以及第 A.2 节的旋转分数 $\alpha_{\max}$ 的平均作为复合指标打分。然后从最好到最差贪心遍历，同时强制以下多样性约束：

- **Panoramic clip diversity：** 每个全景最多 5 个 CameraBench 匹配。
- **CameraBench trajectory diversity：** 每条轨迹最多使用 100 次。
- **POI semantic balance：** 若该全景片段的所有 POI 类别都已超过 1000 次匹配，则跳过。
- **Motion diversity：** 我们把平均光流幅度作为运动分数，并维护宽度为 10 的分箱，每箱最多 2000 个样本。

这些约束虽不保证完美均匀，但大幅减轻 POI 类别（见图 A.2(b)）与运动上的长尾分布。选择之后，每个全景片段与 5 条对齐到 equirectangular 坐标系的逐帧相机旋转配对，共得到 12k 对全景片段与旋转序列。

![图 A.2(a). 多样性约束轨迹匹配前的 POI 类别分布。](../../../arxiv/geometry_control/ucpe/extracted/figure/category/before.png)

![图 A.2(b). 多样性约束轨迹匹配后的 POI 类别分布。](../../../arxiv/geometry_control/ucpe/extracted/figure/category/after.png)

**图 A.2. 多样性约束对 POI 类别分布的影响。** 原始 360° 全景片段在 POI 类别上高度长尾，由山地与街景等场景主导 (a)。在轨迹匹配期间施加多样性约束——限制片段/轨迹复用并强制语义平衡——之后，所得数据集更均衡 (b)，减少过度代表的类别，并改善下游视频合成的语义覆盖。

#### A.4 用多样相机做视频合成

从前一阶段起，每个全景片段配对五条迁移得到的旋转轨迹。为进一步丰富运动变化，我们对这些轨迹施加额外扰动与 panning 运动。随后用 Unified Camera Model（UCM）把全景帧投影到具有变化内参与畸变的虚拟相机，以合成相机多样视频。

**Camera Rotation Augmentation.** 对每条迁移轨迹，我们生成三个增强变体：

- **Consistent yaw perturbation：** 对所有帧加入从 $[-180^\circ, 180^\circ]$ 采样的随机 yaw 偏移。
- **Consistent yaw/pitch perturbation：** 分别从 $[-180^\circ, 180^\circ]$ 与 $[-80^\circ, 80^\circ]$ 采样随机 yaw 与 pitch 偏移，并在整段上均匀施加。
- **Smooth panning motion：** 在片段时长上加入平滑的 yaw–pitch–roll panning 曲线，起止偏移随机，采样范围为 yaw $[-90^\circ, 90^\circ]$、pitch $[-40^\circ, 40^\circ]$、roll $[-30^\circ, 30^\circ]$。

这些增强为每条轨迹产生三个额外旋转变体，共得到 $48\mathrm{k}$ 条独特相机运动。每条增强旋转序列为每个全景帧定义虚拟相机旋转 $\mathbf{R}^{\mathrm{aug}} \in \mathrm{SO}(3)$，稍后用于投影到期望视角。

**Unified Camera Model（UCM）。** 为合成具有多样内参与畸变的视频，我们采用 Unified Camera Model（UCM），它用单一畸变参数 $\xi$ 表示广泛的 central 相机。

**Projection.** 给定相机坐标系中的 3D 点 $\bm{p} = [p_x,\, p_y,\, p_z]^{\top}$，UCM 首先计算：

$$
r = \sqrt{p_x^{2} + p_y^{2} + p_z^{2}},
\qquad
\beta = p_z + \xi r,
\tag{A.3}
$$

再把该点投影到像素坐标：

$$
u = f_x\,\frac{p_x}{\beta} + c_x,
\qquad
v = f_y\,\frac{p_y}{\beta} + c_y,
\tag{A.4}
$$

其中 $f_x, f_y, c_x, c_y$ 表示相机内参。畸变参数 $\xi$ 控制投影非线性：当 $\xi = 0$ 时，UCM 退化为标准针孔模型；更大的 $\xi$ 引入更强的射线弯曲，使 UCM 能在统一表述中近似广角与 fisheye 镜头。

**Ray mapping.** 对像素坐标 $(u,v)$，我们先得到归一化图像坐标：

$$
x = \frac{u - c_x}{f_x},
\qquad
y = \frac{v - c_y}{f_y},
\tag{A.5}
$$

再在 UCM 下把 $(x,y)$ 提升回 3D 射线方向：

$$
\begin{aligned}
\bm{d}^{\mathrm{cam}}
&=
\frac{1}{\sqrt{x^{2} + y^{2} + \big(1 - \xi \rho\big)^{2}}}
\begin{bmatrix}
x \\ y \\ 1 - \xi \rho
\end{bmatrix},\\
\rho
&= \sqrt{x^{2} + y^{2}}.
\end{aligned}
\tag{A.6}
$$

由于 UCM 是 central 的，每条射线都从同一中心 $\bm{o}^{\mathrm{cam}} = \bm{0}$ 出发。

**FoV-based intrinsic re-parameterization.** 我们不通过 $f_x$ 与 $f_y$ 指定内参（这不够直观），而用水平视场 $\mathrm{xFoV}$ 重新参数化相机。对图像宽度 $W$，对应焦距为：

$$
f_x = f_y
=
\frac{W}{2}\,
\frac{\cos\gamma + \xi}{\sin\gamma},
\qquad
\gamma = \tfrac{1}{2}\,\mathrm{xFoV}.
\tag{A.7}
$$

在我们的表述中，始终假设主点位于图像中心，$c_x = \tfrac{1}{2}W,\; c_y = \tfrac{1}{2}H$，其中 $H$ 与 $W$ 表示图像高度与宽度。

**Final ray mapping and projection.** 使用由式 (A.7) 导出的焦距，式 (A.6) 的 UCM 反投影为每个像素 $(u,v)$ 定义 central-camera 射线

$$
\bm{d}^{\mathrm{cam}}_{u,v}
=
\Phi^{\mathrm{UCM}}\!\big(u,\,v;\, \mathrm{xFoV},\, \xi,\, H,\, W\big),
\tag{A.8}
$$

其中 $\Phi^{\mathrm{UCM}}$ 表示以 FoV 参数化的 UCM 射线映射函数。

类似地，3D 点 $\bm{p} \in \mathbb{R}^3$ 到图像平面的 UCM 投影由另一映射给出：

$$
(u,v)
=
\Pi^{\mathrm{UCM}}\!\big(\bm{p};\, \mathrm{xFoV},\, \xi,\, H,\, W\big),
\tag{A.9}
$$

它用同一 FoV 参数化内参施加式 (A.4)。这两个算子构成 UCM 的基础，并在整条管线中用于生成相机多样视角。

**Camera Intrinsics and Distortion Sampling.** 为覆盖广泛的相机几何，我们从若干依赖镜头的均匀范围中采样水平视场 $\mathrm{xFoV}$ 与 UCM 畸变参数 $\xi$。我们把相机组织为从针孔到极端 fisheye 的四类，并在各自区间内均匀抽取 $\mathrm{xFoV}$ 与 $\xi$：

- **Pinhole：** $\mathrm{xFoV} \in [90^\circ,\,110^\circ],\; \xi \in [0.0,\,0.0]$。
- **Wide-angle：** $\mathrm{xFoV} \in [110^\circ,\,140^\circ],\; \xi \in [0.5,\,0.95]$。
- **Fisheye：** $\mathrm{xFoV} \in [140^\circ,\,180^\circ],\; \xi \in [1.05,\,2.0]$。
- **Extreme fisheye：** $\mathrm{xFoV} \in [160^\circ,\,200^\circ],\; \xi \in [1.5,\,2.3]$。

该采样方案对真实内参与畸变水平提供广泛覆盖，使 UCPE 能跨多样相机类型学习一致的射线表示。

**Panoramic-to-UCM Projection.** 给定 equirectangular 全景 $I^{\mathrm{ERP}} \in \mathbb{R}^{H' \times W' \times 3}$，目标是通过把每条 UCM 射线映射到全景上的对应位置，把它投影到 UCM 虚拟相机视图。

对每个像素 $(u,v) \in \{1,\ldots,W\} \times \{1,\ldots,H\}$，我们先用式 (A.8) 得到 FoV 参数化的 UCM 射线 $\bm{d}^{\mathrm{cam}}_{u,v}$。施加虚拟相机旋转 $\mathbf{R}^{\mathrm{aug}} \in \mathrm{SO}(3)$，得到在 equirectangular 相机坐标系中表达的射线：

$$
\bm{d}^{\mathrm{ERP}}_{u,v}
=
\mathbf{R}^{\mathrm{aug}}\,
\bm{d}^{\mathrm{cam}}_{u,v}.
\tag{A.10}
$$

记射线为 $\bm{d}^{\mathrm{ERP}}_{u,v} = [d'_x,\, d'_y,\, d'_z]^{\top}$，其在全景上的对应位置由球面投影得到：

$$
(u',\, v')
=
\left(
\frac{\operatorname{atan2}(d'_x,d'_z)+\pi}{2\pi},\;
\frac{\arcsin(d'_y)+\tfrac{\pi}{2}}{\pi}
\right).
\tag{A.11}
$$

最终 UCM 渲染帧通过在这些坐标上采样全景得到：

$$
I^{\mathrm{UCM}}[u,v]
=
I^{\mathrm{ERP}}\!\big(u',\, v'\big),
\tag{A.12}
$$

从而得到与采样内参 $(\mathrm{xFoV},\xi)$ 及虚拟相机位姿一致的 UCM 图像。对全景片段中的所有帧重复该过程，得到最终合成视频。

**Virtual Camera Pose Composition.** 对每个渲染帧，equirectangular SLAM 系统提供 camera-to-world 位姿 $\mathbf{T}^{\mathrm{ERP}} \in \mathrm{SE}(3)$，我们把它与增强旋转 $\mathbf{R}^{\mathrm{aug}} \in \mathrm{SO}(3)$ 复合，得到最终虚拟相机位姿。记 $\mathbf{R}^{\mathrm{ERP}}$ 与 $\mathbf{t}^{\mathrm{ERP}}$ 为 $\mathbf{T}^{\mathrm{ERP}}$ 的旋转与平移。则虚拟 UCM 位姿为

$$
\mathbf{T}^{\mathrm{UCM}}
=
\begin{bmatrix}
\mathbf{R}^{\mathrm{ERP}}\,\mathbf{R}^{\mathrm{aug}}
&
\mathbf{t}^{\mathrm{ERP}}
\\
\mathbf{0}^{\top} & 1
\end{bmatrix},
\tag{A.13}
$$

它只替换朝向，同时保留原始轨迹平移。该位姿与采样内参 $(\mathrm{xFoV},\xi)$ 配对，用于相机感知训练。

**UCM Video Captioning.** 为给每个合成视频提供描述性 caption 以供 text-to-video 训练，我们用简单 prompt 适配 vLLM 模型：

```text
You are a helpful video captioning assistant.
Please describe this video in detail.
```

总计我们生成大约 $48\mathrm{k}$ 个 81 帧视频片段，分辨率为 $480 \times 832$、$16~\mathrm{fps}$。整个生成过程在单个 A800 GPU 节点上大约需要 3 天。评估时，我们把 CameraBench 的官方测试划分与其余 360° 视频匹配，并施加更严格的多样性约束，得到 $272$ 个测试片段。需指出：尽管我们的数据集用 UCM 合成，UCPE 表示与其他相机模型兼容，因为它以与模型无关的方式编码射线。

#### A.5 相机运动与内参统计

![图 A.3(a). 我们的数据集。](../../../arxiv/geometry_control/ucpe/extracted/figure/panshot/rose_direction.png)

![图 A.3(b). RealEstate10K。](../../../arxiv/geometry_control/ucpe/extracted/figure/re10k/rose_direction.png)

**图 A.3. 相机运动方向分布。** 我们用玫瑰图可视化 (a) 我们的数据集与 (b) RealEstate10K 的水平平移方向。相较 RealEstate10K 强烈的前向运动偏差，我们的数据集提供更丰富、更均匀分布的相机平移方向。

**Camera Motion Direction Distribution.** 为评估数据集中相机运动的多样性，我们可视化首尾帧之间平移方向的分布，如图 A.3。我们把合成数据集与 RealEstate10K 比较；后者是包含训练划分约 $66\mathrm{k}$ 片段的真实世界集合。为公平比较，我们丢弃少于 $81$ 帧的片段，并使用其余视频的前 $81$ 帧，得到 $47\mathrm{k}$ 个片段。如图 A.3(a) 与图 A.3(b) 所示，我们的数据集覆盖更均衡的平移方向范围，这很大程度上得益于增强期间引入的 consistent yaw perturbation；而 RealEstate10K 由房地产巡览特征性的前向平移主导。

![图 A.4(a). 我们的数据集。](../../../arxiv/geometry_control/ucpe/extracted/figure/panshot/fov_hist.png)

![图 A.4(b). RealEstate10K。](../../../arxiv/geometry_control/ucpe/extracted/figure/re10k/fov_hist.png)

**图 A.4. 相机视场分布。** 我们比较 (a) 我们的数据集与 (b) RealEstate10K 的垂直 FoV 分布。我们的数据集呈现宽 FoV 范围（60°–110°），反映合成期间采样的多样内参。相较之下，RealEstate10K 的分布狭窄并集中在约 60°，表明相机内参变化有限。

**Field of View Distribution.** 我们进一步比较我们的数据集与 RealEstate10K 的垂直视场（FoV）分布。如图 A.4，我们的数据集跨越从 60° 到 110° 的宽 FoV 范围，源于合成期间采样的多样内参。相较之下，RealEstate10K 大体集中在约 60°，反映其有限的内参变化。此外，我们的数据集在 Unified Camera Model（UCM）下纳入宽谱畸变水平，覆盖多种镜头类型；而 RealEstate10K 主要包含无畸变针孔相机。

### B. Absolute Orientation Encoding 的细节

如正文第 3.2 节所述，Lat-Up 表示由 latitude map 与 Up map 组成。Latitude 分量直接由世界空间射线方向用正文式 (8) 计算。本节详述 Up map 的推导。

对每个 token $t$，记 $(u_t, v_t)$ 为其在图像平面上的像素中心。使用收集在 $\phi$ 中的 FoV 参数化 UCM 内参，相机坐标系射线方向由式 (A.8) 定义的 UCM 射线映射算子得到：

$$
\bm{d}^{\mathrm{cam}}_t
=
\Phi^{\mathrm{UCM}}_{\phi}\!\big(u_t, v_t\big),
\qquad
\|\bm{d}^{\mathrm{cam}}_t\| = 1.
\tag{B.1}
$$

施加 camera-to-world 旋转 $\mathbf{R}$ 得到世界坐标系射线方向（正文式 (1)）：

$$
\bm{d}_t = \mathbf{R}\,\bm{d}^{\mathrm{cam}}_t,
\qquad
\|\bm{d}_t\| = 1.
\tag{B.2}
$$

**Ray perturbation direction towards world up.** Up map 通过考察每条视线 $\bm{d}_t$ 对朝世界上轴 $\bm{u}^{\mathrm{wld}}=[0,-1,0]^{\top}$ 的无穷小扰动如何响应来构造。直观上，该扰动反映若射线在单位观察球上朝世界上方向略微旋转会如何移动，其图像平面位移揭示与 token $t$ 相关联的 2D Up 方向。

在球面上，射线在该扰动下的瞬时运动必须落在同时正交于 $\bm{d}_t$ 与 $\bm{u}^{\mathrm{wld}}$ 的切向。该切向由叉积给出：

$$
\bm{k}_t = \bm{d}_t \times \bm{u}^{\mathrm{wld}},
\qquad
\hat{\bm{k}}_t = \frac{\bm{k}_t}{\|\bm{k}_t\|}.
\tag{B.3}
$$

单位向量 $\hat{\bm{k}}_t$ 因此指定把 $\bm{d}_t$ 朝世界上方向做无穷小旋转的正确轴，构成后续步骤所定义 Up map 的基础。

**Small-angle perturbation via Rodrigues' formula.** 我们把 $\bm{d}_t$ 绕 $\hat{\bm{k}}_t$ 旋转一个小的固定角度 $\delta$（例如 $\delta=0.1$ rad）。用 Rodrigues 公式，扰动后的世界空间射线方向为

$$
\begin{aligned}
\bm{d}^{\mathrm{rot}}_t
&=
\operatorname{Rot}\!\big(\hat{\bm{k}}_t,\delta\big)\,\bm{d}_t \\
&=
\bm{d}_t \cos\delta
+
\big(\hat{\bm{k}}_t \times \bm{d}_t\big)\sin\delta
+
\hat{\bm{k}}_t\,(\hat{\bm{k}}_t^{\top} \bm{d}_t)\,(1-\cos\delta).
\end{aligned}
\tag{B.4}
$$

随后把该扰动射线变回相机坐标系：

$$
\bm{d}^{\mathrm{cam,rot}}_t = \mathbf{R}^{\top} \bm{d}^{\mathrm{rot}}_t,
\tag{B.5}
$$

并用式 (A.9) 引入的 UCM 投影算子 $\Pi^{\mathrm{UCM}}_{\phi}$ 投影：

$$
\big(u^{\mathrm{rot}}_t, v^{\mathrm{rot}}_t\big)
=
\Pi^{\mathrm{UCM}}_{\phi}\!\big(\bm{d}^{\mathrm{cam,rot}}_t\big).
\tag{B.6}
$$

重要的是，原始 $(u_t, v_t)$ 是 token $t$ 的像素位置，而 $(u^{\mathrm{rot}}_t, v^{\mathrm{rot}}_t)$ 是射线朝世界上方向小旋转后的投影位置。

**Up map definition.** 诱导的图像平面位移为

$$
\Delta u_t = u^{\mathrm{rot}}_t - u_t,
\qquad
\Delta v_t = v^{\mathrm{rot}}_t - v_t,
\tag{B.7}
$$

token $t$ 处的 Up map 定义为

$$
\mathrm{Up}_t
=
\frac{
\left[\,\Delta u_t,\;\Delta v_t\,\right]
}{
\sqrt{\Delta u_t^2 + \Delta v_t^2}
}.
\tag{B.8}
$$

除几何定义外，所得 Up map 还提供强的外观依赖线索：其空间模式随相机绝对 pitch 与 roll 连贯变化，并对建筑与树木的垂直结构等场景语义作出响应。此外，由于扰动是通过投影函数 $\Pi$ 评估的，Up map 自然反映不同镜头的特征性弯曲。因此，它提供同时编码全局朝向、语义规律与镜头诱导畸变的统一视觉信号。

### C. 评估指标

正文第 4.1 节同时评估视频生成质量与相机可控性。此处详述用于相对位姿控制、绝对朝向与镜头可控性基准的相机相关指标计算。

**Relative Camera Pose Control.**

对每个生成视频，我们首先把每一帧校正到针孔投影。该步骤是必要的，因为现有位姿估计方法（包括 ViPE）在强镜头畸变与宽 FoV 弯曲下并不稳健，而这些在我们合成的 UCM 视图中频繁出现。使用 ground-truth UCM 参数 $(\mathrm{xFoV}, \xi)$，把每个畸变帧映射到有效水平 FoV 上限为 $100^\circ$ 的针孔图像。校正用 UCM 射线–像素算子 $\Phi^{\mathrm{UCM}}_{\phi}$ 与 $\Pi^{\mathrm{UCM}}_{\phi}$ 实现，它们在畸变像素与其底层 3D 射线之间提供正反映射。校正后的序列再送入 ViPE，以估计 camera-to-world 轨迹 $\{\hat{\mathbf{T}}^{\mathrm{wc}}_i\}$。

遵循 CameraCtrl 与 MotionCtrl，所有位姿指标都在 *relative* 轨迹上运算。对 camera-to-world 位姿

$$
\mathbf{T}^{\mathrm{wc}}_i
=
\begin{bmatrix}
\mathbf{R}^{\mathrm{wc}}_i & \mathbf{t}^{\mathrm{wc}}_i \\
\mathbf{0}^{\top} & 1
\end{bmatrix}
\in \mathrm{SE}(3),
$$

及其旋转 $\mathbf{R}_i$ 与平移 $\mathbf{t}_i$，对应的相对 ground-truth 位姿与估计位姿定义为

$$
\mathbf{T}_i
=
(\mathbf{T}^{\mathrm{wc}}_0)^{-1}
\mathbf{T}^{\mathrm{wc}}_i,
\qquad
\hat{\mathbf{T}}_i
=
(\hat{\mathbf{T}}^{\mathrm{wc}}_0)^{-1}
\hat{\mathbf{T}}^{\mathrm{wc}}_i.
\tag{C.1}
$$

- **Rotation Error（RotErr）：** 总旋转误差计算为逐帧角偏差之和：

$$
\mathrm{RotErr}
=
\sum_i
\arccos\!\left(
\frac{
\mathrm{Tr}\!\left(
\mathbf{R}_i^{\top}\hat{\mathbf{R}}_i
\right)-1
}{2}
\right).
\tag{C.2}
$$

- **Translation Error（TransErr）：** 对相对平移 $\mathbf{t}_i$ 与 $\hat{\mathbf{t}}_i$，

$$
\mathrm{TransErr}
=
\sum_i
\big\|
\mathbf{t}_i - \hat{\mathbf{t}}_i
\big\|_2 .
\tag{C.3}
$$

- **Camera Motion Consistency（CamMC）：** 把每个相对位姿的前三行展平为 12 维向量并计算

$$
\mathrm{CamMC}
=
\sum_i
\big\|
\operatorname{Vec}(\mathbf{T}_i)
-
\operatorname{Vec}(\hat{\mathbf{T}}_i)
\big\|_2 .
\tag{C.4}
$$

这些指标评估生成轨迹如何紧密跟随目标相机运动，而不依赖于绝对尺度或全局对齐。

由于不同基线生成的视频长度与帧率不同（例如 CameraCtrl 产生 16 帧、$4\,\mathrm{fps}$；AC3D 产生 48 帧、$8\,\mathrm{fps}$；而我们的方法输出 81 帧、$16\,\mathrm{fps}$），直接比较轨迹会受时间采样偏差影响。为公平评估相对相机位姿控制，我们在所有方法上均匀采样相同的 16 个时间戳来计算位姿指标。

**Absolute Camera Orientation and Lens Control.**

我们用 GeoCalib 从生成帧估计逐帧绝对朝向与镜头参数。对每帧 $i$，GeoCalib 输出预测的 pitch $\hat{\theta}_i$ 与 roll $\hat{\varphi}_i$、垂直视场 $\widehat{\mathrm{FoV}}$，以及经典径向模型下的径向畸变系数 $\hat{k}_1, \hat{k}_2$。

Ground-truth 参数 $(\theta_i^{\ast}, \varphi_i^{\ast}, \mathrm{FoV}^{\ast}, k_1^{\ast}, k_2^{\ast})$ 由原始 UCM 帧上的 GeoCalib 估计得到。随后计算下列指标：

- **Pitch / Roll Error：** 我们计算绝对误差

$$
|\hat{\theta}_i - \theta_i^{\ast}|,
\qquad
|\hat{\varphi}_i - \varphi_i^{\ast}|.
\tag{C.5}
$$

- **FoV and Distortion Errors：** 在径向 Brown 模型下，我们计算

$$
|\hat{\mathrm{FoV}} - \mathrm{FoV}^\ast|,
\qquad
|\hat{k}_1 - k_1^\ast|,
\qquad
|\hat{k}_2 - k_2^\ast|.
\tag{C.6}
$$

这些指标共同衡量对绝对相机朝向、镜头内参与畸变的可控性，为相机感知生成质量提供细致评估。

### D. 实现细节

![图 D.1. 基线与消融模型的实现。(a) ReCamMaster 在空间重复后把逐帧相机参数注入每个 Transformer block。(b) Wan CameraCtrl 用卷积 adapter 把 Plücker 编码的射线注入视频 token。(c) 我们的 UCPE 消融把 spatial attention adapter 插入到原始 self-attention 模块之前、之后，或与之并行。](../../../arxiv/geometry_control/ucpe/extracted/figure/pdf/baselines.png)

**Our Method.** 我们采用 Wan2.1-T2V-1.3B 作为基座。它由 text encoder、Diffusion Transformer 与 3D VAE 组成，共计 $7.3$ billion 参数。我们在合成数据集上微调模型，同时冻结全部原始权重，仅用 AdamW 以学习率 $1\mathrm{e}{-4}$ 训练所提出的 attention adapters，共 10k step。训练在 8 张 NVIDIA A800 GPU 上使用 batch size 8，大约一天完成。推理时，我们以 $480 \times 832$ 分辨率、$16~\mathrm{fps}$ 生成 $81$ 帧。由于尚无现有 text-to-video 模型支持完整相机控制（即相对位姿、绝对朝向与镜头参数），我们适配两种近期方法以供比较，详述如下。

**ReCamMaster.** 我们把官方 ReCamMaster 实现适配到带完整相机控制的 text-to-video 生成。如图 D.1(a)，ReCamMaster 把原始相机参数注入每个 Transformer block。在我们的适配中，对每帧 $i$，我们把归一化视场 $\mathrm{xFoV}'=\mathrm{xFoV}/180^\circ$ 与畸变参数 $\xi$ 拼接到展平的 $12$ 维位姿向量 $\operatorname{Vec}(\mathbf{T}^{\mathrm{wc}}_i)$。因为 Wan2.1-T2V-1.3B 在 VAE 中施加 $4\times$ 时间压缩，这些逐帧参数被下采样以匹配 token 序列长度 $n$。遵循原始设计，相机参数随后在所有空间 token（$h \times w$）上广播，得到形状为 $(nhw, 14)$ 的张量，经零初始化线性层投影到 token 维，并在每个 self-attention 层之前作为 bias 相加。一个恒等初始化的线性层在 self-attention 之后做特征投影。训练时，我们按 ReCamMaster 训练方案微调所加线性层以及原始 self-attention 层。

**Wan CameraCtrl.** 我们进一步把 Wan2.1-Fun-V1.1-1.3B-Control-Camera（CameraCtrl 在 Wan 上的第三方实现）适配到 text-to-video 生成。如图 D.1(b)，Wan CameraCtrl 把形状为 $(NHW, 6)$ 的 Plücker 编码射线（$N$ 为帧数，$H, W$ 为输入视频空间分辨率）经带 pixel-unshuffle 下采样的卷积 adapter 转为 token 形状特征，并通过加性 biasing 注入。我们保留该结构，但用式 (A.8) 的射线表述把 Plücker encoding 从针孔相机推广到 UCM 相机。为稳定训练，卷积与最终残差层初始化为零。我们按官方训练配方，以学习率 $1\mathrm{e}{-5}$ 微调卷积 adapter 与 diffusion transformer 的全部参数。

**UCPE Spatial Adapters Ablation.** 如正文第 3.3 节所述，我们把 spatial attention adapter 作为并行分支插入每个 Diffusion Transformer self-attention 层（见正文图 3）。在第 4.3 节，我们额外评估把 adapter 紧挨放在原始 self-attention 之前或之后的两个变体。三种变体汇总于图 D.1(c)，并共享同一 $1/8$ 压缩比。所有模型都在上述相同设定下训练。

**Inference Latency.** 我们在 NVIDIA A800 GPU 上测量 UCPE 与其他基线的推理延迟，如表 D.1。相较基座 Wan2.1 与其他相机条件 adapter，UCPE 引入的计算开销很小，同时生成比 CameraCtrl、AC3D 等更早方法显著更多帧的高分辨率视频。具体地，UCPE 在 $480 \times 832$ 分辨率下生成 81 帧用时 184 秒，与相同设定下的 ReCamMaster（179s）和 Wan CameraCtrl（177s）相当。

**表 D.1. 推理延迟比较。** 延迟在单张 NVIDIA A800 GPU 上测量，并给出对应帧数与分辨率以供参考。

| Method | Time (s) | Frames | Resolution |
| --- | ---: | ---: | --- |
| ReCamMaster | 179 | 81 | $480 \times 832$ |
| Wan CameraCtrl | 177 | 81 | $480 \times 832$ |
| CameraCtrl | 17 | 16 | $256 \times 384$ |
| AC3D | 632 | 48 | $480 \times 720$ |
| UCPE | 184 | 81 | $480 \times 832$ |

### E. 稳健性、泛化、局限性与未来工作

**Robustness to Text-Camera Conflict.** 我们进一步评估当文本提示与指定相机几何冲突时 UCPE 的稳健性。如图 E.1，我们用指定猫的 “flat, telephoto portrait” 的 prompt，对抗矛盾的 fisheye 相机（$\mathrm{xFoV} = 180^\circ, \xi = 2.0$）。尽管文本暗示窄视场与最小畸变，UCPE 仍成功强制所请求的 fisheye 几何，表明相机位置编码能提供强几何引导，覆盖文本中冲突的外观线索。

![图 E.1. 对文本–相机冲突的稳健性。即使文本提示指定 “telephoto” 视角，UCPE 仍按所给相机参数正确合成 fisheye 视频，化解文本与几何之间的冲突。](../../../arxiv/geometry_control/ucpe/extracted/figure/rebuttal/0-f74139ac48f19b3c-fov180-xi2.00-A_flat,_telephoto_portrait_of_a_fluffy_gray_cat_lo-0.png)

**Generalization to Unseen Camera Models.** UCPE 与模型无关，因为它学习通用的射线空间交互。对未观测的镜头类型，只需在推理时替换射线映射函数即可实现控制而无需微调，因为 attention 机制推理的是射线关系而非特定相机参数（例如 $\xi$）。我们在 OpenCV 用于物理镜头标定的复杂 Brown-Conrady 相机模型上评估该能力（$\mathrm{xFoV} = 100^\circ, k_1 = -0.2, k_2 = -0.05$）。如图 E.2，UCPE 成功泛化到该模型，在合成帧上加入写实桶形畸变，而无需额外训练。

![图 E.2. 对未见相机模型的泛化。无需微调，UCPE 只需在推理时替换射线映射函数，即可泛化到 Brown-Conrady 等未观测模型。](../../../arxiv/geometry_control/ucpe/extracted/figure/rebuttal/0-f74139ac48f19b3c-fov100-k1-0.20-k2-0.05-A_fluffy_gray_cat_lounging_on_a_garden_stone_wall_-0.png)

**Limitations.** UCPE 在训练时依赖相机位姿，目前只建模位姿、内参与畸变，尚未捕捉 zoom、focus 或 depth-of-field 等更丰富属性。此外，我们用 Equirectangular projection（ERP）的射线映射函数测试 UCPE，这是训练中未见的极端情形。如图 E.3，尽管 UCPE 在该极端设定下未能正确合成完整 360° 全景，我们期望在全景数据集上做针对性微调后其表现会改善。

![图 E.3. Equirectangular Projection 上的失败案例。用 ERP 射线映射测试会产生伪影，因为模型未在此类极端全景投影上训练。](../../../arxiv/geometry_control/ucpe/extracted/figure/rebuttal/0-f74139ac48f19b3c-ERP-A_fluffy_gray_cat_lounging_on_a_garden_stone_wall_-0.png)

**Future Works.** 把 UCPE 扩展到支持这些额外控制，并进一步降低对准确位姿监督的依赖（例如通过自监督几何损失），仍是有前景的方向。此外，尽管 T2V 生成是我们的主要关注，UCPE 也适用于 Image-to-Video（I2V）任务。与 T2V 不同，I2V 任务受益于输入图像的强刚性像素约束，它自然锚定镜头与初始朝向。因此 UCPE 支持 I2V 任务而无需 Absolute Orientation Encoding（AOE）：一旦镜头被定义（例如经 GeoCalib），仅 Relative Ray Encoding（RRE）即可驱动后续帧。为展示该可行性，我们微调 Wan2.1-I2V-14B。如图 E.4，我们的表示能顺利扩展到在更受约束的 I2V 设定中控制相机位姿。Video-to-Video（V2V）任务（例如 ReCamMaster）同样可能受益于 UCPE 增强的可控性。由于此类任务本质上需要合成的多轨迹数据集，我们把该探索留给未来工作。

![图 E.4. 应用到 I2V 任务。我们微调 Wan2.1-I2V-14B，以展示 UCPE 在更受约束的 Image-to-Video 任务上的可行性。](../../../arxiv/geometry_control/ucpe/extracted/figure/rebuttal/i2v.png)

### F. 补充视频

补充视频提供下列方面的可视化与演示：

- UCPE（Unified Camera Positional Encoding）及其两个分量的总览：Relative Ray Encoding 与 Absolute Orientation Encoding。
- 对内参与畸变可控性的演示。
- 对外参与初始相机朝向可控性的演示。
- UCPE 所启用的应用，包括电影内容创作、自动驾驶与具身智能。
- 来自我们大规模合成数据集的例子，覆盖多样内参、畸变剖面与相机运动（正文第 4.1 节，补充材料第 A 节）。
- 在我们合成数据集上与基线方法的比较，面向完整相机可控视频生成（正文图 4）。
- 在 RealEstate10K 数据集上的泛化结果，展示针孔相机设定下改善的相机可控性与视频质量（正文图 5）。
