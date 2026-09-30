# Track: Spatial Understanding

视觉语言模型的空间理解。与视频生成主线正交。权威条目在 [`papers.yaml`](../../papers.yaml) 的 `spatial`。

两条线在 2026 年汇合。一条只看 RGB 图像或视频，让模型自己推断远近、左右和换视角后的布局。另一条直接吃点云、多视角或重建出的几何，做场景问答和语言定位。当前重点是第三件事：推理时在模型内部形成三维表示，或者把前馈三维重建和语言模型接到一起。

P0 是当前精读。P1 是把这五篇读懂所需要的上下文。P2 等十一篇读完再看。

## 先建立问题（P1）

| 顺序 | slug | 论文 | 先看什么 |
|------|------|------|----------|
| 1 | `spatial_vlm` | SpatialVLM（arXiv 2024） | 问题定义，以及以米为单位的距离、大小、相对位置是怎么自动造出来的 |
| 2 | `spatial_rgpt` | SpatialRGPT（NeurIPS 2024） | 区域指代和显式深度；SpatialRGPT-Bench 是后面表格的对照 |
| 3 | `thinking_in_space` | Thinking in Space（CVPR 2025 Oral） | VSI-Bench。普通思维链提不动空间题；先画认知地图，相对距离会好一截 |
| 4 | `mindcube` | MindCube（ICCV 2025） | 看不见的那一侧：心理地图、视角转换、假设性移动。3DThinker 的直接前奏 |

## 场景级三维（P1）

读的时候对照三张表：ScanQA 是场景问答，SQA3D 是带处境的空间推理，ScanRefer 是语言定位物体。

| 顺序 | slug | 论文 | 先看什么 |
|------|------|------|----------|
| 5 | `llava_3d` | LLaVA-3D（ICCV 2025） | 给二维图块加三维坐标，再用二维视觉语言模型的训练方式做场景问答、描述和定位 |
| 6 | `video_3d_llm` | Video-3D LLM（CVPR 2025） | 三维场景当成视频，加上三维位置编码。多视角图像加位置可以代替直接吃点云 |

PointLLM、3D-LLM、LEO、SceneVerse 是这一支的前史。表格里看到这些名字时再回头翻，不单独入库。

## 当前精读（P0）

| 顺序 | slug | 论文 | 先看什么 |
|------|------|------|----------|
| 7 | `spatialtree` | SpatialTree（CVPR 2026） | 先用这篇画地图：感知、心理地图、心理模拟、智能体执行，共 27 个子能力。低层少想，高层多想 |
| 8 | `think_with_3d` | Think with 3D / 3DThinker（CVPR 2026） | 方法上最值得精读。推理时生成三维隐变量，先和 VGGT 几何对齐，再用答案对错强化整条轨迹 |
| 9 | `g2vlm` | G²VLM（CVPR 2026） | 重建和问答的汇合点。训练用多视角图像和视频，推理时用学到的几何特征预测三维属性 |
| 10 | `spacemind` | SpaceMind（CVPR 2026） | 看融合模块。VGGT 管几何，InternViT 管二维语义，相机用来给空间 token 加权和门控 |
| 11 | `hispatial` | HiSpatial（CVPR 2026） | 和 SpatialTree 对着读。层次相同，这篇偏数据和监督微调，SpatialTree 偏评测和强化学习 |

## 十一篇之后（P2）

| slug | 论文 | 和前面哪篇对照 |
|------|------|----------------|
| `spatial_llm` | SpatialLLM（CVPR 2025 Highlight） | 插在 SpatialRGPT 和 HiSpatial 之间：三维位置和朝向该加在哪一训练阶段 |
| `think3d` | Think3D（arXiv 2026，预印本） | 和 3DThinker 一对：一个把三维藏在隐变量里，一个把三维放到重建工具里 |
| `star_r1` | STAR-R1（CVPR 2026） | 3DThinker 的多视角强化学习变体 |
| `spatial_ssrl` | Spatial-SSRL（CVPR 2026） | 用自监督可验证奖励做空间强化学习 |
| `sympl` | Keep it SymPL（CVPR 2026） | 用符号投影布局做以场景为中心的推理 |

评测留在 Thinking in Space 的 VSI-Bench 和 SpatialTree 的 SpatialTree-Bench 里，不另开评测论文。高斯场景生成、服装生成、驾驶世界模型、机器人动作是下游，不进这个 track。

译本：`translations/spatial/<slug>/paper_zh.md`  
阅读笔记：`notes/spatial/<slug>/{QA,critique,note}.md`  
下载：`python scripts/download_catalog.py --track spatial`
