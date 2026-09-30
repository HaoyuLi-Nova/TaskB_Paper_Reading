# Track: Long Video Generation

分层关键帧、免训练外推、token 流、LLM 分镜，以及两篇综述。
块级自回归 / Forcing 系在 [`../streaming/`](../streaming/)。

| slug | 论文 | arXiv | 路线 |
|------|------|-------|------|
| `lv_survey` | A Survey on Long Video Generation | 2403.16407 | 综述 |
| `storytelling_survey` | Long-Video Storytelling Generation | 2507.07202 | 综述 |
| `phenaki` | Phenaki | 2210.02399 | token 流 |
| `nuwa_infinity` | NUWA-Infinity | 2207.09814 | token 流 |
| `tats` | TATS | 2204.03638 | 分层 |
| `cogvideo` | CogVideo | 2205.15868 | 分层 |
| `nuwa_xl` | NUWA-XL | 2303.12346 | 分层 |
| `gen_l_video` | Gen-L-Video | 2305.18264 | 免训练 |
| `freenoise` | FreeNoise | 2310.15169 | 免训练 |
| `seine` | SEINE | 2310.20700 | 短到长 |
| `videodirectorgpt` | VideoDirectorGPT | 2309.15091 | LLM 分镜 |
| `videopoet` | VideoPoet | 2312.14125 | token 流 |
| `vlogger` | Vlogger | 2401.09414 | LLM 分镜 |
| `fifo_diffusion` | FIFO-Diffusion | 2405.11473 | 免训练 |
| `loong` | Loong | 2410.02757 | token 流 |
| `storydiffusion` | StoryDiffusion | 2405.01434 | 分层 |
| `moviegen` | Movie Gen | 2410.13720 | one-shot |
| `seedance` | Seedance 1.0 | 2506.09113 | 多镜头 one-shot |
| `storyanchors` | StoryAnchors | 2505.08350 | LLM 分镜 |

Sora 等闭源系统无 arXiv TeX，未入库。

译本：[`translations/long_video/`](../../translations/long_video/)  
阅读笔记：[`notes/long_video/<slug>/{QA,critique,note}.md`](../../notes/long_video/)  
下载：`python scripts/download_catalog.py --track long_video --with-tex`
