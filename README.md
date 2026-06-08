# 身体部位解耦的多流残差量化手语生成方法

(Multi-Stream Residual Quantization for Sign Language Production)

本仓库是研究生毕业论文的最终 release 版本，实现并复现两个手语数据集上的 strict gloss-free 文本到 3D 姿态手语生成 (Sign Language Production)，方法包含两个互补创新模块：

- **M1 (多流码本 Multi-Stream Codebook)** — 把 178 个关键点按身体部位 (body / face / hand) 分流，每个部位独立的 VQ 码本；
- **M2 (残差量化 Residual VQ)** — 在 VQ 量化器后追加 residual 量化层，逐级补偿主码本误差；
- **M1+M2 (MSR, Multi-Stream Residual)** — 两者结合：每个身体流上都做 base+residual 量化（共 6 个子流），构成本文最终方法。

数据集：

- **PHIX-14T** (德语手语，SLRTP-178 lift3d, 7K train / 515 dev / 641 test)
- **CSL-Daily** (中文手语，Ivashechkin lift3d, ~18K train / 1077 dev / 1176 test)

## 主要结果（SLRTP-canonical eval）

完整数据见 [`results/ABLATION_SUMMARY.md`](results/ABLATION_SUMMARY.md)。

### PHIX-14T TEST (n=641)

| 模型 | B1 | B4 | CHRF | ROUGE | WER↓ |
|---|---|---|---|---|---|
| Baseline | 24.81 | 6.86 | 26.35 | 25.65 | 94.03 |
| + M1 | 24.77 | 6.82 | 25.80 | 25.37 | 94.35 |
| + M2 | 27.04 | 7.88 | 27.79 | 27.88 | 91.16 |
| **+ M1 + M2 (ours)** | **29.19** | **8.97** | **29.51** | **30.25** | **90.14** |

### CSL-Daily TEST (n=1176)

| 模型 | B1 | B4 | CHRF | ROUGE | WER↓ |
|---|---|---|---|---|---|
| Baseline | 14.47 | 1.02 | 3.00 | 15.40 | 98.10 |
| + M1 | 16.24 | 1.91 | 3.62 | 17.26 | 96.83 |
| + M2 | 14.20 | 1.06 | 2.91 | 14.62 | 99.33 |
| **+ M1 + M2 (ours)** | **18.71** | **3.43** | **4.81** | **20.50** | **93.97** |

### PHIX 对比 (TEST, 统一 SLRTP-canonical 协议下复现)

所有对比方法都在**同一 178-kpt 数据、同一文本编码器 (mBART)、同一 SLRTP-canonical 反向翻译评测器**下复现，因此与本文 MSR **严格可比**（复现代码见 [`baselines/`](baselines/)）。

| 方法 | 范式 | B1 | B2 | B3 | B4 | CHRF | ROUGE | WER↓ |
|---|---|---|---|---|---|---|---|---|
| MDM (复现) | 扩散 | 23.88 | 12.02 | 8.07 | 6.17 | 25.42 | 23.45 | 98.42 |
| T2M-GPT (≈单流基线) | VQ+AR | 24.81 | 13.03 | 8.90 | 6.86 | 26.35 | 25.65 | 94.03 |
| MotionGPT (复现, mT5-587M) | LLM | 27.65 | 15.63 | 10.67 | 8.10 | 27.97 | 29.89 | **87.09** |
| MoMask (复现) | masked VQ | 28.79 | 16.18 | 11.23 | 8.67 | 28.51 | 29.43 | 91.55 |
| **本文 M1+M2 (MSR)** ★ | VQ+AR | **29.19** | **17.00** | **11.83** | **8.97** | **29.51** | **30.25** | 90.14 |

### CSL 对比 (TEST, 统一 SLRTP-canonical 协议下复现)

| 方法 | 范式 | B1 | B2 | B3 | B4 | CHRF | ROUGE | WER↓ |
|---|---|---|---|---|---|---|---|---|
| MDM (复现) | 扩散 | 12.90 | 3.38 | 1.21 | 0.39 | 2.50 | 13.53 | 100.18 |
| T2M-GPT (≈单流基线) | VQ+AR | 14.47 | 5.14 | 2.18 | 1.02 | 3.00 | 15.40 | 98.10 |
| MoMask (复现) | masked VQ | 13.73 | 5.02 | 2.26 | 1.19 | 3.06 | 15.73 | 96.66 |
| MotionGPT (复现, mT5-587M) | LLM | 13.55 | 5.41 | 2.42 | 1.18 | 3.11 | 16.25 | 96.40 |
| **本文 M1+M2 (MSR)** ★ | VQ+AR | **18.71** | **9.20** | **5.26** | **3.43** | **4.81** | **20.50** | **93.97** |

> **关于可比性**：上表所有 baseline 均由本文在统一协议下复现（见 [`baselines/`](baselines/)），与 MSR 严格可比。MSR 在两个数据集上几乎全部指标领先；尤其 MotionGPT 使用约 587M 的 mT5（约为本文 10 倍参数）仍未超过 MSR，说明优势来自结构设计而非模型容量。
>
> 跨论文的绝对 BLEU **不可直接比较**，因为各家反向翻译评测器不同：例如 GLOS 在其自训评测器下复现 MoMask 得 CSL B4 = 3.57，而同一方法在本文 SLRTP-canonical 评测器下仅 1.19。因此 T2S-GPT、Sign-IDD、DARSLP、A²V-SLP、GLOS、SignPR (CVPR 2026) 等已发表方法（各自使用不同评测器 / 姿态表示 / 监督设定）仅作相关工作引用，不纳入上述对照表。本文开源了 SLRTP 同协议自训的 `backTranslation_CSL_model`，便于后续工作在统一基线下相互对比。

---

## 目录结构

```
sign_slp_paper_release/
├── README.md                           本文档
│
├── code/                               全部源代码
│   ├── src/                            训练源代码
│   │   ├── models/                     模型定义
│   │   │   ├── vqvae.py                            baseline 单流 VQ-VAE
│   │   │   ├── vqvae_multistream.py                M1 多流 VQ-VAE (支持非对称码本)
│   │   │   ├── vqvae_residual.py                   M2 残差 VQ-VAE (单流)
│   │   │   ├── vqvae_multistream_residual.py       M1+M2 MSR
│   │   │   ├── t2m_trans_cross.py                  cross-attention AR transformer
│   │   │   ├── text_encoder_char.py                char-level text encoder (CSL)
│   │   │   ├── encdec.py / quantize_cnn.py         共享 1D conv encoder/decoder + EMA quantizer
│   │   │   └── ...
│   │   ├── dataset/                    VQ / Trans 数据集类
│   │   ├── train_vq_sign.py                        训练 baseline VQ
│   │   ├── train_vq_sign_ms.py                     训练 M1 多流 VQ
│   │   ├── train_vq_sign_rvq.py                    训练 M2 残差 VQ
│   │   ├── train_vq_sign_msr.py                    训练 M1+M2 MSR
│   │   ├── tokenize_sign{,_ms,_rvq,_msr}.py        VQ → token 缓存
│   │   ├── interleave_{ms,rvq,msr}_tokens.py       多流 token 交错为单序列
│   │   └── train_trans_sign_cross.py               训练 cross-attn AR transformer
│   │
│   ├── eval/                           评测源代码
│   │   ├── eval_cross_slt_lift3d.py    SLP 生成 (text → 3D pose pickle)
│   │   ├── slrtp_eval_phix.py          PHIX SLRTP-canonical 评测 wrapper
│   │   ├── slrtp_eval_csl.py           CSL SLRTP-style 评测 wrapper
│   │   ├── compute_vq_ceiling_bleu.py  VQ 上限测试 (GT pose → VQ → BT)
│   │   ├── compute_skeleton_validity.py 骨段稳定性 / 有效性分析
│   │   ├── bucket_analysis.py          按句长分桶 BLEU 分析
│   │   ├── diagnose_trans.py           Trans TF/Free accuracy + 长度匹配诊断
│   │   └── _debug/                     一次性诊断 / 历史 ckpt inspector
│   │
│   └── scripts/                        端到端 pipeline 脚本
│       ├── build_char_vocab.py         构建 CSL/PHIX char 词表
│       ├── 01_train_vq_csl.ps1         CSL VQ 训练入口
│       ├── 03_train_trans_csl.ps1      CSL Trans 训练入口
│       ├── 04_eval_all_csl.ps1         CSL 一键 eval
│       └── _train_phix_v2_pipeline.ps1 PHIX 一键 pipeline (tokenize+interleave+4 trans)
│
├── data/                               数据 (硬链接到原始位置，自包含)
│   ├── csl/
│   │   ├── csl_daily_lift3d.{train,dev,test}.pt   178-kpt × 3, sid → {text, gloss, poses_3d}
│   │   └── char_vocab/{txt,gls}.vocab              CSL 中文字符级词表
│   └── phix/
│       ├── phix_lift3d.{train,dev,test}.pt        178-kpt × 3, sid → {text, gloss, poses_3d}
│       └── char_vocab/{txt,gls}.vocab              PHIX 德语词表
│
├── checkpoints/                        全部模型权重 (按数据集分目录)
│   ├── csl/
│   │   ├── vq/vq_{baseline,M1,M2,M1M2}.pt
│   │   ├── vq/stats/                                   mean.npy / std.npy
│   │   ├── trans/trans_{baseline,M1,M2,M1M2}_large.pt
│   │   └── tokens/                                     train/dev/test tokens (含 interleaved)
│   └── phix/
│       ├── vq/vq_{baseline,M1,M2,M1M2}.pt
│       ├── trans/trans_{baseline,M1,M2,M1M2}.pt
│       └── tokens/                                     同上
│
├── bt_eval_kit/                        Back-translation 评测套件
│   └── slrtp_official/                 ★ canonical eval (1:1 镜像 walsharry/SLRTP-Sign-Production-Evaluation)
│       ├── main.py / metrics.py / back_translation/   SLRTP repo 源码
│       ├── backTranslation_PHIX_model/             PHIX BT model (best.ckpt + vocab + config)
│       ├── backTranslation_CSL_model/              CSL BT model (我们训练，同 signjoey 格式)
│       ├── data_official/                          SLRTP 原版 PHIX dev/test/train.pt (硬链接)
│       └── results/                                 BT eval JSON 结果 (每个 ckpt 一个)
│
├── results/                            实验结果与最终表
│   ├── ABLATION_SUMMARY.md             ★ paper 主表 (所有最终数字)
│   ├── phix_{baseline,M1,M2,M1M2}/     PHIX SLP 输出 pickle (dev.pickle / test.pickle)
│   ├── csl_{baseline,M1,M2,M1M2}/      CSL SLP 输出 pickle
│   ├── phix_vqceil_{baseline,M1,M2,M1M2}/  PHIX VQ ceiling 测试 pickle
│   └── _archive/                       中间实验 / 弃用结果备份
│
├── logs/                               训练日志 (每个 ckpt 一份 *.train.log)
└── _archive_v1_phix/                   PHIX 早期实验备份 (论文未使用)
```

---

## 环境准备

**单环境即可**：训练 + SLP 生成 + SLRTP 评测都跑在同一个 Python 环境里。所有依赖见根目录 [`requirements.txt`](requirements.txt)。本机验证用 Python 3.14 + torch 2.11+cu128 + RTX 4090。

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

> **注意（Py 3.14 用户）**：PyTorch 在 cu121/cu124 索引下**没有** Py 3.14 wheel，必须用 **cu128**。默认 PyPI 的 torch wheel 是 CPU-only 的，会让 `torch.cuda.is_available()` 返回 False，请显式带 `--extra-index-url`。如果 cu128 wheel（~2.8 GB）下载慢，手动从 https://download.pytorch.org/whl/cu128/ 拉到本地后 `pip install <path>`。

CPU-only 或 Python 3.11–3.13 用户：删掉 `--extra-index-url` 直接装即可，torch 会走默认 PyPI（cu121 等老 CUDA index 也支持这些 Python 版本）。

> 旧版本 README 提到"双环境"是历史遗留——SLRTP `back_translation/` 内置 Vocabulary，不再依赖 torchtext，所以训练侧的现代 torch 跟评测侧完全兼容，已并入同一环境。原 `bt_eval_kit/_legacy_signjoey/` (torch 1.4 + tf 2.1 的早期 signjoey CLI BT eval) 已删除。

---

## 数据 / Checkpoints 下载

GitHub 仓库只含代码 + README + 训练日志 + ablation 表（~31 MB）。其中 PHIX 数据、checkpoints、BT 评测模型公开打包到 Google Drive 按需下载；CSL-Daily 数据（`data_csl.zip`）受其原数据集使用协议约束，**不公开转发，需联系作者获取**：

| 包 | 大小 | 内容 | 链接 |
|---|---|---|---|
| `data_phix.zip` | 1.9 GB | `data/phix/` — PHIX-14T lift3d 数据 (train/dev/test) + 德语 char vocab | [下载](https://drive.google.com/file/d/1jV6O5I9ogh69jeB3mJFJf1ObzHV7GPI3/view?usp=sharing) |
| `data_csl.zip` | 4.9 GB | `data/csl/` — CSL-Daily lift3d 数据 (train/dev/test) + 中文 char vocab | (!) 受 CSL-Daily 原数据集使用协议约束，不公开转发；请联系作者（chengyaozhu91@gmail.com）获取 |
| `checkpoints.zip` | 7.8 GB | `checkpoints/` — 全部 VQ (4×2) + Trans (4×2) ablation 权重 + token 缓存 | [下载](https://drive.google.com/file/d/1THicz0DE_88TVOEdVOHe-g5zX_0t5Rbd/view?usp=sharing) |
| `bt_eval_official.zip` | 2.5 GB | `bt_eval_kit/slrtp_official/data_official/` (SLRTP 官方 PHIX 数据 + oracle GT + PT baseline preds) + 两个 `backTranslation_*_model/best.ckpt` (PHIX + CSL BT 模型权重) | [下载](https://drive.google.com/file/d/1DjQPYFe_z5p7mkNEVpDUzkHickSaAtfw/view?usp=sharing) |

按数据集组合：

- **只跑 PHIX 复现** = `data_phix.zip` + `checkpoints.zip` + `bt_eval_official.zip` ≈ 12.2 GB
- **只跑 CSL 复现** = `data_csl.zip`（需联系作者获取）+ `checkpoints.zip` + `bt_eval_official.zip` ≈ 15.2 GB（BT 模型在 bt_eval_official.zip 里）
- **完整复现** = 全部 4 个 ≈ 17.2 GB

### 解压步骤

下载完后把 zip 放到 clone 下来的项目根（README.md 所在那层），然后：

```powershell
# 用 7-Zip
& "C:\Program Files\7-Zip\7z.exe" x data_phix.zip
& "C:\Program Files\7-Zip\7z.exe" x data_csl.zip
& "C:\Program Files\7-Zip\7z.exe" x checkpoints.zip
& "C:\Program Files\7-Zip\7z.exe" x bt_eval_official.zip

# 或 PowerShell 自带（慢一些）
Expand-Archive data_phix.zip -DestinationPath .
# ...其他三个同理
```

zip 内部保留了完整的相对路径，解压后会自动叠加到 `data/`、`checkpoints/`、`bt_eval_kit/slrtp_official/` 等目录里，无需手动 mv。

> **注意：** `data/phix/phix_lift3d.{dev,test,train}.pt` 跟 `bt_eval_kit/slrtp_official/data_official/{dev,test,train}.pt` 在我们本地是硬链接（同一份数据用两个路径访问，省 ~2 GB 磁盘）。zip 不保留硬链接关系，解压后会得到两份独立拷贝（共 ~2 GB 冗余）。功能上不受影响。若想节省磁盘可手动改回硬链接，或把其中一处删掉后用 `mklink /H` 重建。

---

## 数据准备

`data/` 中的 .pt 文件结构（dict format）：

```python
{
    "sample_id_001": {
        "text":   "正午的太阳很热",                # 句子原文
        "gloss":  "今天 天气 热",                  # gloss 序列 (训练不使用, 仅记录)
        "poses_3d": torch.Tensor([T, 178, 3]),    # Ivashechkin lift3d 后的 3D pose (z=深度)
        "name":   "sample_id_001",
        "speaker": "..."                          # CSL 才有
    },
    ...
}
```

数据来源：

- **PHIX-14T (SLRTP-178)**: 公开 SLRTP CVPR 2025 challenge data ([下载](https://drive.google.com/file/d/1fjKHigsEWHwsMHnwwWdFYZ8dECXslTKi/view))，硬链接到 `data/phix/`。
- **CSL-Daily (lift3d)**: 我们用 [Ivashechkin et al.] 的 3D-lift 模型预处理 CSL-Daily MediaPipe Holistic 序列，硬链接到 `data/csl/`。**CSL-Daily 受其原数据集使用协议约束，本仓库不公开转发该数据及其 lift3d 衍生版本，请联系作者（chengyaozhu91@gmail.com）获取。**

---

## 训练流程

整个 pipeline 分 3 步：**(1) 训 VQ-VAE → (2) tokenize + interleave → (3) 训 transformer**。下面以 PHIX 为例。CSL 切换 `--dataname csl_lift3d`、`--lang zh_CN`、`--text-encoder char` 即可。

### Step 1: 训练 VQ-VAE

每个 ablation 训练自己的 VQ。下面以 **M1+M2 (MSR, hand-priority)** 为例：

```bash
cd code/src
python train_vq_sign_msr.py \
    --dataname phix_lift3d --exp-name vq_M1M2 \
    --nb-base-body 32 --nb-base-hand 512 --nb-base-face 128 \
    --nb-res-body  32 --nb-res-hand  512 --nb-res-face  128 \
    --code-dim 512 --output-emb-width 512 \
    --down-t 2 --stride-t 2 \
    --width 512 --depth 3 --dilation-growth-rate 3 \
    --vq-act relu --quantizer ema_reset --mu 0.99 --beta 1.0 \
    --commit 0.02 --loss-vel 0.1 --recons-loss l2 \
    --batch-size 128 --window-size 64 \
    --total-iter 100000 --warm-up-iter 1000 \
    --lr 2e-4 --lr-scheduler 50000 80000 --gamma 0.05 \
    --print-iter 500 --eval-iter 2000 --save-iter 20000 \
    --out-dir ../../checkpoints/phix/vq/
```

各 ablation 的训练脚本与关键参数：

| Ablation | 训练脚本 | 关键参数 (PHIX) | 关键参数 (CSL) |
|---|---|---|---|
| baseline | `train_vq_sign.py` | `--nb-code 512` | `--nb-code 4096` |
| M1 (multi-stream) | `train_vq_sign_ms.py` | `--nb-code-body 64 --nb-code-hand 1024 --nb-code-face 256` | `--nb-code 1024` (对称) |
| M2 (residual) | `train_vq_sign_rvq.py` | `--nb-code 512 --nb-code-residual 512` | `--nb-code 2048 --nb-code-residual 2048` |
| M1+M2 (MSR) | `train_vq_sign_msr.py` | `--nb-base-* 32/512/128 --nb-res-* 32/512/128` | `--nb-base-* 512 --nb-res-* 512` (对称) |

PHIX 通用：`--down-t 2 --stride-t 2` (frame-rate 较低)；CSL 通用：`--down-t 1 --stride-t 2`。

训练在 RTX 4090 上单卡 ~5-15 分钟（早停）。

### Step 2: Tokenize + Interleave

把训好的 VQ 应用到 train/dev/test，得到 token 缓存：

```bash
# baseline (单流, 无需 interleave)
python tokenize_sign.py --dataname phix_lift3d \
    --vq-ckpt ../../checkpoints/phix/vq/vq_baseline.pt \
    --stats-dir ../../checkpoints/phix/vq/stats \
    --out-dir ../../checkpoints/phix/tokens/vq_baseline

# M1 / M2 / M1M2 (多流, 需要 interleave 把多流 token 合成单序列)
python tokenize_sign_ms.py --dataname phix_lift3d \
    --vq-ckpt ../../checkpoints/phix/vq/vq_M1.pt \
    --out-dir ../../checkpoints/phix/tokens/vq_M1
python interleave_ms_tokens.py \
    --in-dir  ../../checkpoints/phix/tokens/vq_M1 \
    --out-dir ../../checkpoints/phix/tokens/vq_M1_interleaved \
    --dataname phix_lift3d \
    --nb-body 64 --nb-hand 1024 --nb-face 256
# (interleave_rvq / interleave_msr 类似，传对应 codebook 大小)
```

### Step 3: 训练 Transformer

```bash
python train_trans_sign_cross.py \
    --dataname phix_lift3d \
    --tokens-dir ../../checkpoints/phix/tokens/vq_M1M2_interleaved \
    --vq-ckpt    ../../checkpoints/phix/vq/vq_M1M2.pt \
    --exp-name trans_M1M2 \
    --text-encoder mbart --mbart-name facebook/mbart-large-50 --lang de_DE \
    --num-vq 1344 \
    --embed-dim 512 --text-dim 1024 \
    --num-layers 9 --n-head 8 --block-size 320 \
    --drop-out-rate 0.1 --fc-rate 4 \
    --batch-size 16 --total-iter 20000 --warm-up-iter 1000 \
    --lr 1e-4 --lr-scheduler 10000 15000 --gamma 0.1 \
    --freeze-text 1 --weight-decay 1e-5 \
    --out-dir ../../checkpoints/phix/trans/
```

`--num-vq` 必须等于该 VQ 的**总** codebook 大小：

- PHIX baseline 512，M1 = 64+1024+256 = 1344，M2 = 512+512=1024，M1+M2 = (32+32)+(512+512)+(128+128) = 1344
- CSL baseline 4096，M1 = 3×1024=3072，M2 = 2048+2048=4096，M1+M2 = 6×512=3072

CSL 训练差异：

- `--dataname csl_lift3d --text-encoder char --lang zh_CN`
- `--text-dim 512 --num-layers 8 --block-size 480` (M1+M2)
- `--total-iter 50000 --lr-scheduler 20000 35000 --gamma 0.3`

每个 trans 在 RTX 4090 上 ~10-30 分钟。

### 一键 pipeline (PHIX)

```powershell
# PowerShell: 跑完所有 4 个 ablation 的 tokenize + interleave + train
./code/scripts/_train_phix_v2_pipeline.ps1
```

---

## 评测流程（SLRTP-canonical）

### Step 1: SLP 生成 (text → 3D pose)

```bash
cd code/eval
python eval_cross_slt_lift3d.py \
    --variant msr \
    --vq-ckpt    ../../checkpoints/phix/vq/vq_M1M2.pt \
    --trans-ckpt ../../checkpoints/phix/trans/trans_M1M2.pt \
    --splits dev,test \
    --out ../../results/phix_M1M2/ \
    --dataset phix --lang de_DE \
    --temperature 0.9 --top-k 20 \
    --rep-penalty 1.5 --max-run 4 --rep-streams 6 \
    --max-len 200
```

`--variant`: `base` / `ms` / `rvq` / `msr` (对应 baseline / M1 / M2 / M1+M2)
`--rep-streams`: 1 / 3 / 2 / 6 (per-stream rep penalty 跟踪的流数)

输出 `dev.pickle` / `test.pickle`，每个是 list of dict `{name, signer, gloss, text, sign}`，`sign` 为 (T, 534) 张量。

### Step 2: SLRTP-canonical BT eval (PHIX)

```bash
python slrtp_eval_phix.py \
    --pred-pickle  ../../results/phix_M1M2/test.pickle \
    --gt-pt        ../../bt_eval_kit/slrtp_official/data_official/test.pt \
    --bt-model-dir ../../bt_eval_kit/slrtp_official/backTranslation_PHIX_model \
    --slrtp-repo   ../../bt_eval_kit/slrtp_official \
    --tag phix_M1M2_test --fps 25 \
    --out-dir ../../results/phix_M1M2/
```

结果保存为 JSON: `bt_eval_kit/slrtp_official/results/phix_M1M2_test.json`，含 bleu1-4 / chrf / rouge / wer / dtw_mje / total_distance / avg_duration。

### Step 2': SLRTP-style BT eval (CSL)

CSL BT 模型的 `skeleton_subsample=1`，跟 PHIX 不同 (`=2`)，所以走专门 wrapper：

```bash
python slrtp_eval_csl.py \
    --pred         ../../results/csl_M1M2/slp_pickles/csl_daily.test \
    --gt-pt        ../../data/csl/csl_daily_lift3d.test.pt \
    --bt-model-dir ../../bt_eval_kit/slrtp_official/backTranslation_CSL_model \
    --tag csl_M1M2_test \
    --out-dir ../../results/csl_M1M2/ \
    --no-subsample
```

中文文本会自动 char-tokenize 后参与 sacrebleu。

### VQ Ceiling (理论上限)

```bash
# 1) 用 GT pose 跑 VQ encode→decode→pickle
python compute_vq_ceiling_bleu.py \
    --variant msr \
    --vq-ckpt ../../checkpoints/phix/vq/vq_M1M2.pt \
    --dataset phix --splits dev,test \
    --out ../../results/phix_vqceil_M1M2/

# 2) 然后用 slrtp_eval_phix.py 跑 BT 评测同上
```

### 骨架稳定性诊断

```bash
python compute_skeleton_validity.py \
    --pred-pickle ../../results/phix_M1M2/test.pickle \
    --gt-pickle   ../../data/phix/phix_lift3d.test.pt
```

输出每个骨段 (7 body + 20 LH + 20 RH = 47 bones) 的时间方差系数 CV 和无效率 (3σ from GT)。

---

## 关键设计 (论文 contributions)

### M1: 多流码本 (Multi-Stream Codebook)

PHIX-178 / CSL-178 的 keypoint layout 是 8 body + 128 face + 42 hand。
单一全局 VQ 必须同时表征 body 慢动作 / hand 高频 / face 表情，码本利用率不均衡。
M1 把 3 部分**完全解耦**：

- 各自有独立的 encoder block + EMA-reset 量化器 + decoder block (后期 fuse 到统一 z)
- **codebook 大小可不对称**（PHIX 用 hand 1024 / face 256 / body 64，hand-priority，因为手语核心信息在手部）
- token 序列按 (body₁, face₁, hand₁, body₂, face₂, hand₂, …) 交错送给 AR transformer

### M2: 残差量化 (Residual VQ)

单层 VQ 只能表征到码本最近邻精度。M2 引入二级 codebook 补偿残差：

```
encoder → z → quant_base(z) = ẑ_base
                              ↓
                              residual = z - ẑ_base
                              quant_residual(residual) = ẑ_res
                              z_q = ẑ_base + ẑ_res
                              decoder(z_q) → x̂
```

AR 阶段，token 序列变为 (base₁, res₁, base₂, res₂, …)。

### M1 + M2 (MSR)

最终配置：每个 body / face / hand stream 都做 base + residual = **6 个子流**。AR token 序列 6 路交错：

```
(body_base₁, body_res₁, hand_base₁, hand_res₁, face_base₁, face_res₁, body_base₂, …)
```

每个子流 codebook 大小独立：PHIX 用 (32, 32, 512, 512, 128, 128) = 1344 codes 共 6 路。

### Per-stream Repetition Penalty (推理 trick)

AR sampling 容易让某个 stream 退化到重复某一 token (mode collapse)。我们扩展标准 repetition penalty 为 **per-stream tracking**：position k 对应的 stream s = k % S，penalty 只考虑该 stream 中前 N 个 token，不跨流污染。

---

## 评测协议 (SLRTP-canonical)

为了跟 SLRTP Sign Production Workshop (CVPR 2025) [walsharry/SLRTP-Sign-Production-Evaluation](https://github.com/walsharry/SLRTP-Sign-Production-Evaluation) 协议完全对齐，所有数字都用以下固定设置：

| 设置项 | 值 |
|---|---|
| BT model | signjoey-format, PHIX 用挑战赛官方 / CSL 用我们同协议自训练 |
| Decoding | beam=3, length-norm α=−1 (固定，不做 dev sweep) |
| BLEU | `sacrebleu.raw_corpus_bleu` (无内置 tokenization) |
| CHRF | `sacrebleu.metrics.CHRF` |
| ROUGE | SLRTP 内置 (ROUGE-L) |
| WER | `jiwer` (英文标准 transforms + lowercase + remove punct) |
| Pose subsample | PHIX: ::2 (`fps=25` → 12.5fps) ; CSL: 无 (`skeleton_subsample=1`) |
| 中文处理 | CSL 文本在 BLEU/CHRF/ROUGE 计算前 char-tokenize (空格分字符) |

**注**：前期实验有用 signjoey CLI 自带的 `test` 命令（含 dev beam sweep），但发现 sacrebleu 与 signjoey 内置 BLEU 实现不一致，PHIX 上 signjoey 会**低估** ~2.5 BLEU。最终全部以 SLRTP main.py 协议为准。

---

## 模型 ckpt 详细配置

### PHIX (mBART text encoder, hand-priority 不对称 MSR)

| Variant | VQ 配置 | Trans 配置 |
|---|---|---|
| baseline | `vqvae.VQVAE_251`, nb_code=512, down_t=2 | mBART + 9L cross-attn, num_vq=512, block=160 |
| M1 | `MultiStreamVQVAE`, body=64 / hand=1024 / face=256 | num_vq=1344, block=320 |
| M2 | `ResidualVQVAE`, base=512 + res=512 | num_vq=1024, block=160 |
| **M1+M2** | `MultiStreamResidualVQVAE`, base/res each (body=32 / hand=512 / face=128) | num_vq=1344, block=320 |

公共: code_dim=512, width=512, depth=3, dilation=3, l2 recon。Trans: embed=512, text_dim=1024 (mBART hidden), heads=8, fc=4, dropout=0.1, 20K iter, lr=1e-4 [10K,15K] γ=0.1。

### CSL (char text encoder, 对称 MSR)

| Variant | VQ 配置 | Trans 配置 |
|---|---|---|
| baseline | nb_code=4096, down_t=1 | char + 8L cross-attn, num_vq=4096, block=160 |
| M1 | 3 streams × 1024 = 3072 | num_vq=3072, block=480 |
| M2 | base=2048 + res=2048 | num_vq=4096, block=160 |
| **M1+M2** | 6 substreams × 512 = 3072 | num_vq=3072, block=480 |

Trans: embed=512, text_dim=512, heads=8, fc=4, dropout=0.1, 50K iter, lr=1e-4 [20K,35K] γ=0.3。

---

## FAQ

### Q1: 为什么 PHIX 和 CSL 用不同的 text encoder？
A: mBART 在德语 (PHIX) 上有强 pretrained prior；CSL 中文为了避免分词不一致 + 跟 SLRTP signjoey 协议的 char-level vocab 对齐，从头训 2 层 char encoder。两套都保持 strict gloss-free。

### Q2: 为什么 PHIX 用 hand-priority 不对称码本，CSL 用对称码本？
A: PHIX 7K train 样本数据有限，**手部信息密度最高**，把 codebook 都给手部利用率更高。CSL 18K 训练充分，对称 split 已足够。

### Q3: 复现训练为什么需要双 Python 环境？
A: 不需要。早期版本因为 SignJoey BT 模型基于老 PyTorch + torchtext 训练而写了双环境步骤，但 SLRTP repo 内置了独立 `back_translation/` 不依赖 torchtext，已实测 Py 3.14 + torch 2.11+cu128 单环境同时承载训练和评测。统一依赖见根目录 `requirements.txt`。

### Q4: gloss 字段是干什么的？
A: 数据集自带 gloss 标注但我们**完全不使用**（strict gloss-free 设定）。仅在 SLRTP eval 计算 ROUGE-L 时作为参考字段。

### Q5: 怎么从头训 CSL BT eval 模型？
A: 见 `slrtp_eval_kit/slt_train/scripts/`（外部目录，未打包到 release）。BT 模型在 lift3d 178-kpt 上从头训练 100K iter，约 4 小时 RTX 4090。

---

## 未采用的探索 (Negative results)

完整 release 内附带两组未进入论文主结果的对比实验，便于复现验证 / future work 参考。

### 1) Discrete Flow Matching (DFM) 作为生成器

在共享同一套 M1+M2 VQ tokenizer + mBART 文本编码器的前提下，把 AR transformer 替换为 **discrete flow matching**（参考 Stark et al. NeurIPS 2024 的离散 token 速度场建模），用 classifier-free guidance 做推理：

- `code/src/train_dfm_sign.py` — 训练 (cross-attn DFM transformer)
- `code/src/models/t2m_dfm_cross.py` — 模型
- `code/eval/eval_dfm_phix.py` — 推理 + SLRTP eval
- `code/scripts/sweep_dfm_inference{,_v2}.ps1` — 推理超参 sweep (steps / CFG / temperature / len_mult)

**PHIX M1+M2 tokens, SLRTP-canonical TEST B4**：

| 方法 | steps | CFG | T | TEST B4 |
|---|---|---|---|---|
| AR (论文主方法) | — | — | — | **8.97** |
| DFM s50_cfg3_len10 | 50 | 3.0 | 1.0 | 7.60 |
| DFM s24_cfg3_len12 | 24 | 3.0 | 1.0 | 7.45 |
| DFM s24_cfg3 | 24 | 3.0 | 1.0 | 7.43 |
| DFM s24_cfg3_t08 | 24 | 3.0 | 0.8 | 7.34 |
| DFM s24_cfg2 | 24 | 2.0 | 1.0 | 6.82 |
| DFM s16_cfg2 | 16 | 2.0 | 1.0 | 7.22 |
| DFM s24_cfg1 | 24 | 1.0 (no CFG) | 1.0 | 4.80 |
| DFM s24_cfg4_len10 | 24 | 4.0 | 1.0 | 6.67 |

**结论**：同 tokenizer 下，DFM 最佳配置（50 步 + CFG=3）TEST B4 仍比 AR 低约 1.4。AR 在 PHIX 这种 7K 低数据量场景下，凭借 strong-prior + per-stream rep penalty 的稳健性显著占优。DFM 代码保留以备 future work / 更大数据量再评估。DFM ckpts 因负结果未上传 Drive，需要复现可重训（参数同 AR transformer，约 17K iter ≈ 17 分钟 RTX 4090）。

### 2) AR 解码超参 sweep

为说明论文采用 `T=0.9 / top_k=20 / rep_penalty=1.5 / max_run=4` 的合理性，跑了 4 组对照（同 M1+M2 ckpt）：

| config | T | top_k | rep_penalty | max_run | DEV B4 | TEST B4 |
|---|---|---|---|---|---|---|
| **default (论文)** | 0.9 | 20 | 1.5 | 4 | **9.28** | **8.97** |
| lowtemp | 0.7 | 20 | 1.5 | 4 | 6.92 | 7.37 |
| widetop | 0.9 | 50 | 1.5 | 4 | 6.06 | 5.77 |
| norep | 0.9 | 20 | 1.0 | ∞ | 7.16 | 6.54 |

低温过度收敛、放宽 top_k 增加噪声、关掉 rep penalty 让流陷入 mode collapse——三者都明显劣化。Sweep 脚本：`code/scripts/sweep_phix_decoding_slrtp.ps1`。

---

## 引用

```bibtex
@mastersthesis{zhu2026sign_slp,
  title={Multi-Stream Residual Quantization for Sign Language Production},
  author={Zhu, Chengyao},
  year={2026},
  school={Graduate School},
}
```

后端 SLRTP 评测套件来自：

```bibtex
@inproceedings{walsh2025slrtp,
  title={SLRTP Sign Production Challenge},
  author={Walsh, H. et al.},
  booktitle={CVPR Workshop},
  year={2025}
}
```

---

## 联系

实验细节 / 复现问题：作者 (chengyaozhu91@gmail.com)。

数据 / ckpt 不在 GitHub release 里（体积过大），下载链接见上面"数据 / Checkpoints 下载"一节。
