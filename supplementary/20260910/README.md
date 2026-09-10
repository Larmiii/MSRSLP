# 2026年9月补充实验

本目录提供独立补充实验的代码、配置、完整指标和复现入口。实验包括人体部位分组对照、单层大码本与两层残差量化对照，以及骨段统计、固定后处理和生成器适配分析。原 release 的实现与结果继续保留在仓库原有目录中。

补充实验采用本目录记录的配置，结果单独报告。运行时需要对应的数据、词表、量化器和生成器权重，不能用其他配置的检查点替代。这里没有上传论文、原始数据、逐帧生成姿态或模型权重。

## 目录

- [`runtime/`](runtime/)：补充实验使用的训练、量化、token处理与模型代码。
- [`reproduce.py`](reproduce.py)：VQ训练、token准备及AR训练入口，默认打印命令，显式指定`--execute`才执行。
- [`configs/`](configs/)：配置、依赖及数据约定。
- [`tools/`](tools/)：生成、重建、回译及成对统计入口。
- [`analysis/`](analysis/)：骨段统计与固定后处理。
- [`advanced/`](advanced/)：掩码生成器与参考方法适配。
- [`results/README.md`](results/README.md)：各结果文件的字段、来源及计算口径。

## 结果概览

主配置使用训练种子42，分组及码本对照使用42、43、44。下表是本目录配置下的完整测试集回译BLEU-4。

| 数据集 | 基线 | PAMVQ | TLRVQ | PAMRVQ |
|---|---:|---:|---:|---:|
| PHOENIX14T | 6.79 | 7.47 | 6.92 | 8.46 |
| CSL-Daily | 0.93 | 2.44 | 1.40 | 3.81 |

等token分组对照中，每个潜在位置均输出6个token，码字总数同为1344，名义索引预算同为42 bit，分支规模和共享解码器一致。PHOENIX14T上，人体部位分组的整体和双手位置误差分别降低7.16%和9.54%，面部误差增加8.71%。三次BLEU-4均值为固定随机分组8.35、人体部位分组8.79。固定三对模型平均差的追加样本区间为[-0.03, 0.91]，包含零。三次训练的样本标准差与该区间分别报告。

等码字存储量对照中，PHOENIX14T比较单层1024与两层512+512，CSL-Daily比较单层4096与两层2048+2048。两层配置的整体位置误差均值分别下降22.02%和21.14%，三个种子的变化方向一致。该对照匹配码字存储量，token数和名义比特预算仍不同。

其他补充分析也保留完整结果。例如，CSL-Daily的PAMRVQ生成长度CV降低13.78%，相邻帧长度相对变化增加24.02%。固定平滑后投影使PAMRVQ测试BLEU-4从3.81变为4.22，基线则从0.93变为0.81。长度统计只描述关键点间距变化，不等于人工自然度评价。

PHOENIX14T上，本地掩码生成器测试BLEU-4为6.62，AR为8.46。适配参考中MoMask为8.65、MDM为5.69、MotionGPT风格模型为8.08。各生成器及参考模型的规模、文本编码、训练预算和长度机制不同，完整条件见相应记录，结果不作为严格同训练预算的排名。

## 数据与环境

本轮记录的训练环境为Python 3.12、PyTorch 2.11.0+cu128、Transformers 4.46.3及RTX 5090。依赖见[`configs/requirements.txt`](configs/requirements.txt)。GPU与CPU入口的用途不同，运行前先查看相应`--help`。

沿用原仓库的数据获取方式准备文件。`--data-root`指向以下目录的根部，也可使用`SLP_SIGN_DATA_DIR`配置数据位置：

```text
DATA_ROOT/
  phix/
    phix_lift3d.train.pt
    phix_lift3d.dev.pt
    phix_lift3d.test.pt
  csl/
    csl_daily_lift3d.train.pt
    csl_daily_lift3d.dev.pt
    csl_daily_lift3d.test.pt
    char_vocab/txt.vocab
```

两数据集的点序分别在配置中指定，载入和输出保留各自的数据顺序。PHOENIX14T的有效训练、验证、测试样本数为7060、515、641，CSL-Daily为18400、1077、1176。第一阶段使用满足64帧条件的训练与验证样本，完整序列评测保留各划分全部有效样本。

第一阶段的训练和验证目标采用归一化坐标，公开重建误差则在反归一化后的完整测试序列上计算，单位为数据集坐标单位。回译使用冻结评测器，PHOENIX14T按既定流程下采样一次，CSL-Daily采用字符级文本处理。WER按原始编辑统计报告，可能超过100%。

## 核心实验入口

在本目录运行。先列出配置并查看命令，确认数据位置和输出目录：

```bash
python reproduce.py list
python reproduce.py vq --dataset phix --variant pamrvq --seed 42 --data-root /path/to/data --run-root /path/to/runs
```

确认命令后加入`--execute`执行对应阶段：

```bash
python reproduce.py vq --dataset phix --variant pamrvq --seed 42 --data-root /path/to/data --run-root /path/to/runs --execute
python reproduce.py tokenize --dataset phix --variant pamrvq --seed 42 --data-root /path/to/data --run-root /path/to/runs --execute
python reproduce.py ar --dataset phix --variant pamrvq --seed 42 --data-root /path/to/data --run-root /path/to/runs --execute
```

token准备包含多流交错及完整长度、截断标记的写入，AR训练据此处理人工截断末端的EOS。生成上限为320个原始动作帧，EOS只在完整token组的边界出现。样本生成随机状态由固定生成种子、数据集、划分和样本ID派生。

E1使用PHOENIX14T的`pamrvq`和`pamrvq_random`，各运行42、43、44三个种子。E2-PHX使用`single_large`和`tlrvq`，E2-CSL使用`baseline`和`tlrvq`。E2比较第一阶段重建，不要求为每个重复训练AR模型。

生成与评价的路径参数见`tools/`内各入口的`--help`。骨段和后处理步骤见[`analysis/README.md`](analysis/README.md)，掩码生成器及适配参考见[`advanced/README.md`](advanced/README.md)。这些入口需要相应数据和检查点，本仓库中的汇总指标不是可直接替代输入的模型文件。

迁移字符编码器检查点时，使用`--char-vocab-path`指定内容一致的词表。旧检查点未记录词表哈希时，还需通过`--char-vocab-sha256`提供对应哈希。回译入口要求显式指定`--evaluator-dir`，指向包含相应权重、配置和词表的冻结评测器目录。

下列测试仅使用小型CPU样例，检查数据路径、token截断标记、对照参数和后处理计算，不训练模型：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python analysis/test_cpu.py
```

## 结果与来源

[`results/`](results/)保留完整浮点精度、各随机种子、验证与测试划分及对应条件。展示表格进行舍入，差值和相对变化由未舍入结果计算。来源文件哈希与公开结果文件哈希分别记录，用于区分原始运行记录与整理后的公开表格。

公开副本只对文件位置、命令入口和配置读取作可移植性处理。模型与指标计算按对应源码实现。第三方实现的来源和原有权利声明继续保留，数据和预训练权重的获取与使用条件以各发布方为准。
