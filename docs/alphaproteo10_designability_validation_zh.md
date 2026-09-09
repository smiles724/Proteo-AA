# AlphaProteo10 designability validation

## 目标

在完全一致的 10-target、binder length、seed、PXDesign sampler 和 AF2-IG
过滤协议下比较：

| Backbone | Sequence | 用途 |
| --- | --- | --- |
| 官方 PXDesign | ProteinMPNN | 复现 PXDesign two-stage baseline |
| Proteo-AA | ProteinMPNN | 单独衡量我们生成的 backbone 是否有设计性 |
| Proteo-AA | AA head `final` | 衡量最终推理 sigma 的 sequence |
| Proteo-AA | AA head `target_sigma` | 衡量训练覆盖较好的 sigma≈0.4 readout |
| Proteo-AA | AA head `confidence_best` | 衡量逐 residue 最高 confidence readout |

后三个 Proteo-AA AA 臂和 ProteinMPNN 臂来自同一条 diffusion trajectory，
只替换 CIF 中的 binder residue identity。因此 backbone 完全相同，差异可以归因于
sequence strategy。

严格 designability 定义为所有生成设计中同时满足以下条件的比例：

- AF2-IG unscaled ipAE `< 7.0`；
- binder pLDDT `> 0.9`；
- AF2 prediction 与设计 binder backbone RMSD `< 1.5 Å`。

评估失败或缺少分数的设计按失败计入分母，避免只汇报成功完成 AF2 的样本。

## 环境边界

Generation 使用当前 `proteoaa` 环境和 Protenix 2.0。AF2-IG/ProteinMPNN
scoring 必须使用独立的 PXDesignBench v0.1.2 环境，因为官方 evaluator 依赖
Protenix 0.5、JAX、ColabDesign，不能安装进当前训练环境。

conda 环境和 PXDesignBench 源码体积较小，可以留在 home；模型权重必须放在
`/hai/scratch/shenjm`。先安装环境：

```bash
git clone --branch v0.1.2 https://github.com/bytedance/PXDesignBench.git \
  /hai/users/s/h/shenjm/tools/PXDesignBench-v0.1.2

cd /hai/users/s/h/shenjm/tools/PXDesignBench-v0.1.2
bash install.sh --env pxdbench --pkg_manager conda --cuda-version 12.1
```

不要直接运行官方 `download_tool_weights.sh`：它还会下载 binder 流程不使用的约
8 GB ESMFold 权重。使用仓库中的纯 CPU job，只下载/校验 AF2 和 ProteinMPNN：

```bash
cd /hai/users/s/h/shenjm/Proteo-AA
mkdir -p logs/setup
sbatch scripts/utilities/slurm_download_pxdesignbench_binder_weights.sh
```

该 job 的权重目录和临时目录分别是
`/hai/scratch/shenjm/pxdesign_tool_weights` 与 `/hai/scratch/shenjm/tmp`；home 中只保留
conda env 和约数 MB 的 PXDesignBench 源码。

安装后应存在：

```text
<pxdbench env>/bin/python
/hai/scratch/shenjm/pxdesign_tool_weights/af2/params_model_1.npz
/hai/scratch/shenjm/pxdesign_tool_weights/mpnn/vanilla_model_weights/
```

## 第一步：paired generation

先跑 smoke，每个 target 一条、固定长度 105：

```bash
cd /hai/users/s/h/shenjm/Proteo-AA

PROTEOAA_CHECKPOINT=/path/to/our/stage3/checkpoints/stepNNNN.pt \
PROTEOAA_LABEL=proteoaa_stepNNNN \
NUM_DESIGNS_PER_TARGET=1 \
FIXED_LENGTH=105 \
N_STEP=400 \
RUN_TAG=smoke_stepNNNN \
bash scripts/evaluation/submit_alphaproteo_generation.sh
```

该命令提交两个 job：官方 PXDesign generation 和 Proteo-AA generation。终端打印的
`RUN_ROOT` 必须保存，后续所有阶段共用。

正式 benchmark 去掉 `FIXED_LENGTH`，让每个模型使用完全相同的 80–130 uniform
length schedule：

```bash
PROTEOAA_CHECKPOINT=/path/to/our/stage3/checkpoints/stepNNNN.pt \
PROTEOAA_LABEL=proteoaa_stepNNNN \
NUM_DESIGNS_PER_TARGET=328 \
LENGTH_MIN=80 LENGTH_MAX=130 \
N_STEP=400 SEED=42 \
RUN_TAG=full_stepNNNN \
bash scripts/evaluation/submit_alphaproteo_generation.sh
```

如果要严格复现 A-CODE/PXDesign 的 TNFa 行，应先决定采用 PXDesign 扩展 hotspot
集合 `A31,A32,A113,C73,C87`；仓库当前 YAML 默认保留 AlphaProteo Table S1 的窄集合
`A113,C73`。二者不能混在同一张对照表中。

## 第二步：ProteinMPNN + AF2-IG

generation 完成后先提交 4 个 scoring task，避免触发账户的
`MaxSubmitJobsPerAccount`：

```bash
RUN_ROOT=/hai/scratch/shenjm/proteo_aa_runs/alphaproteo10_designability/<RUN_TAG> \
PXDBENCH_DIR=/hai/users/s/h/shenjm/tools/PXDesignBench-v0.1.2 \
PXDBENCH_PYTHON=/path/to/miniconda3/envs/pxdbench/bin/python \
TOOL_WEIGHTS_ROOT=/hai/scratch/shenjm/pxdesign_tool_weights \
TASK_START=0 TASK_COUNT=4 MAX_CONCURRENT=4 \
bash scripts/evaluation/submit_alphaproteo_scoring.sh
```

脚本会打印总 task 数以及下一批的 `TASK_START`。上一批完成后，用相同命令依次提交
`TASK_START=4,8,12,...`。已经存在 `sample_level_output.csv` 的 task 会自动跳过，
所以中断后可以安全重跑。

scoring task 的定义是一个 `(checkpoint, target, sequence arm)`。每个 task 内所有
design 使用同一个 AF2-IG 配置；ProteinMPNN 每个 backbone 默认生成一条 sequence。

## 第三步：汇总

所有 scoring task 完成后：

```bash
RUN_ROOT=/hai/scratch/shenjm/proteo_aa_runs/alphaproteo10_designability/<RUN_TAG> \
sbatch scripts/evaluation/slurm_summarize_alphaproteo_designability.sh
```

主要输出：

- `summary/designability_by_target.csv`：每个 target/模型/sequence arm；
- `summary/designability_summary.csv`：pooled 和 target-macro designability；
- `summary/designability_summary.json`：过滤阈值、缺失值规则及机器可读结果。

## 解读顺序

1. 先比较 `PXDesign+MPNN` 与 `Proteo-AA+MPNN`：若后者下降，问题主要在 backbone。
2. 再比较同一 Proteo-AA backbone 的 `MPNN` 和三个 AA readout：差值主要反映 AA head。
3. 必须同时看 `coverage`；本流程将缺失 scoring 计为失败，但 coverage 过低时不能下模型结论。
4. smoke 只验证流程，不估计 designability。每 target 一条的结果不能与论文百分比比较。
