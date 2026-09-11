# PXDesign 怎么做靶点条件化，以及我们已经做到了什么

写于 2026-09-11，分支 `sjm/stage3-binder-design`。
起因是 [`alphaproteo10_designability_result_zh.md`](alphaproteo10_designability_result_zh.md)
里 Proteo-AA 组 0/3280 的结果，以及随之提出的「Stage III 缺靶点条件化」假设。

**结论先说：那个假设不成立。Stage III 已经忠实实现了 PXDesign 的靶点条件化，
采样器参数也逐项一致。没有「cogenerate 的问题」需要修。** 下面是证据。

## PXDesign 官方是怎么做的

来源：`PXDesign/assets/technical_report.pdf` p24（Appendix C，本地就有这份 PDF）。
注意 `PXDesign/` 是 **ByteDance 官方仓库**（`github.com/bytedance/PXDesign`），
不是复现；复现的是我们自己的 `pxdesign_train/`（其 `pyproject.toml` 写明
"Unofficial training code … Reconstructed from the technical report"）。

报告原文（重点为本文所加）：

> During training, the coordinates of **all atoms** are perturbed with noise and
> subsequently denoised through a learned diffusion process. Target residues are
> **soft-conditioned through pairwise features**. Specifically, the single-token
> condition `s` is initialized by embedding basic residue-level features, such as
> amino acid identity, hotspot annotations, etc. The pairwise condition `z` is
> initialized by embedding **binned pairwise distances derived from the target
> structure**. If no distance information is available for a residue pair, we
> assign it to a special bin. **Because binned pairwise distances offer strong
> structural constraints, we find it unnecessary to freeze the coordinates of
> target residues during training.** Instead, the model learns to recover
> structure directly from these embedded pairwise signals. For the
> condition-based generation task, unlike previous methods that relied on
> inpainting, PXDesign-d directly generates the coordinates of all atoms from
> noisy structure. Throughout the process, **no additional constraints are
> imposed on the noise.**

拆成四条：

1. **不固定 receptor 坐标** —— 明确说「没必要」。
2. **所有原子一起加噪去噪**，receptor 也在内。
3. 条件化走 **pair 表示 `z`**：靶点结构的 binned 成对距离；无距离的 pair 进一个特殊 bin。
4. 单 token 表示 `s` 带残基级特征（氨基酸身份、hotspot 等）。
5. 明确**对立于 RFdiffusion 的 inpainting**，也不对噪声施加任何额外约束。

官方实现对得上（`PXDesign/pxdesign/model/embedders.py:172`）：

```python
class ConditionTemplateEmbedder(nn.Module):
    def __init__(self, c_templ_in: int = 64 + 1, c_z: int = 128):
        self.embedder = nn.Embedding(self.c_templ_in, self.c_z)
    def forward(self, input_feature_dict):
        conditional_templ = input_feature_dict["conditional_templ"]
        pair_mask = input_feature_dict["conditional_templ_mask"]
        conditional_templ = pair_mask * (1 + conditional_templ)   # bin 0 = 无距离
        return self.embedder(conditional_templ)
```

即 64 个距离 bin + 1 个特殊 bin，嵌入进 `z`。

## 我们已经有了

`pxdesign_train/data/_helpers.py:105` 的 docstring 自己写明是对官方
`get_condition_template_feature` 的 **byte-for-byte** 复现（只把 bin 范围参数化），
默认 `no_bins=64, min_bin=2.0, max_bin=22.0`，也用同样的「masked pair 读 bin 0」约定。

**运行时验证**（用 Stage IV smoke 存下的真实 batch，48 token / 24 设计位）：

| pair 类型 | `conditional_templ_mask` 覆盖 |
| --- | --- |
| target–target | **576 / 576（100%）** |
| target–design | 0 / 576 |
| design–design | 0 / 576 |

target–target 上 bin 取值 0–63（552 个非零），design 相关 pair 全为 bin 0。
**完全是 PXDesign 的设计。**

采样器参数也一致（运行时读 `parse_configs(training_configs)`，即 benchmark 实际用的）：

| | 官方 `configs_base.py` | 我们实际用的 |
| --- | --- | --- |
| `gamma0` | 1.0 | 1.0 |
| `gamma_min` | 0.01 | 0.01 |
| `noise_scale_lambda` | 1.003 | 1.003 |
| `N_step` | 400 | 400 |
| `eta_schedule` | `piecewise_65`, 1.0→2.5 | 同 |

`configs_train.py:10` 直接 `from pxdesign.configs.configs_base import configs`，
所以这些值是继承来的。`step_scale_eta` 在 config 里不存在是正常的 ——
`PXDesign/pxdesign/model/pxdesign.py:97-99` 会从 `eta_schedule` 注入。

## 所以之前的判断错在哪

我在这个问题上连续错了两次，记下来免得再走：

1. **「benchmark 接错了函数，该换成目标条件化的采样器」** —— 错。PXDesign 自己就是
   自由生成所有原子（binder 任务 400 步，和我们一致），`cogenerate` 是对的路径。
2. **「Stage III 缺靶点条件化」** —— 也错。它有，而且是忠实复现的。我当时只查了
   `stage4_fixed_context`（坐标冻结）这一条路径，就断定没有条件化，
   **漏掉了 pair 表示这条真正的路径**。

顺带一个值得告诉 yfsun 的推论：**Stage IV 的坐标冻结
（`generator.py:101` 的 `stage4_fixed_context`）是相对 PXDesign 的一处偏离**，
不是补齐。报告明确说冻结「没必要」。这不代表 Stage IV 做错了 —— 固定靶点能省去
重建靶点的负担，对一个冻结序列头的 phase 可能反而更稳 —— 但这是一个有意识的
设计分歧，应该知道它存在。

## 那 0% 还剩什么解释

架构和协议都对上了，剩下的是**训练成熟度**，而且差距很大：

| | PXDesign-d | Proteo-AA 111408 |
| --- | --- | --- |
| 训练方式 | from scratch，两阶段课程 | 从 Stage II warm start |
| crop | 640 | 448 |
| batch / diffusion batch | 64 / 8 | 1 item / 1 |
| 步数 | —（完整训练） | **6650 / 30000，超时中断** |
| lr | 5e-4 | 见 111408 报告 |
| 数据 | PDB(≤2021) + AFDB + MGnify 蒸馏 | PDB 复合体 + PINDER |

其中**最可能直接对应观测到的失败模式**的是课程差异。报告 p24：

> PXDesign-d is trained from scratch in **two stages**. In the **first stage, we
> upweight monomer-only distillation data, enabling the model to learn the
> fundamentals of protein backbone geometry in an unconditional setting**. In
> the second stage, we gradually shift the sampling distribution toward
> experimentally resolved PDB complexes and target-conditioned design tasks.

而 Stage III 的 monomer 比例是**恒定 0.25**（`--stage2-start-monomer-frac 0.25
--stage2-end-monomer-frac 0.25`），没有 monomer 加权的第一阶段。Stage II 虽然是
monomer，但那是**侧链** warmup，不是 backbone 生成。

我们观测到的恰恰是「不会最基本的主链几何」：CA–CA 键长中位 2.75 Å（应 3.8），
93.9% 越界，`alpha 0.00 / beta 0.00 / loop 0.96`。**这正是 PXDesign 第一阶段
存在的理由所要解决的问题。**

## 已测：问题早于 Stage III，而且 Stage III 在变好

用同一个 harness、同靶点、同 seed、同 400 步跑 Stage III 的 warm-start 起点
（Stage II `fixed_global_decay_from_50k/step52500.pt`，job 114649）：

| checkpoint | bhrf1 CA–CA 中位 / 坏键 / 最短 / Rg | pdl1 |
| --- | --- | --- |
| PXDesign official | 3.83 / **0.0%** / 2.93 / 13.7 | 3.81 / **0.0%** / 3.74 / 12.9 |
| **Stage II 52500** | **1.82 / 98.1% / 0.12 / 6.8** | **1.74 / 97.8% / 0.39 / 6.9** |
| Stage III 111408 s6000 | 2.96 / 85.2% / 0.50 / 15.6 | 2.89 / 88.8% / 0.43 / 14.1 |

两个靶点一致，结论有两条：

**1. 问题完全早于 Stage III。** Stage II 的自由生成是彻底塌缩的 —— 104 残基的链
Rg 只有 6.8 Å（正常应 ~13），相邻 CA 最短 0.12 Å，98% 键长越界。那不是蛋白，
是一团点。

这本来就该预料到：**Stage II 是侧链 warmup，从来没有被训练过、也从来没有被评估过
「从噪声生成主链」这件事**。Stage III 直接从它 warm start，等于在一个不会生成主链的
底座上开始训 binder 设计。

**2. Stage III 的 6650 步是在变好，只是远远不够。** 坏键 98.1% → 85.2%，
Rg 6.8 → 15.6（从塌缩恢复到大致正确的尺寸）。方向是对的，量级不够。

这直接对上 PXDesign 报告 p24 那句话 —— 他们的第一阶段 upweight monomer 蒸馏数据，
就是为了「learn the fundamentals of protein backbone geometry in an unconditional
setting」。**我们跳过了这一阶段**，Stage II 用 monomer 但做的是侧链，不是主链生成。

## 下一步

判别已经做完，方向清楚了：**缺的是无条件主链生成的训练阶段，不是靶点条件化。**

1. **按 PXDesign 补一个 monomer 加权的主链生成阶段**，在进 binder 任务之前。
   报告的做法是第一阶段上调 monomer-only 蒸馏数据（AFDB + MGnify），
   学会无条件主链几何后再逐步转向复合体。我们现在是恒定 0.25 monomer，
   而且 Stage II 练的是侧链。
2. **判据用无条件 monomer designability**，不要用 binder。PXDesign 有现成协议
   （报告 Figure 2a：长度 200–1400，ProteinMPNN-CA + ESMFold，scRMSD < 2）。
   在主链几何没达标之前，任何 binder 数字都读不出信息 —— 这次 0/3280 就是例子。
3. **规模差距也要正视**：PXDesign crop 640 / batch 64 / diffusion batch 8 /
   from scratch；我们 crop 448 / batch 1 / 6650 步超时。即使课程补对，
   这个量级的训练预算差异也要先想清楚。

顺带记一个 worktree 的坑：`benchmarks/alphaproteo10/.gitignore` 忽略
`structures/`，所以新建 worktree 里那 10 个 PDB 结构不存在，而报错信息是
`could not parse <file>.cif` 而不是「文件缺失」，会把人引向解析逻辑。
软链主仓库的那份即可。另外别手搓已有 launcher 的等价命令 ——
`slurm_generate_alphaproteo_designability.sh` 传了 `--sampler-mode pxdesign_native`，
漏掉它会静默换成 `minimal_euler`，测的就不是同一条轨迹了。
