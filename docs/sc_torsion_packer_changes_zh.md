# 改动清单：新增了什么，改了哪些 .py，该盯哪几处

> 分支 `sjm/sc-apm-torsion-packer`（从 `sjm/sc-chi-output` 分出），
> worktree `/hai/scratch/shenjm/wt_torsion_packer`。
> 方法与实验设计见 [`sc_torsion_packer_apm_zh.md`](sc_torsion_packer_apm_zh.md)，
> 这一份只回答「代码在哪、我该看哪里」。
>
> 总计 **22 个文件，+2963 / −41 行**。其中 2034 行是新文件，改动到现有文件的
> 只有约 500 行，而且大部分是注释。

## 一、新增文件

| 路径 | 行数 | 是什么 |
| --- | ---: | --- |
| `pxdesign_train/sidechain/packer.py` | 482 | **`TorsionPacker`** —— APM `SideChainModel` 的移植，加可插拔的序列通道。核心新代码。 |
| `pxdesign_train/sidechain/ipa.py` | 388 | IPA / `StructureModuleTransition` / `EdgeTransition` / `AngleResnet` / AF2 的 `Linear` 初始化器。移植自 `apm/models/ipa_pytorch.py`（Apache-2.0，带版权头）。 |
| `pxdesign_train/sidechain/plm.py` | 147 | 冻结 ESM-2 650M 的包装 + APM 的 `plm_s_combine` / `plm_s_mlp`。 |
| `pxdesign_train/sidechain/torsion_loss.py` | 185 | $\mathcal L_\chi$（AF2 Alg. 27）+ GT χ 目标 + stage4 胶水。 |
| `tests/test_sc_torsion_packer.py` | 551 | 19 个测试。 |
| `tests/test_sc_ipa.py` | 167 | 6 个测试：IPA 不变性、mask 语义、rotvec log map、设备搬迁、BuildSC 梯度。 |
| `tests/test_sc_only_aa_backend.py` | 114 | 7 个测试：`sc_only` 不建 AA head 的护栏。 |
| `scripts/training/slurm_sc_torsion_packer_hai.sh` | 84 | 四臂 array launcher（none / a_token / plm / both）。 |
| `scripts/training/slurm_sc_torsion_packer_smoke_hai.sh` | 42 | 30 步四臂 smoke。 |
| `docs/sc_torsion_packer_apm_zh.md` | — | 方法、与 APM 的逐项异同、验收标准、踩过的坑。 |

新文件本身不改变任何默认行为：`sidechain.torsion_packer` 默认 `False`，
`--aa-backend` 默认 `fampnn`。不打开开关，这些代码一行都不会执行。

## 二、改到的 .py 文件，按「该盯的程度」排序

### 1. `pxdesign_train/sidechain/buildsc.py`（+17）—— 最需要看

只有一处，但它**影响既有代码**：

- **127 行**：二面角计算前，把 inactive 行的四个点换成一个非退化 tetrad。
  没有 χ_k 的残基把 slot 0 索引四次，`atan2(0,0)` 前向是 0、**反向是 NaN**，而
  `torch.where` 消不掉（NaN × 0 仍是 NaN）。
  **这条 bug 同样存在于既有的 `chi_output` 路径上**（共用 `BuildSC`），只是它的
  小样例测试恰好在 NaN 传到参数前把它 mask 掉了。真实 batch 上的症状是
  `FloatingPointError: Nonfinite gradient before optimizer update 0`。

### 2. `pxdesign_train/model.py`（+168 −10）

| 行 | 内容 | 为什么值得看 |
| ---: | --- | --- |
| **503** | **`c_atom` 被改绑成 packer 的 node 宽度** | `HResFeedback`、`ATokenFusion`、`QAtomBSFusion` 全从这一个变量构造。packer 是残基级、宽度是自己的 256，不改绑就会在**第一次 feedback 调用**时形状不匹配。 |
| 491–530 | 构造分支。每个开关都用**字面量 key** 读（不是 helper） | `tests/test_train_inference_parity.py` 靠扫 `getattr(sc_cfg, "<key>"` 证明没有开关能静默改变训练行为；写成 lambda 会把它们藏起来。 |
| 332–348 | `torsion_packer` 开关 + 与 `edm`/`chi_output`/`template_residual` 的互斥 raise | 那三个描述的是 Cartesian 模块怎么消费带噪侧链输入，packer 没有这个输入。 |
| 232 / 243–258 | `packing_stack` 属性；`aa_backend == "sc_only"` 分支与两个 raise | `packing_stack` 是这次引入的新概念，全仓库的 Stage IV 判断都改读它。 |
| 1151 / 1176 / 1179 | 三处 gate 从 `aa_backend == "fampnn"` 改成 `packing_stack` | 第三处漏改，`sc_only` 会掉进 backbone 扩散训练路径，而且不报错。 |
| 1861–1870 | `_fa` 对 packer 恒真 | packer 在残基 frame 里预测扭转，缺 frame 时 raise，不静默降级。 |
| 2000–2011 | 传 `residue_index`（packer 专属 kwarg） | APM 的 node 有残基序号嵌入、edge 有相对位置项。 |
| 2090 | 把 `last_torsions` 写进 `out["sc_pred_chi_*"]` | 坐标契约没动，L_χ 是**加**上去的一项。 |

### 3. `scripts/training/train_protenix_monomer.py`（+95 −3）

| 行 | 内容 |
| ---: | --- |
| **2017** | **`seed_all(int(configs.seed))`，在建模型之前** —— 原本 `--seed` 只到数据采样器，模型初始化没被 seed，于是只差一个架构开关的两臂**初值也不一样**。必须含 numpy：AF2 式 `Linear(init=...)` 走 scipy 的 truncnorm。 |
| 1019–1045 | 打开 packer 时强制关掉描述 Cartesian 模块的开关，只保留 `bb_context` / `type_logits_input`。写在 `SCRATCH_SC_LAYOUT` **之后**，顺序不能反。 |
| 1047–1075 | `sc_only` 分支与两条前置 raise。 |
| 1754–1790 | 新 CLI：`--aa-backend`、`--sc-torsion-packer`、`--sc-packer-seq-cond`、`--sc-packer-plm-checkpoint`、尺寸开关、`--stage4-weight-sc-chi`、`--detect-anomaly`。 |

守卫测试要求每个 `--sc-*` flag 被显式赋值进某个 `configs.*`，**不能写成循环赋值**。

### 4. `pxdesign_train/sidechain/torsion_loss.py` 的一处（新文件，但这行是踩坑修的）

- `norm = (raw.square().sum(-1) + eps).sqrt()`，用 openfold 的形式而不是
  `vector_norm` + clamp。两者只在 `raw == 0` 处不同，而 masked 行的 `raw`
  **恰好是 0**（AngleResnet 残差分支零初始化 + bias 零），此时 `vector_norm`
  的梯度是 NaN。

### 5. `pxdesign_train/stage4.py`（+26 −5）

| 行 | 内容 |
| ---: | --- |
| 17 | `SUPERVISED_SC_PHASES` 常量。 |
| 211–216 | `supervised_sc_forward` 里加 L_χ 与 `torsion/*` 指标。 |
| **257** | **`diagnose_packing` 解门** —— 原本只在 `adaptation_protocol == "sc_only_v1"` 下才算，而 scratch warm-up 用 legacy protocol，也就是说验收标准里写的 bond 违例率 / chi recovery **根本不会被算出来**。现在所有 supervised SC 相位的验证都算。**这条会改变既有 legacy SC run 的日志内容**（只增不减），eval-only + `no_grad`。 |
| 410 | `checkpoint_identity` 改用 `aa_head_identity`（否则没有 head 时在**存档**才 AttributeError）。 |

### 6. `pxdesign_train/runner/trainer.py`（+25 −11）

- 9 处 `aa_backend == "fampnn"` → `packing_stack`。纯机械替换、语义不变，但覆盖
  `apply_phase`、DDP `find_unused_parameters`、optimizer 分组、loss 路由、
  checkpoint identity 的写与校验。漏一处就是静默的配置漂移，所以有测试禁止这个
  文件里再出现 `aa_backend`。
- 629–635：`total += weight_sc_chi * sc_chi`。

### 7. `pxdesign_train/checkpoints.py`（+48 −5）

- `SC_LAYOUT_KEYS_OPTIONAL`（bool）与 `SC_LAYOUT_KEYS_STR`（`packer_seq_cond`，
  字符串）：两组新键进 donor 架构记录，**按默认值比较**，所以既有 donor 仍能加载，
  而不同臂之间互相加载会被拒绝。
- `aa_head_identity()`：没有 head 时返回 `backend="absent"` + 理由，而不是不写
  这个字段。
- `compose_components` 接受 `sc_only`。

### 8. 其余

| 路径 | 改动 |
| --- | --- |
| `configs/configs_train.py`（+42） | `sidechain.torsion_packer` / `packer_seq_cond` / 14 个尺寸与行为键；`stage4.weight_sc_chi`。 |
| `cogenerate.py`（+12 −1） | 采样器的 `_fa` 对 `chi_output` / `torsion_packer` 恒真。**顺带修掉了 base 分支上本来就红着的 `test_train_inference_parity`**。 |
| `scripts/utilities/preflight_stage4_fampnn.py`（+6 −2） | `sc_only` 时不再要求 FaMPNN 权重。 |
| `scripts/training/slurm_stage4_fampnn_binder.sh`（+19 −2） | `AA_BACKEND=sc_only` 时不传 `--fampnn-checkpoint`，也**不把 fampnn 目录放进 PYTHONPATH**（`pip freeze` 会遍历 sys.path，不可读目录直接 PermissionError）。默认分支行为不变。 |
| `tests/test_train_inference_parity.py`（+34） | `TRAIN_ONLY` 里加 20 个 packer 构造参数，每条附理由。 |

## 三、建议的阅读顺序

1. `docs/sc_torsion_packer_apm_zh.md` 的「和 APM 完全一致 / 有意偏离」两节。
2. `pxdesign_train/sidechain/packer.py` 的 module docstring + `forward`。
3. `pxdesign_train/sidechain/ipa.py` 的 docstring（移植里唯一的实质改动 + 纳米尺度）。
4. `model.py:503` —— 唯一一处结构性接线。
5. `buildsc.py:127` 与 `torsion_loss.py` 的 `norm` 行 —— 两个 NaN 陷阱。
6. 其余是 gate 改名与配置管道，`git diff` 扫一眼就够。

## 四、如果只看三处

1. **`buildsc.py:127`** —— 修的是**既有代码**的潜伏 NaN 梯度，`chi_output` 路径
   同样受益。
2. **`train_protenix_monomer.py:2017`** —— 不 seed 就没有干净的 A/B。
3. **`stage4.py:257`** —— 这一条改变了既有 legacy SC run 的日志内容（只增不减），
   是这次唯一一处「顺手改了别人路径上的行为」。

## 五、状态（2026-09-15）

- 全量 `pytest tests`：**755 passed, 2 skipped**。
- 四臂 GPU smoke job **117026** 全部 COMPLETED；`bond_mae 7.2e-4 Å`、
  `bad_bond_fraction 0`；约 1.4 s/step。
- 正式四臂训练 job **117035** 已提交（array 0-3，四臂各一张 H200）。
