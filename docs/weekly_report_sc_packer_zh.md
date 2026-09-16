# 周报：Stage-2 侧链 Packer（APM one-step torsion packing）

**周期** 2026-09-15 ~ 09-16 为本周新工作；"前因"一节引用 09-11 ~ 09-14 的记录，
为动因而非本周产出，来源已逐条标注
**代码** `/hai/scratch/shenjm/wt_torsion_packer`，branch `sjm/sc-apm-torsion-packer`
**最后提交** `0938b10`

---

# 第一部分：汇报概要

## 一句话

上周定位到我们自己的 checkpoint designability 是 **0/3280**（官方 8.26%），
根因是主链几何塌缩、外加侧链读出层的一个机制性缺陷（Jensen 收缩，侧链键
平均短 **0.41 Å**、92.7% 偏短）。据此决定**主链换成官方 PXDesign 并冻结、
侧链模块推倒重做成扭转角形式**——APM 的 one-step torsion packer 原生就是这个
形式，所以选它做 V0。

本周把它完整搬过来并验证与官方实现**逐张量、逐输出等价**（546/546 张量，
chi 差 1e-5 rad），然后用 APM 自己的数据和目标函数重训了四个序列条件臂。

**结论：两条序列条件（结构 a_token、冻结 ESM-2）都没有带来可测的收益，
同时加还更差；真正的差距在数据量（0.26 Å，是被消融项的 14 倍）。**
过程中发现并修掉 4 个不报错的静默错误，其中 1 个推翻了上一轮已经写进文档的结论。

## 前因：为什么会推倒重做 sidechain module

这条线不是从"要一个 packer"开始的，是从一个 0% 开始的。以下引用自
2026-09-11 ~ 09-14 的记录（`docs/alphaproteo10_designability_result_zh.md`、
`docs/session_handoff_2026-09-11.md`、`docs/official_backbone_metrics_2026-09-12.md`、
`docs/sc_chi_output_zh.md`），**不属于本周新工作，是本周工作的动因**。

### ① AlphaProteo-10 designability 全量：官方 8.26%，我们 0.0%

| Backbone | 打分条数 | coverage | 严格通过 | designability |
|---|---|---|---|---|
| 官方 PXDesign v0.1.0 | 3280 | 1.0 | 271 | **8.26%** |
| Proteo-AA `111408/step6000` | 3280 | 1.0 | **0** | **0.0%** |

两组 coverage 都是 1.0，3280 条全部真实打分完成——**不是"任务没提交"的假零**
（上一轮确实出现过那种假零）。严格判据三项全部大幅失败，不是边缘擦过：
binder RMSD 中位 **16.00 Å**（阈值 <1.5）、pLDDT **0.51**（阈值 >0.9）。

### ② 往上游定位：生成的主链不是蛋白

| | 连续 CA–CA | 偏离 3.8 ± 0.3 Å 的比例 |
|---|---|---|
| PXDesign binder | median 3.83 | **0.0%** |
| Proteo-AA binder | median **2.75**（0.84–5.02） | **93.9%** |

相邻 CA 最近到 **0.84 Å**，物理上不可能。连带现象全部自洽：二级结构
`alpha 0.00 / beta 0.00 / loop 0.96`；ProteinMPNN 因为几何承不住侧链而退化成
多聚甘氨酸/丙氨酸（G+A 占比 p90 达 0.83，17% 的样本过半是 G/A）；于是 AF2
折不出来。`Rg = 12.95 Å` 说明链没炸开，是**塌缩成无二级结构的团**。

血缘定位（无条件单体 vs 复合体 binder 链，坏键=CA–CA 偏离 3.8±0.3）：

| checkpoint | 单体坏键 | 复合体坏键 |
|---|---|---|
| PXDesign official | 8.0% | 0.0% |
| Stage II 52500 | 60.1% | 98.1% |
| Stage III 111408 s6000 | 60.5% | 85.2% |

结论是**两个失效叠加**：基础主链几何不合格（60% vs 8%），复合体再叠一层
退化（同 checkpoint 60% → 85–98%）。

### ③ 一个必须标注的混淆：采样步数

09-12 重测发现，**官方 checkpoint 在 20 步下也是 88–91% 坏键，400 步才是 0%**：

| Source | 采样步数 | 坏键率 |
|---|---|---|
| Monomer | 20 | 88.12% |
| Monomer | **400** | **0.00%** |
| Binder 设计区 | 20 | 91.49% |
| Binder 设计区 | **400** | **0.00%** |

所以采样预算是一个真实的混淆项，默认已改成 400 步。

**但这不改变①的结论，而且我要标注一处口径不干净**：上表 ②的 60% 用的是旧
probe（不检查链断裂），而我们自己的 Stage II/III checkpoint **从未在 400 步 +
修正 probe 下重测过**。所以"我们 60% vs 官方 0%"不是一次同口径比较。
干净的事实只有两条：**我们的模型在 designability 上是 0/3280，且生成的
CA 间距中位 2.75 Å（塌缩）——而官方在 20 步下的失效方向是 CA 均值 7.66 Å
（张开），是不同的失效模式。**

### ④ 侧链模块自己另有一个独立问题：Jensen 收缩

这跟主链那个是**两个不同的问题**，不要混为一谈。

确定性 Cartesian 头 + 坐标 MSE，最优解是条件均值。对任意一根键，
由 Jensen 不等式 `‖E[a]−E[b]‖ = ‖E[a−b]‖ ≤ E[‖a−b‖] ≈ ideal`，
等号只在键方向几乎处处不变时成立。侧链跨 rotamer 多峰 → 方向变 →
**每一根方向会变的键都被系统性压短**。这不是调参问题，是目标函数最优解的性质。

在 warmup donor 上实测，2 个蛋白 369 根侧链内部键：

| term | 来源 | n | 平均带符号误差 | frac < 0 |
|---|---|---|---|---|
| **bond_sc** | model | 369 | **−0.4084 Å** | **0.927** |
| bond_sc | native（对照） | 369 | −0.0076 Å | 0.531 |
| bond_attach (CB–CA) | model | 117 | −0.0026 Å | 0.530 |
| bond_attach | native | 117 | +0.0008 Å | 0.521 |

关键在最后两行：`bond_attach`(CB–CA) 被 GT backbone frame 钉死、方向不变，
于是 **完全干净**。**收缩恰好只出现在多峰的自由度上，在被钉死的地方一点都没有——
同一个网络、同一次前向、同一个 loss。这是机制本身，不是相关性。**

### ⑤ 因此做的两个决定

**决定一：主链换成官方 PXDesign v0.1.0 并冻结。** 理由是它是手上**唯一**
单体/复合体几何都成立的骨架。可行性已验证：骨架部分与 Stage III
**形状完全一致**（`diffusion_module` 636 + `design_condition_embedder` 96 张量
零不符），可以拼成"官方骨架 + 我们的 packer"。代价是 packer 当初是在
Stage II/III 的骨架表示上训的，换骨架后是分布外——所以 packer 要重做。

**决定二：侧链读出层从 Cartesian 回归换成扭转角。** 直接针对④：键长在
builder 里是常数、Rodrigues 旋转保长，**网络在结构上没有能力破坏共价几何**。

**而 APM 的 one-step torsion packer 正好原生就是这个形式**——这是选它做 V0
的主要原因，不只是"论文里效果好"。

## 本周的目标

在①–⑤的基础上：用官方冻结骨架 + 重做的扭转角 packer，把 APM 的
one-step torsion packing 作为 Stage-2 V0（FlowPacker 作为后续 V1），
因为它一步出角、无采样循环、有官方开源实现和权重可以逐张量对照。

要回答的问题：**在主链和残基类型已知的前提下，额外给 packer 一路序列/结构
条件信息，能不能提升侧链重建精度？** 2×2 设计：

| 臂 | 输入 |
|---|---|
| `none` | 主链 frame + 残基类型（APM 的 released 设定） |
| `a_token` | + 冻结 PXDesign trunk 的结构感知 token |
| `plm` | + 冻结 ESM-2 650M（全 34 层，学习 softmax 加权） |
| `both` | 两者都加 |

四臂构造**同一套参数**（17,498,890），差异只在信息，不在容量。

## 做了什么

**1. 移植并证明等价。** `pxdesign_train/sidechain/packer.py` + `ipa.py`，
对照 `github.com/bytedance/apm @ a98e59b1`。

- released checkpoint 的 **546/546 张量**逐个同名同形状载入
- 同权重同输入下，与 APM 自己的 forward 比较：chi 的 `max|diff| = 1.0e-05 rad`
- 名字对得上不等于算的是同一个函数，所以专门做了 forward 回放
  （`check_packer_matches_apm.py` + APM 环境里的 `dump_apm_reference_forward.py`）

**2. 让"APM 的逻辑"真的是 APM 的代码，而不是我们的移植。**
发现 apm_reference 里 vendored 了完整 openfold，补三个导入期依赖后，
APM 的 featuriser 和损失在我们环境里可以**直接调用**：

| 部分 | 来源 |
|---|---|
| featurisation | **APM 本人** `_process_csv_row_FAESM` |
| L_chi | **openfold 本人** `supervised_chi_loss` |
| sidechain FAPE | APM glue 逐行誊写，张量操作仍是 openfold 的函数 |
| batch sampler | 单副本复刻，**逐 batch 与 APM 输出完全相同**（526/528/528） |
| 过滤 / 优化器 | APM 自己的过滤函数；AdamW(1e-4, β=0.95/0.999), clip 5.0 |

唯一"复刻"而非"调用"的是 sampler，因此专门做了逐 batch 比对并固化成测试。

**3. 四臂训练。** APM 的 18.3k 条 PDB monomer，200 epoch，各约 8–9 小时单卡 H200。

**4. 搭 a_token 特征桥并验证。** APM 的 pkl 不带 Protenix 特征，
所以走 `APM pkl → mmCIF → CifFileProvider → 冻结 trunk → a_token`，
零噪声下预计算成 cache。9 条链的一致性验证：长度与逐位点残基名 100%，
重复 forward 差 3e-5。

## 结论

### 成立的

**① 两条序列条件都没买到东西，同时加还更差。**（449 条，统一估计器）

| 臂 | symmetry_rmsd ↓ | vs none | chi1_acc ↑ | rotamer_rec ↑ |
|---|---|---|---|---|
| `none` | 1.6109 | — | 0.7963 | 0.6044 |
| `plm` | 1.6119 | **+0.0010** | 0.7940 | 0.6025 |
| `a_token` | **1.5929** | **−0.0180** | 0.7969 | 0.6027 |
| `both` | 1.6379 | **+0.0270** | 0.7874 | 0.5962 |

- `plm` 单独加 = **完全持平**（0.001 Å，远小于抖动）
- `a_token` 单独加 = **微弱正向**（0.018 Å，略大于训练末期抖动 0.006，
  但单 seed，最多算迹象）
- `both` = **最差的一臂**，比任何单通道都差。这是个非可加的负向交互。

`both` 这一条比其它两条更可信，因为**两个独立估计器都同意**：训练内曲线
从 epoch 60 起它就一直垫底（1.626–1.641 vs 其它 ~1.58），统一估计器也给
最差（1.638）。chi 指标上 `both` 同样全面最低。

**② 真正的瓶颈是数据量，不是条件信息。**
同架构、同损失、同 sampler，我们 1.611 vs APM 官方 1.347，**差 0.26 Å**，
是序列条件那一项（0.018）的 **14 倍**。最可能的原因：APM 的 `pdb_dataset`
开了 `use_AFDB: True` 和 `use_multimer: True`，我们只用了 18.3k 条 PDB
monomer（200 epoch × 3,536 cluster ≈ 707k 样本）。这是我当初主动记录的偏离，
现在看代价比预期大。

**③ 目标函数与度量的对齐是真实效应。**

| | 损失 | symmetry_rmsd | chi_rec_20° |
|---|---|---|---|
| 旧臂（PXDesign 数据） | 我们的 frame-aligned 坐标损失 | **1.35** | 0.571 |
| 新臂（APM 数据） | APM 的 chi + FAPE | 1.61 | **0.602** |

两类损失各自优化自己度量的那个量。**APM 的目标函数不是最小化侧链 RMSD 的那个。**

### 被推翻的

**上一轮"a_token 有害"的结论是错的。** 评测脚本 `predict_apm` 无条件把
`h_res` 喂成零，所以所有 a_token / both 臂在评分时**从来没拿到 a_token**。
受影响的数字：`ours_a_token` 1.531、`ours_both` 1.486（上一轮），
`apmdata_a_token` 1.686（修复前）。文档里那节"a_token 为什么没用：
最可能的解释"是在无效数字上做的推断，需要删除重写。

修复后正确结论是"**无差别**"，不是"有害"。

### 未定的

- **上一轮四臂的 a_token 效应无法测量。** 它们训练时的 a_token 带 σ≈0.4
  噪声和每步重抽的随机旋转，逐链不可复现。喂零得 1.531、喂干净 cache 得
  1.882，两个都不是它们见过的输入。只能重训。
- **`both` 为什么反而最差**，机制不明。两路条件各自无害（0.001 / −0.018），
  合起来 +0.027，是非可加的。可能是两路都投影到同一个 `c_node` 后相加、
  互相干扰；也可能只是单 seed 噪声。判定同样需要多 seed。

## 下一步（按优先级）

**P0 — 补数据。** 结论②说明这是唯一量级够大的方向。**不需要新数据、不需要
重新预处理**：四个源和完整 metadata 都已在盘上（2026-09-16 核实）。

| 源 | 磁盘 | 过 APM 自己的过滤后 | 现在是否使用 |
|---|---|---|---|
| `data_APM/pdb_monomer` | 18,684 | 18,373 | ✅ |
| `data_APM/pdb_multimer` | 11,620 | ~11,600 | ❌ |
| `data_APM/afdb` | 235,692 | **27,487**（`plddt_mean ≥ 95` 砍掉 84%） | ❌ |
| `swissprot_data` | 201,030 | **166,618** | ❌ |
| 合计 | ≈467,000 | **≈224,000** | 现用 **4.0%** |

metadata 在 `/hai/scratch/shenjm/apm_weights/metadata_all/`（`metadata_90.csv`
336 MB / `swissprot_metadata.csv` / `pdb_metadata.csv` 及各自 `.clusters`），
对磁盘文件 **100% 覆盖**。全开后训练集 **≈22.4 万条，是现在的 12 倍**。

注意真正的大头是 **SwissProt 的 16.7 万**，不是 AFDB —— AFDB 被 plddt 阈值
砍掉 84%，只加 2.7 万。

**两处需要先说清楚的（都是我的判断失误）：**

1. 我排除 AFDB 的理由是"AFDB 侧链是 AF2 的预测值，拿它训 packer 学到的是
   复现 AF2 而不是晶体学"。这个顾虑**站不住**：官方配置开着它并达到 1.347。
   我把自己的顾虑当成了理由，而没有先去验证它。
2. 我当时只读了 `data_APM/meta_data.csv`（58,338 行的合并子集，AFDB 只覆盖
   28k），据此以为 AFDB 的 metadata 不全。**完整 metadata 一直在我自己的目录里，
   我没去找。** SwissProt 这个源我在最初的数据清单里整个漏掉了，因为它在
   `extracted/` 顶层而不在 `data_APM/` 下。

**建议的最小实验**：只跑 `none` 一臂 + 全数据，足以回答"0.26 Å 是不是数据量
造成的"，比再跑四臂省 4 倍卡时。**这是最可能关闭 0.26 Å 差距的一步。**

**P1 — 如果要继续判定 a_token，需要多 seed。** 0.018 Å 在单 seed 下不可
区分。建议 3 seed × {none, a_token}，只有这样才能给出"有/无收益"的判断。
优先级低于 P0，因为即便 a_token 真有 0.018 Å，也只有数据差距的 1/14。

**P2 — 明确 V0 的度量目标。** 结论③说明选哪个损失取决于下游要什么：
如果 Stage-2 下游关心侧链坐标精度，我们自己的坐标损失更对；如果关心
rotamer/chi 正确性，APM 的目标更对。这个需要产品侧给一个答案。

**P3 — FlowPacker 作为 V1。** 原计划。建议在 P0 结束、V0 基线稳定后再启动。

---

# 第二部分：详细结果

## 实验配置

| 项 | 值 |
|---|---|
| 架构 | APM `SideChainModel`：6 × [IPA → LayerNorm → TransformerEncoder(4层4头) → Linear → Transition → EdgeTransition] + AngleResnet |
| 维度 | node 256 / edge 128；IPA c_hidden 16, heads 8, qk_points 8, v_points 12 |
| 参数 | 17,498,890 总；其中 16,896,244 与 APM checkpoint 同名同形状；`none` 臂实测吃梯度 17,093,096 |
| 损失 | `supervised_chi_loss(chi_weight=1, angle_norm_weight=0.02)` + `sidechain_fape`，**两项权重均 1.0** |
| 数据 | APM `pdb_monomer` 18,373 条（`a_token`/`both` 18,346，见下）；3,540 cluster |
| batch | 同长度分组，`min(64, 400000//L²+1)`，无 padding；单卡 `--accum 8` 对齐 APM 的 8 卡 |
| 优化 | AdamW(1e-4, β=0.95/0.999)，grad clip 5.0 norm，fp32 |
| 时长 | none 8h19m / plm 9h02m / a_token 7h57m，各 200 epoch |

## 完整结果（449 条 APM post-2021 held-out monomer，统一估计器）

完整四臂（job 117866，全部 200 epoch）：

| metric | apm_ckpt | apmdata_none | apmdata_plm | apmdata_a_token | apmdata_both | ours_none | ours_plm | dunbrack_template |
|---|---|---|---|---|---|---|---|---|
| symmetry_rmsd ↓ | **1.3477** | 1.6109 | 1.6119 | 1.5929 | 1.6379 | 1.3525 | **1.3439** | 2.0415 |
| chi_recovery_40° ↑ | **0.7870** | 0.7264 | 0.7256 | 0.7263 | 0.7192 | 0.7313 | 0.7348 | 0.6754 |
| chi_recovery_20° ↑ | **0.6779** | 0.6023 | 0.6022 | 0.5996 | 0.5931 | 0.5712 | 0.5759 | 0.5761 |
| chi1_accuracy_40° ↑ | **0.8552** | 0.7963 | 0.7940 | 0.7969 | 0.7874 | 0.8010 | 0.8040 | 0.7207 |
| chi1+chi2_acc_40° ↑ | **0.7263** | 0.6396 | 0.6346 | 0.6404 | 0.6269 | 0.6403 | 0.6475 | 0.5529 |
| rotamer_recovery ↑ | **0.6843** | 0.6044 | 0.6025 | 0.6027 | 0.5962 | 0.6116 | 0.6162 | 0.5489 |
| bond_mae | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| bad_bond_fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

四臂训练时长：none 8h19m / plm 9h02m / a_token 7h57m / both 8h57m，各 200 epoch。

> `ours_a_token` / `ours_both` **有意不列入**：无法忠实评分（见"未定的"）。
> 记录在案的两个值：喂零 1.531 / 1.486，喂干净 cache 1.882 / 1.794。

## 双估计器一致性核对

发现 a_token 评分 bug 的线索，也是修复正确性的证据。同一批 449 条链：

| arm | 训练内估计器 | 统一估计器 | 差 |
|---|---|---|---|
| none | 1.6039 | 1.6108 | 0.007 |
| plm | 1.6143 | 1.6117 | 0.002 |
| a_token（修复后） | 1.5910 | 1.5932 | 0.002 |
| a_token（修复前） | 1.5910 | 1.6863 | **0.095** |

两个估计器在 none/plm 上吻合到 0.007，却只在 a_token 上差 0.095 —— 这种
"只偏一臂"的形态排除了口径差异，指向那一臂的输入。plm 不受影响，因为 PLM
是在 packer 内部由 `type_idx` 算的，不经外部通道。

## 训练曲线（100 条验证链，训练内估计器，`symmetry_rmsd / chi1_acc`）

| epoch | none | plm | a_token | both |
|---|---|---|---|---|
| 9 | 1.837/.727 | 1.837/.734 | 1.873/.718 | 1.829/.728 |
| 49 | 1.655/.794 | 1.631/.800 | 1.605/.799 | 1.635/.793 |
| 99 | 1.601/.812 | 1.614/.809 | 1.584/.810 | 1.632/.801 |
| 149 | 1.582/.813 | 1.596/.813 | 1.575/.815 | — |
| 199 | 1.577/.817 | 1.592/.816 | 1.572/.817 | — |

epoch 150 之后四臂都在 ±0.006 内抖动，200 epoch 不是欠训。

## 等价性验证

**移植 vs APM 官方实现**（`check_packer_matches_apm.py`）

```
load_state_dict: 0 unexpected, 3 missing
  missing (ours only, expected): ['a_proj.bias', 'a_proj.weight', 'atom_embed.weight']
unnormalised sin/cos   max|diff| = 6.44e-06   OK
unit sin/cos           max|diff| = 8.46e-06   OK
chi (radians)          max|diff| = 1.00e-05   OK
EQUIVALENT within 1e-4: the port computes APM's function.
```

**sampler vs APM 官方**（`tests/test_apm_sampler_parity.py`）
真实过滤后 index 上，epoch 0/1/5 各 526/528/528 个 batch，**index 对 index 完全相同**。

## a_token 特征桥

**σ 的处理。** 两件都叫 noise 的事必须分开：

1. **坐标高斯扰动 = 严格 0**（`clean_coordinate_input=True`，`randn_like` 那行不执行）
2. **time channel 的条件 σ 钉在 4e-4**（Protenix 自己的 `s_min`），
   因为 `ln(σ/σ_data)/4` 在 σ=0 是 −∞，σ=0 对这个网络不是可表示的输入。
   它带来的 `c_in = 0.0625000000`，与 σ=0 处相同到打印精度。

构造后硬断言 + 每次 forward 从 `out["sigma"]` 读回核对。

**验证结果（9 条链）**

```
101m  L=154/154  restype 154/154  repeat=0.0e+00  rot-aug=0.0e+00  |A-B|=10.85 (151%)
102l  L=163/163  restype 163/163  repeat=2.7e-05  rot-aug=2.5e-05  |A-B|=11.79 (168%)
103l  L=159/159  restype 159/159  repeat=3.5e-05  rot-aug=2.8e-05  |A-B|=11.98 (166%)
104m  L=153/153  restype 153/153  repeat=3.2e-05  rot-aug=3.3e-05  |A-B|=11.94 (186%)
16vp  L=311/311  restype 311/311  repeat=4.9e-05  rot-aug=5.6e-05  |A-B|=12.45 (144%)  ← 45 残基 gap
1a06  L=279/279  restype 279/279  repeat=0.0e+00  rot-aug=0.0e+00  |A-B|=11.88 (165%)  ← 28 残基 gap
VERDICT: bridge aligned, sequence-carrying and deterministic
```

`|A-B|` = regime A（零噪声 + σ_floor）与 regime B（带噪声 + σ=0.4）之间的
a_token 差异，占 a_token 自身幅度（~7）的百分比。**σ=0.4 会把 a_token
改变 150–186%** —— 上一轮那一臂条件化的东西和干净结构下的 a_token
基本不是同一个特征。

## 发现并修掉的问题

**静默的（不报错，这是危险的那一类）**

| # | 问题 | 量级 | 后果 |
|---|---|---|---|
| 1 | `predict_apm` 把 `h_res` 恒喂零 | a_token 臂 +0.095 Å | **推翻了上一轮已写入文档的结论** |
| 2 | a_token 对随机刚体增广不不变（scipy 用 numpy RNG，`torch.manual_seed` 管不到） | ~10.4，与 σ=0.4 噪声的 ~11 同量级 | 上一轮 a_token 臂训练时有两个同量级污染源 |
| 3 | 顺序编号抹掉晶体学 gap | 32% 有 gap 的链 | trunk 的相对位置编码把 gap 合上了 |
| 4 | `aa_mask_mode` 误抄 CASP 的 `"all"` | 154/154 token 变 `[xpb]` | a_token 会完全不含序列信息 |

**会报错的**

| # | 问题 | 报错位置离原因的距离 |
|---|---|---|
| 5 | 插入码重号 → gemmi 合并成 16 原子残基 | 报在对称置换表，隔好几层 |
| 6 | 顺序编号 → Cα 相邻判定 → 7.25% 链被丢 | 报成"找不到合法 crop" |
| 7 | 编号连续但 Cα 相距 23–39 Å 的尾部残基 | 同上 |
| 8 | 补齐无上限 → 2081 token 超 crop | 报在 cropper |
| 9 | `PROTENIX_ROOT_DIR` 少一级，CCD 字典找不到 | 报成解析出空结果 |
| 10 | `Protenix/scripts/__init__.py` 遮蔽同名 namespace package | `ModuleNotFoundError` |
| 11 | 多加一层 batch 维 | 报在 atom-attention encoder 的断言 |

问题 3 和 7 的修法都基于一个事实：Protenix 丢链的判据是
`(Cα 距离 > 10Å) AND (label_seq_id 相邻)`（`filter.py:146`），
**只有编号声称相邻时才检查距离**。补齐上限定在 32，因为相对位置编码
把偏移钳在 ±32（`embedders.py:168`），补超过 32 对模型完全不可见。

## 有意的偏离（四臂一致，不影响臂间比较）

| 偏离 | APM | 我们 | 理由 |
|---|---|---|---|
| 并行 | 8 卡 DDP | 单卡 `--accum 8` | 有效 batch 对齐，梯度是累积而非 all-reduce |
| 数据 | PDB monomer + multimer + AFDB | **仅 PDB monomer** | AFDB 侧链是 AF2 预测值；见 P0 |
| a_token 噪声 | （上一轮）σ≈0.4 + 随机旋转 | 零噪声 + 固定取向 | 见 a_token 桥一节 |
| 丢链 | — | `a_token`/`both` 少 27 条（0.15%） | 生成的 mmCIF 未通过 Protenix 解析；两臂丢的是同一批 |

新 a_token 臂相比上一轮去掉了**两样**东西（噪声 + 随机取向），
**不是"只改噪声"的严格对照**。`--keep-random-rotation` 开关留着以便分离。

## 照抄的一个 APM bug

`apm/data/datasets.py:247`：

```python
torsions_1   = torsion_angles[:, -4:]   # chi1..chi4
bb_torsion_1 = torsions_1[:, :3]        # 于是 bb_torsions_1 = chi1..chi3
```

而 `cal_sidechain_fape_loss` 把它拼到 `torsion_angles_to_frames` 读作
omega/phi/psi 的槽位。影响范围有限（只驱动 rigid group 1-3，atom14 里只有
主链 O 属于 group 3），后果是主链 O 的 frame 错误、给 FAPE 贡献近似常数项。
**照抄是因为 released checkpoint 就是在这个目标函数下训出来的**，
在这里"修好"就意味着新 run 优化的东西和 reference 从来不是同一个。

## 代码与数据位置

| | 路径 |
|---|---|
| 我们的代码 | `/hai/scratch/shenjm/wt_torsion_packer`（`sjm/sc-apm-torsion-packer`） |
| APM 原始代码 | `/hai/scratch/shenjm/apm_reference`（`bytedance/apm @ a98e59b1`） |
| APM 权重 / 测试集 | `/hai/scratch/shenjm/apm_weights` |
| 训练数据 | `/hai/scratch/yfsun/apm/extracted/data_APM` |
| 四臂 checkpoint | `/hai/scratch/shenjm/proteo_aa_runs/packer_apm_data/{none,plm,a_token,both}` |
| a_token cache | `/hai/scratch/shenjm/proteo_aa_runs/a_token_cache/{train,val}` |
| 评测结果 | `/hai/scratch/shenjm/proteo_aa_runs/apm_testset_eval/` |
| 旁装依赖 | `/hai/scratch/shenjm/pyextra`（dm-tree, lightning-utilities, torch_scatter 垫片） |

## 复现命令

```bash
# 等价性
PYTHONPATH=$REPO:$REPO/PXDesign:$REPO/Protenix LAYERNORM_TYPE=torch \
  python scripts/evaluation/check_packer_matches_apm.py

# sampler 一致性
pytest tests/test_apm_sampler_parity.py

# a_token 桥验证
sbatch scripts/data/slurm_a_token_bridge_check_hai.sh

# 建 a_token cache（train + val）
sbatch scripts/data/slurm_build_a_token_cache_hai.sh

# 四臂训练
sbatch --array=0-1 scripts/training/slurm_packer_apm_data_hai.sh   # none, plm
sbatch --array=2-3 scripts/training/slurm_packer_apm_data_hai.sh   # a_token, both

# 全量评测
sbatch scripts/evaluation/slurm_eval_apm_all_methods_hai.sh

# 双估计器对照
sbatch scripts/evaluation/slurm_compare_estimators_hai.sh
```

## 相关作业号

| 作业 | 内容 |
|---|---|
| 117035 | 上一轮四臂（PXDesign 数据） |
| 117511 | 上一轮四臂评测（a_token 数字已知无效） |
| 117544 | 本轮 `none` / `plm` |
| 117588 | 本轮 `a_token` / `both` |
| 117578 | a_token cache 构建 |
| 117805 | 评测（a_token 喂零，已废弃） |
| 117807 | 双估计器对照 |
| 117814 | 评测（a_token 修复后，三臂） |
| 117866 | **评测（最终，四臂全量）** |
