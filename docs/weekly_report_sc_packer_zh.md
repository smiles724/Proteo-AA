# 周报：从 designability 0% 到 Stage-2 侧链 Packer

**周期** 2026-09-09（上周三）~ 2026-09-16（本周三）
**代码** `/hai/scratch/shenjm/wt_torsion_packer`，branch `sjm/sc-apm-torsion-packer`
**相关文档** `alphaproteo10_designability_result_zh.md`、`session_handoff_2026-09-11.md`、
`official_backbone_metrics_2026-09-12.md`、`sc_chi_output_zh.md`、`sc_torsion_packer_apm_zh.md`

---

# 第一部分：汇报概要

## 一句话

本周从一个 **0%** 开始：我们自己的 checkpoint 在 AlphaProteo-10 上
designability 是 **0/3280**（官方 8.26%）。往上游查出**两个独立的失效**——
主链几何塌缩、以及侧链读出层的一个机制性缺陷（Jensen 收缩，侧链键平均短
**0.41 Å**、92.7% 偏短）。据此决定**主链换成官方 PXDesign 并冻结、侧链模块
推倒重做成扭转角形式**，而 APM 的 one-step torsion packer 原生就是这个形式。

后半周完成移植（与官方实现**逐张量、逐输出等价**：546/546 张量，chi 差 1e-5 rad），
并用 APM 自己的数据和目标函数训了四个序列条件臂。**结论：两条序列条件
（结构 a_token、冻结 ESM-2）都没有可测收益，同时加还更差；真正的差距在
数据量——0.26 Å，是被消融项的 14 倍。**

## 时间线

| 日期 | 阶段 | 产出 |
|---|---|---|
| 09-09 ~ 09-11 | **定位** | designability 全量 0/3280；血缘定位到"两个失效叠加" |
| 09-12 | **排除混淆** | 采样步数是真实混淆项（官方 20 步也坏 88%，400 步才 0%）；定 400 步为默认 |
| 09-13 ~ 09-14 | **机制** | 找到侧链读出层的 Jensen 收缩；决定改成扭转角输出 |
| 09-15 | **重做** | 移植 APM `SideChainModel`，证明等价；PXDesign 数据上跑首轮 2×2 |
| 09-16 | **对齐 + 消融** | 换成 APM 自己的数据/损失/sampler；a_token 特征桥；四臂完成并评测 |

---

## 阶段一（09-09 ~ 09-12）：为什么推倒重做

### ① designability 全量：官方 8.26%，我们 0.0%

| Backbone | 打分条数 | coverage | 严格通过 | designability |
|---|---|---|---|---|
| 官方 PXDesign v0.1.0 | 3280 | 1.0 | 271 | **8.26%** |
| Proteo-AA `111408/step6000` | 3280 | 1.0 | **0** | **0.0%** |

两组 coverage 都是 1.0，3280 条全部真实打分完成——**不是"任务没提交"的假零**
（上一轮出现过那种假零，这次专门排除了）。严格判据三项全部大幅失败，
不是边缘擦过：binder RMSD 中位 **16.00 Å**（阈值 <1.5）、pLDDT **0.51**（阈值 >0.9）。

### ② 根因在上游：生成的主链不是蛋白

| | 连续 CA–CA | 偏离 3.8 ± 0.3 Å 的比例 |
|---|---|---|
| PXDesign binder | median 3.83 | **0.0%** |
| Proteo-AA binder | median **2.75**（0.84–5.02） | **93.9%** |

相邻 CA 最近到 **0.84 Å**，物理上不可能。连带现象全部自洽：二级结构
`alpha 0.00 / beta 0.00 / loop 0.96`；ProteinMPNN 因为几何承不住侧链而退化成
多聚甘氨酸/丙氨酸（G+A 占比 p90 达 0.83，17% 的样本过半是 G/A）；于是 AF2
折不出来。`Rg = 12.95 Å` 说明链没炸开，是**塌缩成无二级结构的团**。

血缘定位（无条件单体 vs 复合体 binder 链）：

| checkpoint | 单体坏键 | 复合体坏键 |
|---|---|---|
| PXDesign official | 8.0% | 0.0% |
| Stage II 52500 | 60.1% | 98.1% |
| Stage III 111408 s6000 | 60.5% | 85.2% |

**两个失效叠加**：基础主链几何不合格（60% vs 8%），复合体再叠一层退化
（同 checkpoint 60% → 85–98%）。

### ③ 一个必须标注的混淆：采样步数

09-12 重测发现，**官方 checkpoint 在 20 步下也是 88–91% 坏键，400 步才 0%**：

| Source | 20 步 | 400 步 |
|---|---|---|
| Monomer | 88.12% | **0.00%** |
| Binder 设计区 | 91.49% | **0.00%** |

默认已改成 400 步。

**这里要更正我先前写错的一句话。** 我一度在本报告里写"我们的 checkpoint
从未在 400 步下测过，所以 60% vs 8% 不是同口径比较"。**查了原始 summary，
这是错的**：`monomer_geom_probe/` 下两份 `monomer_geometry_summary.json`
显示两者都是 `n_step: 400`、同一个 probe、同 12 条单体、同 crop：

| | official | Stage II 52500 |
|---|---|---|
| `n_step` | 400 | 400 |
| `bad_bond_fraction_mean` | **0.0804** | **0.6008** |
| `ca_ca_median` | 3.810 | 3.656 |
| **`ca_ca_min`** | 3.318 | **1.158** |

**所以 8.0% vs 60.1% 本来就是一次同口径比较，成立。** Stage II 在 400 步下
最近的一对 CA 仍然是 **1.158 Å**——物理上不可能，不是采样预算的问题。

真正的口径差异不在步数，而在**两个不同的 probe**：官方那个 0.00% 来自 09-12
的新 probe（排除链断裂、不同数据协议），旧 probe 对同一份官方权重给的是 8.0%。
09-12 的文档自己也写了这两个数不可直接比。新 probe 依赖 `stage4_fampnn`
（我读不到 FaMPNN 权重），所以**它至今只跑过官方骨架**。

**已补测并关闭这个 caveat**：Stage III 111408/step6000（真正拿去做
designability 的那个）在同 probe + 400 步下是 **60.5% 坏键、最近 CA 对
1.006 Å**，与 09-11 记录的 60.5% 完全重合。**三个 checkpoint 现在都在
400 步同 probe 下齐了**（见第二部分"400 步主链几何"）：

| checkpoint | 坏键率 | CA–CA 中位 | **CA–CA 最小** |
|---|---|---|---|
| PXDesign official | **8.0%** | 3.810 | 3.318 |
| Stage II 52500 | 60.1% | 3.656 | **1.158** |
| Stage III 111408 s6000 | 60.5% | 3.922 | **1.006** |

**结论：主链几何的失效是真的，不是采样预算的产物。** 400 步下我们两个
checkpoint 仍有 ~60% 坏键，最近的一对 CA 只有 1.0 Å。

---

## 阶段二（09-13 ~ 09-14）：侧链模块自己另有一个独立缺陷

这与②的主链问题是**两个不同的问题**，机制不同，不要混为一谈。

### ④ Jensen 收缩：为什么 Cartesian 回归头必然把键压短

**问题设定。** 原来的侧链读出层是：每个原子回归一个 3 维 Cartesian 偏移，
训练目标是坐标 MSE。对确定性网络 + MSE，已知最优解是**条件均值**：

    f*(x) = E[ y | x ]

也就是说，给定同一个主链环境 x，网络输出的是所有可能侧链构象的**平均位置**。

**为什么平均会压短键。** 考虑一根键的两个端点 a、b。网络输出的键长是

    ‖ E[a] − E[b] ‖ = ‖ E[a − b] ‖

而真实键长恒等于理想值：‖a − b‖ ≈ ℓ（共价键长几乎不变）。由 **Jensen 不等式**
（范数是凸函数）：

    ‖ E[a − b] ‖  ≤  E[ ‖a − b‖ ]  ≈  ℓ

**等号只在键向量 (a−b) 的方向几乎处处相同时成立。** 直观地说：把一组长度
相同但方向不同的向量求平均，得到的向量一定更短；方向越分散，越短。

侧链的构象跨多个 rotamer（χ 角有多个峰），所以键的**方向是随环境多峰分布的**
→ 平均后方向相消 → **每一根方向会变的键都被系统性压短**。

**这不是调参问题，是目标函数最优解的性质。** 加大权重、换优化器、训更久都
不会改变它——只要读出层是自由 Cartesian 且损失是坐标 MSE，最优解就在那里。

**实测（2 个蛋白，369 根侧链内部键）：**

| term | 来源 | n | 平均带符号误差 | frac < 0（偏短比例） |
|---|---|---|---|---|
| **bond_sc**（侧链内部键） | model | 369 | **−0.4084 Å** | **0.927** |
| bond_sc | native 对照 | 369 | −0.0076 Å | 0.531 |
| **bond_attach**（CB–CA） | model | 117 | **−0.0026 Å** | 0.530 |
| bond_attach | native 对照 | 117 | +0.0008 Å | 0.521 |

**关键在后两行，这是把"相关"变成"机制"的那个对照。** `bond_attach`(CB–CA)
的方向被 GT backbone frame **钉死**、不随构象变化，于是 Jensen 无从作用——
它**完全干净**（−0.0026 Å，偏短比例 0.53 = 随机）。

> **同一个网络、同一次前向、同一个 loss：收缩恰好只出现在多峰的自由度上，
> 在被钉死的地方一点都没有。** 这是机制本身，不是相关性。

### ⑤ 因此做的两个决定

**决定一：主链换成官方 PXDesign v0.1.0 并冻结。**
理由是它是手上**唯一**单体/复合体几何都成立的骨架。可行性已验证：骨架部分
与 Stage III **形状完全一致**（`diffusion_module` 636 + `design_condition_embedder`
96 张量零不符），可拼成"官方骨架 + 我们的 packer"。代价是 packer 当初在
Stage II/III 的骨架表示上训练，换骨架后是分布外——所以 packer 必须重做。

**决定二：侧链读出层从 Cartesian 回归换成扭转角。**
直接针对④：

    dchi  = atan2(sin, cos + 1)                  # 零初始化 ⇒ 恰好 0
    chi   = chi_from_local(template) + dchi
    local = build_sidechain_local(type, chi)      # 键长键角是常数
    x0    = to_global(local, frame_R, frame_t)

键长在 builder 里是常数，Rodrigues 旋转保长，所以**网络在结构上没有能力
破坏共价几何**。验证：同样的随机权重只翻这个开关，`False` 给 mean −0.9891 Å、
frac<0 = 1.000。

**而 APM 的 one-step torsion packer 原生就是这个形式**——这是选它做 V0 的
主要原因，不只是"论文里效果好"。

---

## 阶段三（09-15 ~ 09-16）：重做 packer 并做序列条件消融

### 移植并证明等价

`pxdesign_train/sidechain/packer.py` + `ipa.py`，对照
`github.com/bytedance/apm @ a98e59b1`：

- released checkpoint 的 **546/546 张量**逐个同名同形状载入
- 同权重同输入下与 APM 自己的 forward 比较：chi 的 **max|diff| = 1.0e-05 rad**

名字对得上不等于算的是同一个函数，所以专门做了 forward 回放，而不是只看
`load_state_dict` 不报错。

### 让"APM 的逻辑"真的是 APM 的代码

apm_reference 里 vendored 了完整 openfold，补三个导入期依赖后，APM 的
featuriser 和损失在我们环境里可以**直接调用**：

| 部分 | 来源 |
|---|---|
| featurisation | **APM 本人** `_process_csv_row_FAESM` |
| L_chi | **openfold 本人** `supervised_chi_loss` |
| sidechain FAPE | APM glue 逐行誊写，张量操作仍是 openfold 的函数 |
| batch sampler | 单副本复刻，**逐 batch 与 APM 输出完全相同**（526/528/528） |
| 过滤 / 优化器 | APM 自己的过滤函数；AdamW(1e-4, β=0.95/0.999), clip 5.0 |

唯一"复刻"而非"调用"的是 sampler，因此做了逐 batch 比对并固化成测试。

### 2×2 设计

| 臂 | 输入 |
|---|---|
| `none` | 主链 frame + 残基类型（APM 的 released 设定） |
| `a_token` | + 冻结 PXDesign trunk 的结构感知 token |
| `plm` | + 冻结 ESM-2 650M（全 34 层，学习 softmax 加权） |
| `both` | 两者都加 |

四臂构造**同一套参数**（17,498,890），差异只在信息、不在容量。
APM 的 18.3k 条 PDB monomer，各 200 epoch，单卡 H200 各 8–9 小时。

---

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
- `both` = **最差的一臂**，比任何单通道都差，非可加的负向交互

`both` 这一条比其它两条更可信，因为**两个独立估计器都同意**：训练内曲线从
epoch 60 起它就一直垫底（1.626–1.641 vs 其它 ~1.58），统一估计器也给最差。

**② 真正的瓶颈是数据量，不是条件信息。**
同架构、同损失、同 sampler，我们 1.611 vs APM 官方 **1.347**，**差 0.26 Å**，
是序列条件那一项（0.018）的 **14 倍**。

**③ 目标函数与度量的对齐是真实效应。**

| | 损失 | symmetry_rmsd | chi_rec_20° |
|---|---|---|---|
| 首轮（PXDesign 数据） | 我们的 frame-aligned 坐标损失 | **1.35** | 0.571 |
| 本轮（APM 数据） | APM 的 chi + FAPE | 1.61 | **0.602** |

两类损失各自优化自己度量的那个量。**APM 的目标函数不是最小化侧链 RMSD 的那个。**

### 被推翻的（两条，都是我自己的错误）

**(a) 首轮"a_token 有害"的结论是错的。** 评测脚本 `predict_apm` 无条件把
`h_res` 喂成零，所以所有 a_token / both 臂在评分时**从来没拿到 a_token**。
受影响：`ours_a_token` 1.531、`ours_both` 1.486、`apmdata_a_token` 1.686（修复前）。
修复后正确结论是"**无差别**"而不是"有害"。暴露它的线索是一个"只偏一臂"的
估计器不一致（none/plm 差 0.006，a_token 差 0.095）。

**(b) "训练噪声中位数是 0.3" 这个说法是错的，漏了 ×σ_data。**

采样器是 `sigma = sigma_data * exp(p_mean + p_std·z)`，σ_data = **16**
（`Protenix/protenix/model/generator.py:59`）。所以：

| σ | 真实分位 | 之前的说法 |
|---|---|---|
| 4e-4（本轮用的下界） | **0.00%** | — |
| 0.3 | **3.21%** | 被当成"中位数" |
| 0.4（首轮 a_token 用的） | **4.85%** | 曾说"落在 57.5 分位" |
| **4.82** | **50%（真实中位数）** | — |
| 16 | 78.81% | — |

**这翻转了解读。** 之前的说法是"σ=0.4 是分布正中，σ→0 才是分布外"；
实际是 **σ=0.4 本身就在最干净的 5% 尾部**，trunk 绝大多数训练样本的噪声
远大于它（中位 4.82 Å）。

对 a_token 的含义：**"用近乎干净的主链去查询这个 trunk"本身就是分布外
查询**，0.4 和 4e-4 只是同一条尾巴上的两个点。这比我之前撤回的那个
"a_token 与几何冗余"的解释更站得住，也给结论①提供了一个候选机制——
但它同样只是假设，没有实验支撑。

c_in 的部分之前是对的：`c_in = 1/√(σ²+256)` 在 σ ≤ 1 时几乎不变
（σ=0 时 0.06250000，σ=0.4 时 0.06248048，相对差 3e-4）。

### 未定的

- **`both` 为什么反而最差**，机制不明。两路条件各自无害（+0.001 / −0.018），
  合起来 +0.027，非可加。可能是两路都投影到同一个 `c_node` 后相加、互相干扰；
  也可能只是单 seed 噪声。
- **首轮四臂的 a_token 效应无法补测**：它们训练时的 a_token 带 σ≈0.4 噪声和
  每步重抽的随机旋转，逐链不可复现。喂零得 1.531、喂干净 cache 得 1.882，
  两个都不是它们见过的输入。只能重训。
- **阶段一②的 60% 坏键**从未在 400 步 + 修正 probe 下复测。

---

## 下一步（按优先级）

**P0 — 补数据。** 结论②说明这是唯一量级够大的方向。**不需要新数据、不需要
重新预处理**：四个源和完整 metadata 都已在盘上（09-16 核实）。

| 源 | 磁盘 | 过 APM 自己的过滤后 | 现在是否使用 |
|---|---|---|---|
| `data_APM/pdb_monomer` | 18,684 | 18,373 | ✅ |
| `data_APM/pdb_multimer` | 11,620 | ~11,600 | ❌ |
| `data_APM/afdb` | 235,692 | **27,487**（`plddt_mean ≥ 95` 砍掉 84%） | ❌ |
| `swissprot_data` | 201,030 | **166,618** | ❌ |
| 合计 | ≈467,000 | **≈224,000** | 现用 **4.0%** |

metadata 在 `/hai/scratch/shenjm/apm_weights/metadata_all/`，对磁盘文件
**100% 覆盖**。全开后训练集 **≈22.4 万条，是现在的 12 倍**。真正的大头是
**SwissProt 的 16.7 万**，不是 AFDB——AFDB 被 plddt 阈值砍掉 84%，只加 2.7 万。

**两处我的判断失误**：(1) 我排除 AFDB 的理由是"侧链是 AF2 预测值"，
但官方配置开着它并达到 1.347，这个顾虑站不住；(2) 我只读了
`data_APM/meta_data.csv`（58,338 行的合并子集），以为 AFDB metadata 不全，
**完整 metadata 一直在我自己目录里，我没去找**；SwissProt 这个源在最初的
数据清单里整个漏掉了，因为它在 `extracted/` 顶层而不在 `data_APM/` 下。

**建议的最小实验**：只跑 `none` 一臂 + 全数据，足以回答"0.26 Å 是不是数据量
造成的"，比再跑四臂省 4 倍卡时。

**P1 — 多 seed 才能判定 a_token。** 0.018 Å 在单 seed 下不可区分。
建议 3 seed × {none, a_token}。优先级低于 P0：即便 a_token 真有 0.018 Å，
也只有数据差距的 1/14。

**P2 — 明确 V0 的度量目标。** 结论③说明选哪个损失取决于下游要什么：
关心侧链坐标精度 → 我们自己的坐标损失；关心 rotamer/chi 正确性 → APM 的目标。
这个需要产品侧给一个答案。

**P3 — 回到 designability 闭环。** 阶段一的 0% 还没有被重新测过。
新 packer + 官方冻结骨架应该重跑一次 AlphaProteo-10，否则"修好了"没有证据。

**P4 — FlowPacker 作为 V1。** 原计划，建议 P0 结束、V0 基线稳定后启动。

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

## 400 步主链几何（补测，关闭一个 caveat）

同一个 probe（`probe_monomer_backbone_geometry.py`，repo `Proteo-AA-stage3-binder`
@ `d15de33`）、同 `n_step=400`、同 12 条 recentPDB 单体、同 crop 200、
同 sampler `pxdesign_native`：

| checkpoint | 坏键率 | CA–CA 中位 | CA–CA 最小 | Rg 中位 | job |
|---|---|---|---|---|---|
| PXDesign official v0.1.0 | **8.04%** | 3.810 | 3.318 | 12.94 | 114652 era |
| Stage II `step52500` | **60.08%** | 3.656 | **1.158** | 12.97 | 同期 |
| Stage III `111408/step6000` | **60.53%** | 3.922 | **1.006** | 17.00 | **117994（本次补测）** |

坏键 = `|CA_i − CA_(i+1)|` 偏离 3.8 ± 0.3 Å 的比例。

**这关闭了报告里一个我自己写错的 caveat。** 我一度以为"我们的 checkpoint
从未在 400 步下测过"，所以 60% vs 8% 不可比。查原始 summary 后发现 Stage II
早就是 400 步；本次补测 Stage III，结果与 09-11 记录的 60.5% 完全重合
（60.53%）。**所以那组数字一直是同口径的。**

值得注意的两点：

1. **Stage III 的 CA–CA 中位（3.922）比 Stage II（3.656）更接近理想值 3.8，
   但坏键率一样是 60%。** 说明失效不是一个系统性的偏移，而是**方差极大**——
   中位数正常掩盖不了 60% 的样本在容差外。
2. **最近的一对 CA：Stage II 1.158 Å、Stage III 1.006 Å。** 两个原子核相距
   1 Å 在物理上不可能（C–C 单键 1.54 Å）。400 步采样预算充足的情况下仍然如此，
   所以这不是采样不足，是模型本身。

**仍然存在的限制**：09-12 那个新 probe（排除链断裂、不同数据协议，对官方
骨架给 0.00%）依赖 `stage4_fampnn`，而 FaMPNN 权重和 yfsun 的 env 都不可读，
所以**它至今只跑过官方骨架**。本次已把它的"主链必须与官方逐张量相同"从
前置断言改为测量并上报（`--allow-different-backbone`），一旦 FaMPNN 可读，
三个 checkpoint 就能在新 probe 下一起测。

## 训练噪声分布（更正 "noise 一直是 0.3"）

采样器（`Protenix/protenix/model/generator.py:59`）：

```python
noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
#              p_std=1.5            p_mean=-1.2              sigma_data=16.0
```

即 `σ = 16 · exp(−1.2 + 1.5·z)`，z ~ N(0,1)。**之前说"中位数 σ=0.301"漏了
`× sigma_data`**：`exp(−1.2) = 0.301` 是去掉 16 倍之后的数。

2,000,000 次采样的实测分位（与解析值一致到 0.1%）：

| 分位 | 1% | 5% | 25% | **50%** | 75% | 95% | 99% |
|---|---|---|---|---|---|---|---|
| σ | 0.146 | 0.409 | 1.753 | **4.806** | 13.23 | 56.77 | 157.6 |

反过来查，某个 σ 落在什么分位：

| σ | 分位 | 说明 |
|---|---|---|
| 4e-4 | **0.00%** | 本轮 a_token 用的下界（Protenix `s_min`） |
| 0.01 | 0.00% | |
| 0.3 | **3.21%** | 之前被误当成中位数 |
| 0.4 | **4.85%** | 首轮 a_token 用的；之前误算成 57.5% |
| 1.0 | 14.72% | |
| **4.82** | **50.00%** | 真实中位数 = 16·e^−1.2 |
| 16.0 | 78.81% | = σ_data |

**解读的翻转。** 之前的说法是"σ=0.4 在分布正中，σ→0 才是分布外"。实际上
σ=0.4 本身就在**最干净的 5% 尾部**，trunk 绝大多数训练样本的噪声远大于它。
所以"用近乎干净的主链去查询这个 trunk"本身就是分布外查询；0.4 和 4e-4
只是同一条尾巴上的两个点，差别在于 4e-4 更极端（0.00% vs 4.85%）。

`c_in` 的部分之前是对的：

| σ | `c_in = 1/√(σ²+256)` | `c_noise = ln(σ/16)/4` |
|---|---|---|
| 0 | 0.06250000 | **−∞（不可表示）** |
| 4e-4 | 0.06250000 | −2.6492 |
| 0.3 | 0.06248902 | −0.9941 |
| 0.4 | 0.06248048 | −0.9222 |
| 4.82 | 0.05984352 | −0.3000 |

σ ≤ 1 时 `c_in` 相对变化 < 4e-4，起作用的确实只有时间嵌入 `c_noise`，
而它在 σ=0 处是 −∞——这是"σ 必须有个下界"的全部理由。

复现：

```python
from protenix.model.generator import TrainingNoiseSampler
import torch; torch.manual_seed(0)
x = TrainingNoiseSampler()(torch.Size([2_000_000]))
print(x.quantile(torch.tensor([.01,.05,.25,.5,.75,.95,.99])))
```

## 发现并修掉的问题

**静默的（不报错，这是危险的那一类）** —— 第 1 和第 5 条各推翻了一个已写进文档的结论

| # | 问题 | 量级 | 后果 |
|---|---|---|---|
| 1 | `predict_apm` 把 `h_res` 恒喂零 | a_token 臂 +0.095 Å | **推翻了上一轮已写入文档的结论** |
| 2 | a_token 对随机刚体增广不不变（scipy 用 numpy RNG，`torch.manual_seed` 管不到） | ~10.4，与 σ=0.4 噪声的 ~11 同量级 | 上一轮 a_token 臂训练时有两个同量级污染源 |
| 3 | 顺序编号抹掉晶体学 gap | 32% 有 gap 的链 | trunk 的相对位置编码把 gap 合上了 |
| 4 | `aa_mask_mode` 误抄 CASP 的 `"all"` | 154/154 token 变 `[xpb]` | a_token 会完全不含序列信息 |
| 5 | 噪声分布算错（漏 `×σ_data=16`），中位数 0.301 而非 4.82 | σ=0.4 的分位 57.5% → **4.85%** | 对"σ 该取多少"的整个推理方向反了 |

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
