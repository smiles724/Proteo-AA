# 设计：侧链 → 主链的坐标回馈（frozen APM packer）

> **已被取代（2026-09-17）**：注入点改回旧 Stage II-B 的 `h_res'` → `s_trunk`，
> 见 `sc_to_bb_hres_feedback_zh.md`。本文保留，因为第 2 节那个
> `xpb` 只有 5 个原子槽的约束、以及 PXDesign 经 `conditional_templ` 送几何的
> 机制，两者都仍然成立且仍然相关。

> 2026-09-17。设计文档，未实现。目标是把 APM released packer 摆出的侧链
> **坐标**送回主链模块，作为 Stage-2 的 SC→BB 回馈。

## 1. 和旧设计的区别

| | 旧 Stage II-B | 本设计 |
|---|---|---|
| 回馈的东西 | **特征** `h_res'`（packer 的 node embedding） | **坐标** 侧链原子的全局位置 |
| 注入点 | `HResInjector` → 加到 `s_trunk`（token 单体表示） | binned 成对距离 → 加到 `z`（pair 表示） |
| packer | 我们自己训的 | **APM released ckpt，冻结** |
| packer 是否学习 | 是（`trunk_grad_scale` 控回流） | 否（冻结，已验证 1.347） |

用冻结的官方 packer 是这个设计的主要优点：**回馈信号的质量是已知的**
（symmetry_rmsd 1.347，我们自己最好的全数据臂还在评测中），所以测出来的是
"回馈机制值多少"，而不是"我们的 packer 有多弱"。

## 2. 硬约束：设计区没有侧链原子槽

```python
>>> RES_ATOMS_DICT["xpb"]
['N', 'CA', 'C', 'O', 'OXT']      # 5 个
```

PXDesign 的设计 token 是 `xpb`，Protenix 原子轴上它只有 5 个槽。
**坐标无法写回原子数组。** 三条路及其代价：

| 方案 | 代价 | 采纳 |
|---|---|---|
| 扩 `xpb` 原子集到全原子 | 原子轴维度变化，**官方主链权重不能直接载入** | ✗ |
| 设计区不标 `xpb` | PXDesign 整套设计条件化建立在 `xpb` 上 | ✗ |
| 坐标 → 几何特征 → `z` | 新增 embedder；零初始化时是严格 no-op | **✓** |

第三条是 PXDesign 自己的机制：靶点条件化走 `conditional_templ`
（`_helpers.py:105`，64 bin / 2–22 Å → `nn.Embedding(65, 128)` → 加到 `z`，
`embedders.py:184`）。所以这条路已经被跑过，而不是新发明的。

## 3. 数据流

```
pass 1  ──────────────────────────────────────────────────────────────
  x_noisy(σ) ─► DiffusionModule ─► x_hat_0            [.., N_atom, 3]
                    │
                    └─ gather sc_bb_atom_idx[:, :3]  (N, CA, C)
                          │
                          ▼
                    build_frame ─► (R, t)             [.., L, 3,3] [.., L, 3]

侧链 ────────────────────────────────────────────────────────────────
  (R, t) + restype ─► APM TorsionPacker (frozen, released ckpt)
                          │
                          ├─ last_torsions["chi"]     [.., L, 4]
                          └─ x0_global                [.., L, MAX_SC, 3]  ← 坐标

回馈 ────────────────────────────────────────────────────────────────
  x0_global ─► SidechainGeometryEncoder
                  ├─ (a) pair: binned min-distance ─► z_sc   [.., L, L, c_z]
                  └─ (b) node: pooled local geometry ─► s_sc  [.., L, c_s]

pass 2  ──────────────────────────────────────────────────────────────
  DiffusionModule(x_noisy(σ), s_trunk + s_sc, z_trunk + z_sc) ─► x_hat_0'
```

关键点：**pass 2 的 `x_noisy` 和 σ 与 pass 1 完全相同**，只有条件表示变了。
这样两次 pass 的差异可以干净归因到回馈，而不是归因到不同的噪声实现。
`sample_diffusion_training(precomputed_input=(x_gt_aug, sigma, x_input))`
已经支持这个用法（Stage III 的 refinement 就是这么用的）。

## 4. 模块设计

```python
class SidechainGeometryEncoder(nn.Module):
    """侧链全局坐标 -> 加到 (s_trunk, z_trunk) 的两路条件。

    坐标不能写回 Protenix 的原子轴（xpb 只有 5 个槽），所以走 PXDesign 自己
    送几何的那条路：binned 成对距离进 pair 表示。
    """

    def __init__(self, c_s=384, c_z=128, no_bins=64,
                 min_bin=2.0, max_bin=22.0, node_geom_dim=...):
        # pair：+1 是 "没有距离" 那一档，和 ConditionTemplateEmbedder 对齐
        self.pair_embedder = nn.Embedding(no_bins + 1, c_z)
        # node：局部几何标量 -> c_s
        self.node_proj = nn.Sequential(
            nn.LayerNorm(node_geom_dim), nn.Linear(node_geom_dim, c_s))
        # 两路都零初始化：未训练时严格 no-op，pass 2 == pass 1
        nn.init.zeros_(self.pair_embedder.weight)
        nn.init.zeros_(self.node_proj[-1].weight)
        nn.init.zeros_(self.node_proj[-1].bias)
```

### (a) pair 路：残基间最小侧链距离

对每一对残基 (i, j)，取**两侧侧链重原子之间的最小距离**，binned 成 64 档。

为什么用最小距离而不是质心距离：packing 的下游意义是**空间冲突和接触**，
而这两件事由最近的一对原子决定，不由质心决定。质心距离会把一个伸出来的
Arg 和一个缩着的 Ala 算成同样的接触。

```python
d = cdist(sc_i, sc_j)                      # [.., L, L, MAX_SC, MAX_SC]
d = d.masked_fill(~pair_slot_mask, inf)
d_min = d.amin((-2, -1))                   # [.., L, L]
bins = bucketize(d_min, boundaries)        # 0..no_bins-1
z_sc = pair_embedder(mask * (1 + bins))    # 掩掉的对读第 0 档
```

`[L, L, 10, 10]` 在 L=640 时是 4.1e7 个距离 —— 需要分块，或者退化成
"每残基取 4 个代表原子"。**这是实现时要量的第一个数**。

### (b) node 路：每残基的局部侧链几何

不重复 IPA 已经读到的东西（frame），只给它 frame 读不到的：

| 标量 | 为什么 |
|---|---|
| 侧链伸展半径 `max‖x_sc − CB‖` | 侧链占多大空间 |
| 侧链质心在局部 frame 中的方向 | 往哪边伸 |
| 已摆出的侧链原子数 | 残基大小的代理 |
| χ1..χ4（sin, cos） | packer 的直接输出，8 维 |

χ 角这一路和 APM 的 `RefineModel` 相同（它把 torsions 同时喂 node 和 edge），
所以这部分等于复现了 APM 的做法；前三项是本设计额外加的。

## 5. 梯度：两臂对照

这是本设计里唯一的**研究问题**，不是工程问题。

| 臂 | packer | 回馈编码器 | 主链 |
|---|---|---|---|
| A `detach` | 冻结 | 训练 | 训练 | APM 的选择（`refine_model.py:132` 用 `.detach()`） |
| B `flow` | 冻结 | 训练 | 训练 | 梯度经坐标回流到 packer 输入侧 |

注意 packer 冻结时 A 和 B 的差别**只在于梯度是否穿过 `x0_global` 回到
frame**（即回到 pass 1 的主链输出）。这是一个比"回流到 packer 权重"更弱、
但更安全的回流：它让主链学会**产出让侧链摆得开的 frame**。

如果要做 APM 没做的那个对照（梯度回流到 packer 权重），需要解冻 packer，
建议放到后面，因为它会把 packer 从 1.347 拉走。

## 6. 验收标准（先写，后跑）

**零假设**：回馈编码器零初始化 ⇒ pass 2 输出与 pass 1 **逐位相同**。
这必须作为第一个测试，否则后面任何差异都可能来自实现噪声而不是回馈。

| 指标 | 判据 |
|---|---|
| no-op 测试 | 零初始化下 `max\|x_hat_0' − x_hat_0\| == 0` |
| 主链几何 | 坏键率（CA–CA 偏离 3.8±0.3）**下降**；当前基线：官方骨架 8.04% |
| 主链精度 | BB RMSD / CA RMSD 不退化 |
| designability | AlphaProteo-10，当前基线：官方 8.26%、我们 0.0% |

**最后一条是真正的目标。** 前三条是过程指标。

## 7. packer 用哪一个：现在可以两者都行

P0 已出结果（job 118011 + 评测 118931，449 条统一估计器）：

| packer | symmetry_rmsd |
|---|---|
| APM released ckpt | 1.3469 |
| 我们的全数据 `plm` | **1.3608**（+0.014） |
| 我们的全数据 `none` | 1.3823（+0.035） |

**差距已经小到不影响本设计的结论。** 所以：

- **建议先用 APM released ckpt** 做第一版。理由不是它更准，而是它是**外部
  固定基准**：回馈机制的效果不会和"我们 packer 这一版有多好"纠缠在一起，
  以后换 packer 也能对照。
- 我们自己的 `plm` 全数据 ckpt 作为第二个 packer 变体，用来检查结论是否
  依赖具体 packer。两者差 0.014 Å，如果回馈的效果在两者间差很多，那说明
  回馈对 packer 质量极度敏感，本身就是个发现。

`docs/weekly_report_sc_packer_zh.md` 的结论②记录了这次验证。

## 8. 实现顺序

1. `SidechainGeometryEncoder`（两路，零初始化）+ **no-op 测试**
2. 两 pass 的编排（复用 `precomputed_input`）
3. 量 `[L, L, 10, 10]` 的显存，决定要不要分块或代表原子
4. A/B 两臂（detach vs 梯度回流到 frame）
5. 坏键率 + BB RMSD
6. designability 闭环
