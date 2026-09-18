# 设计：APM packer → 邻域几何 → h_res' → 主链（沿用 Stage III 的注入点）

> 2026-09-17。设计文档，未实现。
> 取代 `sc_to_bb_coordinate_feedback_zh.md` 的 pair-表示方案（那版改了注入点，
> 且带 `[L,L,10,10]` 显存风险）。本版回到 Stage III 实际训练过的注入点。

## 0. 已经存在的东西（逐条核对过，含出处和形状）

| 组件 | 位置 | 出处 | Stage III 训过？ |
|---|---|---|---|
| `HResInjector` | `sidechain/coevolution.py:25` | 我们自己（Stage II-B 设计） | ✅ 权重在 `111408/step6000` |
| `pool_side_chain_atoms` | `sidechain/coevolution.py:43` | 我们自己 | ✅（无参数） |
| `_CrossAtomBlock` + `residue_neighbours` | `sidechain/module.py:44` / `:88` | **我们自己**，commit `ddfdbf6`（2026-08-07） | ✅ 176 张量 |
| `sc_feats`（每原子特征） | `sidechain/packer.py:554` | APM packer 的 `node_embed` + `atom_embed` | ✗（新） |
| `x0_global`（侧链坐标） | `sidechain/packer.py:561` | APM packer | ✗（新） |

**两处更正我先前说错的：**

1. `residue_neighbours` **不是 APM 的**，是我们 2026-08-07 自己写的。APM 的
   `SideChainModel` 走 IPA + 全长 TransformerEncoder，**不做 KNN 稀疏化**；
   APM 代码里唯一叫 neighbor 的是 `all_atom.py:206` 的 `calculate_neighbor_angles`，
   那是算键角的，无关。
2. 我一度按 checkpoint 里"没有 `cross_atom` 字样"推断 Stage III 用的是残基级
   block，**那是错的**——`module.py:224` 里 `self.cross_res_blocks = ModuleList([
   _CrossAtomBlock(...)])`，属性名沿用了旧前缀，装的是**原子级**的类。
   所以 `residue_neighbours` 正是 Stage III 用的那个（`module.py:439` 调用）。

**关键宽度不匹配（决定了不能热启动）：**

| | 旧 `sidechain_module` | APM packer |
|---|---|---|
| 每原子特征宽度 | **768** | **256**（`packer.py:333`，= `c_node`） |

Stage III 的 `hres_injector` 是 `LayerNorm(768) → Linear(768→384)`，
`q_atom_fusion.sc_proj` 是 `Linear(768→128)`。接 APM packer 就得是 `c_hres=256`。
所以 **`HResInjector` 必须新实例、零初始化重训，不能载 Stage III 的权重** ——
这不只是"分布外"的顾虑，是形状不兼容。

## 1. 为什么编码邻域而不是自己

**逐残基把"自己的侧链坐标"编码进 h_res 不带任何新信息。** 键长键角在 builder
里是常数、Rodrigues 旋转保长，所以

    coord_local(i)  ⟺  (restype(i), χ(i))

是双射。`coord → h_res`（只看自己）≡ `torsion → h_res`。

坐标比扭转角多出来的信息**只在跨残基时出现**：邻居的侧链原子落在我的 frame
里的什么位置。这也是这条回馈唯一能提供、而 IPA 读 `(R,t)` 读不到的东西。

`_CrossAtomBlock` 的 docstring 早就写了同一件事（"a side-chain atom cannot
tell WHICH neighbouring atom sits in its way, or in which direction, only that
some residue is nearby. Mean-pooling is direction-blind."）——所以这个设计不是
新发明，是把那个模块接到新的消费者上。

## 2. 具体流程

```
── pass 1 ────────────────────────────────────────────────────────────────
x_noisy(σ), s_trunk=0, z_trunk ─► B_theta ─► x_hat_0        [.., N_atom, 3]
                                              │
                        gather sc_bb_atom_idx[:, :3]  (N, CA, C)
                                              ▼
                                   build_frame ─► (R, t)    [.., L, 3,3] [.., L, 3]
                                                  ca = x_hat_0[sc_bb_atom_idx[:,1]]

── 侧链（冻结） ──────────────────────────────────────────────────────────
TorsionPacker(frozen, APM released ckpt)
    输入: (R, t), restype logits, sc_atom_name_ids, sc_chemical_mask
    输出: x0_global  [B, L, 10, 3]    侧链坐标（全局）
          sc_feats   [B, L, 10, 256]  每原子特征

── 回馈（新增的唯一一处组合） ─────────────────────────────────────────────
nbr_idx = residue_neighbours(ca, res_mask, M=16)            现成，[B, L, M]

for blk in env_blocks:                                      _CrossAtomBlock(256, 8, 16)
    sc_feats = blk(sc_feats, x0_global, chem_mask, ca, res_mask, nbr_idx)
                    ▲            ▲
                    │            └─ 坐标从这里进：块内按原子间距离做几何注意力
                    └─ 特征

h_res_sc = pool_side_chain_atoms(sc_feats, chem_mask)       现成，[B, L, 256]
s_trunk  = s_trunk + HResInjector(c_hres=256, c_trunk=384)(h_res_sc)   现成类，新实例

── pass 2 ────────────────────────────────────────────────────────────────
B_theta(x_noisy(σ), s_trunk, z_trunk) ─► x_hat_0_post
        ▲                ▲
        └── 与 pass 1 完全相同的 x_noisy 和 σ ──┘
```

**pass 2 只改条件表示，不改噪声实现**。这样两次 pass 的差异能干净归因到回馈。
`sample_diffusion_training(precomputed_input=(x_gt_aug, sigma, x_input))`
已支持（Stage III 的 refinement 就这么用）。

## 3. 需要新增的代码

只有一个薄封装 —— 其余全是现成类的新实例。

```python
class SidechainEnvFeedback(nn.Module):
    """APM packer 的输出 -> 加到 s_trunk 的 h_res' 增量。

    三个组成部分都已存在且在 Stage III 训练过，本类只负责把它们按 APM packer
    的宽度（c_atom=256，不是旧模块的 768）串起来：

        _CrossAtomBlock  × n_blocks   邻域原子级几何注意力（坐标从这里进）
        pool_side_chain_atoms         每残基 masked mean（无参数）
        HResInjector                  -> s_trunk，输出层零初始化

    零初始化的后果：未训练时 pass 2 与 pass 1 逐位相同。这是判据 0。
    """

    def __init__(self, c_atom=256, c_trunk=384, n_blocks=2,
                 n_heads=8, n_neighbors=16, use_env=True):
        super().__init__()
        self.use_env = use_env
        self.env_blocks = nn.ModuleList([
            _CrossAtomBlock(c_atom, n_heads, n_neighbors) for _ in range(n_blocks)
        ]) if use_env else nn.ModuleList()
        self.injector = HResInjector(c_hres=c_atom, c_trunk=c_trunk)

    def forward(self, sc_feats, sc_coords, chem_mask, ca, res_mask):
        nbr = None
        for blk in self.env_blocks:
            if nbr is None:
                nbr = blk.residue_neighbours(ca, res_mask, blk.n_neighbors)
            sc_feats = blk(sc_feats, sc_coords, chem_mask, ca, res_mask, nbr_idx=nbr)
        h_res_sc = pool_side_chain_atoms(sc_feats, chem_mask)
        return self.injector(h_res_sc)          # [B, L, c_trunk]，加到 s_trunk
```

`use_env=False` 就退化成臂 F（下一节），所以一份代码覆盖两臂。

## 4. 三臂对照（同一个冻结 packer）

| 臂 | `use_env` | h_res' 来源 | 问的问题 |
|---|---|---|---|
| **F** feature | `False` | `pool(sc_feats)` 直接注入 | packer 的内部表示够不够 |
| **C** coord | `True` | 先过邻域几何块再 pool | **邻域坐标是否比内部表示多买到东西** |
| **C-only** | `True`，`sc_feats` 换成 `atom_embed(ids)` | 只有原子身份 + 坐标，不含 packer 的 node_embed | 收益是来自坐标还是来自 packer 的表示 |

**F 必须先跑**：它只多一个 `HResInjector`（~0.3M 参数），如果 F 就够了，
C 的复杂度不值得。**C-only 是关键对照**——没有它就无法区分"坐标有用"和
"APM 的 node_embed 有用"。

## 5. 梯度：两个选择

packer 冻结，所以 packer 权重不会被拉走。剩下的选择是梯度是否经 `x0_global`
和 `(R, t)` 回流到 pass 1 的主链输出。

| 臂 | 语义 | 参照 |
|---|---|---|
| `detach` | 纯"用侧链修主链" | APM `RefineModel` 就是这样（`refine_model.py:132` `.detach()`） |
| `flow` | 主链学会**产出让侧链摆得开的 frame** | Stage II-B 的做法（`trunk_grad_scale` 控强度） |

`flow` 是 APM 没做的对照，且因 packer 冻结而比旧 Stage II-B 安全。

## 6. 验收标准（先写，后跑）

| # | 判据 | 为什么必须先写 |
|---|---|---|
| **0** | 零初始化下 `max\|x_hat_0_post − x_hat_0\| == 0`（逐位） | 否则后面任何差异都可能来自实现噪声 |
| **1** | 刚体不变性：整体旋转平移 ⇒ `h_res_sc` 不变（到数值精度） | `_CrossAtomBlock` 用的是原子间距离，理论上不变，但要测 |
| 2 | 邻域块不改变 packer 的侧链预测（它只改特征，不改 `x0_global`） | 防止把回馈实现成偷偷改侧链 |
| 3 | 主链坏键率下降。基线：官方骨架 **8.04%**（400 步同 probe） | — |
| 4 | BB RMSD / CA RMSD 不退化 | 回馈可能以精度换几何 |
| **5** | **designability**。基线：官方 8.26%，我们 0.0% | **真正的目标**，前面都是过程指标 |

## 7. packer 用哪个

P0 结果（449 条统一估计器）：APM released **1.3469** / 我们全数据 `plm`
1.3608 / `none` 1.3823。

**建议先用 APM released**：它是外部固定基准，回馈的效果不会和"我们 packer
这一版有多好"纠缠在一起。差 0.014 Å，不影响结论。

## 8. 实现顺序

1. `SidechainEnvFeedback` + **判据 0、1、2**（纯单测，不占卡）
2. 两 pass 编排（复用 `precomputed_input`，pass 2 同 σ 同 `x_noisy`）
3. 臂 F（最少改动，拿基线）
4. 臂 C、C-only
5. `detach` vs `flow`
6. 判据 3/4，最后判据 5
