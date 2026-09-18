# 设计：APM torsion → 坐标 → h_res' → 主链（沿用旧 Stage II-B 的注入点）

> 2026-09-17。设计文档，未实现。
> 取代 `sc_to_bb_coordinate_feedback_zh.md` 里的 pair-表示方案：那一版绕开
> `xpb` 原子槽限制的办法是进 `z`，但注入点和旧设计不同，而且带一个
> `[L,L,10,10]` 的显存风险。本版回到旧设计的注入点。

## 0. 先说三件已经成立的事实

**① 旧注入点还在，可直接用。** `pxdesign_train/sidechain/coevolution.py`：

```
HResInjector:  h_res' --LayerNorm--> Linear(c_hres, c_trunk) --> 加到 s_trunk
               输出层零初始化 ⇒ 未训练时严格 no-op
```
然后 **复用同一个 B_theta** 重跑一次去噪（不新增 refinement head），
`s_trunk` 在 PXDesign 里本来是零，所以这是个干净的注入口。

**② `sc_feats` 已经由我们移植的 APM packer 产出。** `packer.py:554`：
```python
h_out = node_embed.to(h_res.dtype)                      # APM 的 node embedding
sc_feats = self.atom_embed(atom_name_ids) + h_out[:, :, None, :]
return x0_global, sc_feats, bb_feats
```
所以**"旧设计 + APM packer"是零新代码的**：
`sc_feats → pool_side_chain_atoms → HResInjector → s_trunk`。

**③ `residue_neighbours` 已经有了。** `sidechain/module.py:87`，按 CA 距离取
top-M，无效残基推到 +inf。所以邻域编码不需要全局 `[L,L]` 张量。

## 1. 一个必须先讲清楚的信息论问题

**逐残基地把"自己的侧链坐标"编码进 h_res，不带任何新信息。**

对残基 i，把它自己的侧链坐标放进它自己的 frame：键长键角在 builder 里是常数，
Rodrigues 旋转保长，所以局部坐标由 (残基类型, χ1..χ4) **完全决定**，是双射。

    coord_local(i)  ⟺  (restype(i), χ(i))

所以 `coord → h_res`（只看自己）≡ `torsion → h_res`，绕一圈没多出东西。

**坐标比扭转角多出来的信息只在跨残基时出现**：邻居的侧链原子落在我的 frame
里的什么位置。这才是"主链得知这里侧链要撞上了"的来源，也是这个回馈唯一能
提供、而 IPA 读 frame 读不到的东西。

所以本设计的核心不是"坐标"，而是**邻域侧链坐标**。

## 2. 模块

```python
class SidechainEnvEncoder(nn.Module):
    """邻域侧链坐标 -> h_res'（每残基一个向量，喂 HResInjector）。

    不重复 IPA 已有的信息：IPA 直接读 (R, t)，所以主链自身几何不编码。
    只编码"邻居的侧链原子在我的局部 frame 中的位置"——frame 读不到这个。
    """

    def __init__(self, c_hres=..., n_neighbors=16, c_hidden=128,
                 n_rbf=16, d_cut=12.0):
        self.rbf = RadialBasis(n_rbf, d_cut)      # 距离 -> 平滑基，无参数
        self.atom_mlp = nn.Sequential(            # 单个邻居原子 -> c_hidden
            nn.Linear(n_rbf + 3 + c_atom_id, c_hidden), nn.ReLU(),
            nn.Linear(c_hidden, c_hidden))
        self.out = nn.Sequential(
            nn.LayerNorm(c_hidden), nn.Linear(c_hidden, c_hres))
        nn.init.zeros_(self.out[-1].weight)       # 和 HResInjector 一样零初始化
        nn.init.zeros_(self.out[-1].bias)
```

### forward

```
输入:  x0_global  [B, L, A, 3]   packer 摆出的侧链坐标（全局）
       R, t       [B, L, 3, 3] [B, L, 3]   pass 1 的主链 frame
       atom_mask  [B, L, A]     哪些槽位真实存在
       ca         [B, L, 3]     取邻居用

1. nbr = residue_neighbours(ca, res_mask, M)          [B, L, M]   ← 现成
2. gather 邻居的侧链坐标和 mask                        [B, L, M, A, 3]
3. 变到残基 i 的局部 frame:
       x_loc = to_local(x_nbr, R_i, t_i)              ← 现成，保证刚体不变
4. 每个邻居原子的特征 = concat(RBF(‖x_loc‖), x_loc/‖x_loc‖, atom_id_emb)
       —— 距离用 RBF（平滑、不会让网络去拟合 1/r 的尖峰）
       —— 方向用单位向量（在局部 frame 里，所以是不变量）
5. atom_mlp -> masked mean over (M, A) -> [B, L, c_hidden]
6. out -> h_res'_sc                                    [B, L, c_hres]
```

`[B, L, M, A, 3]` 在 L=640, M=16, A=10 时是 3.1e6 个坐标 —— 比上一版的
`[L,L,10,10]`（4.1e7）小一个数量级，不需要分块。**M 是唯一的显存旋钮。**

### 与旧设计合并

```python
h_res_prime = h_res                       # 旧的（可选）
h_res_prime = h_res_prime + env_encoder(x0_global, R, t, ...)   # 本设计
s_trunk = s_trunk + HResInjector(h_res_prime)                   # 旧注入点
x_hat_0_post = B_theta(x_noisy, sigma, s_trunk, z_trunk)        # 复用 B_theta
```

## 3. 三个可对照的臂

都用同一个冻结 packer，只改回馈内容：

| 臂 | h_res' 来源 | 问的问题 |
|---|---|---|
| **F** feature | `pool(sc_feats)`（旧设计，零新代码） | packer 的内部表示够不够 |
| **C** coord | `env_encoder(x0_global)`（本设计） | 邻域坐标是否比内部表示更有用 |
| **F+C** | 两者相加 | 是否互补 |

**F 是必须跑的基线**，因为它零成本，而且如果 F 就够了，C 的复杂度不值得。

## 4. 梯度

packer 冻结，所以只剩一个选择：梯度是否经 `x0_global` 回流到 pass 1 的 frame。

| 臂 | 语义 |
|---|---|
| `detach` | 纯"用侧链修主链"。APM 的 `RefineModel` 就是这样（`.detach()`） |
| `flow` | 主链学会**产出让侧链摆得开的 frame** |

`flow` 是 APM 没做的对照，而且因为 packer 冻结，它不会把 packer 从 1.347 拉走
——比旧 Stage II-B 的回流安全。

## 5. 验收标准（先写，后跑）

| # | 判据 | 为什么先写 |
|---|---|---|
| 0 | **零初始化下 pass 2 与 pass 1 逐位相同**（`max\|Δ\| == 0`） | 否则后面任何差异都可能来自实现噪声 |
| 1 | **刚体不变性**：整体旋转平移输入，h_res'_sc 不变（到数值精度） | `to_local` 保证，但要测，不要假设 |
| 2 | 主链坏键率下降。基线：官方骨架 **8.04%**（400 步，同 probe） | — |
| 3 | BB RMSD / CA RMSD 不退化 | 回馈可能以精度换几何 |
| 4 | **designability**。基线：官方 8.26%，我们 0.0% | **这是真正的目标**，前面都是过程指标 |

## 6. packer 用哪个

P0 已出结果（449 条统一估计器）：

| packer | symmetry_rmsd |
|---|---|
| APM released | 1.3469 |
| 我们全数据 `plm` | 1.3608 |
| 我们全数据 `none` | 1.3823 |

**建议先用 APM released**：它是外部固定基准，回馈的效果不会和"我们 packer
这一版有多好"纠缠。差 0.014 Å，不影响结论。

## 7. 实现顺序

1. `SidechainEnvEncoder` + 判据 0（no-op）+ 判据 1（刚体不变）
2. 臂 F（零新代码，跑基线）
3. 两 pass 编排（复用 `precomputed_input`，pass 2 与 pass 1 同 σ 同 x_noisy）
4. 臂 C、F+C
5. detach vs flow
6. 判据 2/3，最后判据 4
