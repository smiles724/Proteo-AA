# `sidechain.chi_output`：把共价几何变成常数

> 2026-09-14。默认 `False`，不影响任何现有 checkpoint。

## 要解决的问题

确定性 Cartesian 头 + 坐标 MSE，最优解是条件均值。而对任意一根键，

    ||E[a] - E[b]|| = ||E[a-b]|| <= E[||a-b||] ~ ideal        (Jensen)

等号只在键向量方向几乎处处不变时成立。侧链构象跨 rotamer 多峰 → 方向变 →
**每一根方向会变的键都被压短**。这不是调参问题，是目标函数最优解的性质。

在 warmup donor（`official_sc_rigid_warmup/114967/checkpoints/step46000.pt`，EMA）
上实测，2 个蛋白 369 个侧链内部键：

| term | 来源 | n | 平均带符号误差 | rms | frac < 0 |
| --- | --- | ---: | ---: | ---: | ---: |
| **bond_sc** | model | 369 | **−0.4084 Å** | 0.5028 | **0.927** |
| bond_sc | native（对照） | 369 | −0.0076 Å | 0.0333 | 0.531 |
| bond_attach | model | 117 | −0.0026 Å | 0.0374 | 0.530 |
| bond_attach | native | 117 | +0.0008 Å | 0.0109 | 0.521 |

注意 `bond_attach`（CB–CA）是干净的：它被 GT backbone frame 钉死，方向不会变，
Jensen 无从作用。**收缩恰好只出现在多峰的自由度上，在被钉死的地方完全不出现** ——
同一个网络、同一次前向、同一个 loss。这是机制本身，不是相关性。

## 做法

读出层从「每原子回归 10×3 自由 Cartesian 偏移」换成「每残基预测 4 个扭转角」，
再由 `buildsc.build_sidechain_local` 用理想几何摆原子：

    pooled = masked_mean(atom_feats)            # [B, L, c_atom]
    dchi   = atan2(sin, cos + 1)                # 零初始化 => 恰好 0
    chi    = chi_from_local(template) + dchi    # NaN 表示该残基没有这个 chi
    local  = build_sidechain_local(type, chi)   # 键长键角是常数
    x0     = to_global(local, frame_R, frame_t)

键长在 builder 里是常数，rodrigues 旋转保长，所以**网络没有能力破坏共价几何**。

## 验证

`tests/test_sc_chi_output.py`：

* 同样的随机权重，只翻这个开关 —— `False`：mean −0.9891 Å、frac<0 = 1.000；
  `True`：mean +0.0000 Å、**max |err| 8.3e-7 Å**。
* builder 本身：20 种残基 × 8 组随机 χ × 全部键 = 584 组，最大偏差 1.2e-6 Å。
* 零初始化输出 = Dunbrack 模板（偏差 4.8e-7 Å）。训练学的是对模板的修正。
* 梯度有限，`chi_out` 正常收到梯度；缺 frame 时 raise；与 `template_residual` 互斥。

## 下游承接

**不变**：`atom_feats` [B,L,10,c_atom] 和 `bb_feats` [B,L,4,c_atom] 在坐标读出
**之前**产生，所以 `sidechain_feedback` → `h_res_prime`（老 Stage III 的 B_post）、
`a_token_fusion`（a_direct）、`q_atom_fusion`（q_direct）三条回路收到的张量逐项同构。
`sc_pred_global` 仍是 [B,L,10,3]，所有 loss、`clash_loss`、`state.sc_xyz`、几何诊断
和 mmCIF 导出都不用改。**loss 也不用改** —— 现有的 symmetry-aware 坐标 loss 直接
作用在构建出的坐标上；χ 的周期性顺带覆盖了对称残基（PHE/TYR 的 χ2 是 180° 周期）。

**变**：`chi_output` 进了 `SC_LAYOUT_KEYS` 和 `SIDECHAIN_LAYOUT_KEYS`。同一批权重在
两种模式下含义不同，donor 混用必须被拒绝。

## 边界

1. 需要 frame-aware（要 `frame_R`/`frame_t` 把局部构象映回全局），否则 raise。
2. `sc_adapt` 下 `force_gt_type_logits=False`，`type_idx` 变成预测身份。这是正确
   行为，且**没有引入新的离散依赖** —— 原子清单本来就是从同一个 argmax 实例化的。
3. **χ 空间里仍然会平均。** 两个 rotamer 的 χ 平均得到的是另一个**化学合法**的构象，
   而不是不可能的结构。所以 RMSD 可能不降甚至略升 —— 这是预期代价，不要读成失败。
   要连 χ 的多峰一起解决，需要 rotamer mode 分类（离散承诺）或在 χ 上做扩散。

## 验收建议

用**绝对**标准，不要用「相对 control 降百分之多少」：`bond_sc` / `angle_sc` 违例率
应当接近 0，然后再看 RMSD 和 sc-lDDT 的代价。参照点是 Dunbrack 模板基线 —— 它不用
模型、违例率 ~0。打不过它就不该往下游走。
