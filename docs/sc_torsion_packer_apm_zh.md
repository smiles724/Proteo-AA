# Stage 2 V0：APM 的 Sidechain Module，以及 a-token / PLM 的 2×2

> 2026-09-15。分支 `sjm/sc-apm-torsion-packer`（从 `sjm/sc-chi-output` 分出），
> worktree `/hai/scratch/shenjm/wt_torsion_packer`。默认 `torsion_packer=False`，
> 不影响任何现有 checkpoint、任何现有 run。
> 参考实现：`https://github.com/bytedance/apm` @ `a98e59b1`（2025-05-29），
> 本地 clone 在 `/hai/scratch/shenjm/apm_reference`。
> 改动清单见 [`sc_torsion_packer_changes_zh.md`](sc_torsion_packer_changes_zh.md)。

## 要回答的问题

不是「torus diffusion 能不能 work」——那个已经被 DiffPack / SidechainDiff /
FlowPacker 反复证明过。要回答的是：

$$
\boxed{\;BB + res\_type + (\,a\text{-}token\ ?\ \mathrm{PLM}\ ?\,)\;\longrightarrow\;\chi_{1:4}}
$$

**Stage 1 传下来的 a-token 到底给 full-atom decoding 提供了多少信息，以及其中
有多少是一个现成的蛋白语言模型已经有的。** 这是 Proteo-AA 的核心论点，和扩散
无关。一步 packer 没有 noise schedule、没有 score/velocity target、没有 sampler，
所以结果不好时可查的地方少一半。

## 架构：APM `SideChainModel` 的移植

`pxdesign_train/sidechain/packer.py` 的 `TorsionPacker`，尺寸照抄
`apm/configs/model.yaml` 的 `packing_model` 块：

```
node 256 / edge 128
trunk = 6 × [ InvariantPointAttention(c_hidden 16, 8 heads, 8 qk_points, 12 v_points)
              → LayerNorm
              → nn.TransformerEncoder(4 层, 4 头, dropout 0.2)
              → post Linear(init="final")
              → StructureModuleTransition
              → EdgeTransition（最后一块除外） ]
head  = AngleResnet（AF2 Alg. 20 行 11–14，4 个 block）
pair  = dense L×L：cross-concat(node) + relpos + CA 距离直方图(22 bin)
        + AngularEncoding(torsions) ×2 + diffuse_mask
```

参数量 **17.35 M**（APM 报 ~22M，同一量级；差在我们没有 PLM attention map 分支
等）。IPA 与结构模块的 block 在 `pxdesign_train/sidechain/ipa.py`，是
`apm/models/ipa_pytorch.py` 的移植（Apache-2.0，带版权头）。

**移植里唯一的实质改动**：APM 用 openfold 的 `Rigid` 对象，调
`r[..., None].apply(pts)` / `invert_apply`；这里直接收 `(R, t)` 张量——也就是
`sidechain/frames.py` 本来就在用的表示——把那两个操作手写出来。为两个方法调用
vendor 整个 openfold geometry 包不值得。`tests/test_sc_ipa.py` 钉住了风险面：
全局旋转+平移后 IPA 输出**逐位不变**（差 0.0）。

另一个容易漏的点：APM 在进 trunk 前把平移**换算成纳米**（`rigids_ang_to_nm`）。
这不是装饰——IPA 的点注意力里有一个写死的 `9/2` 方差常数，假定的就是纳米尺度。
`packer.py` 里做了同样的换算。

## 和 APM 完全一致的部分

- **$\mathcal L_\chi$**：APM 调 openfold 的 `supervised_chi_loss`
  （`angle_norm_weight=0.02`）。我们的 `sidechain/torsion_loss.py` 与它**逐项
  等价**，已对着他们 vendored 的 openfold 源码核过。
- **`chi_pi_periodic` 表**：ASP χ2 / GLU χ3 / PHE χ2 / TYR χ2，与 openfold
  的表**逐项相同**（核对过）。
- **sin/cos 通道顺序是 (sin, cos)**，和 APM 的 `gt_sin_cos` 与 openfold 一致。
  数学上任意，实践上要命：搞反了每个角都关于 π/4 反射。
- **随机扭转输入**：APM 的 packing 模式里 `torsions_t` 是 $U[0,2\pi)$ 的**纯随机**
  角，`torsions_sc`（自条件）在训练时恒为零（`interpolant.py:589`），只有采样
  循环才填上一步的预测。两个通道在这里都不携带信息，而随机那个会让 packer 的
  输出变成**随机的**。照搬了，开关是 `packer_random_torsion_input`。
- **`rotvecs` 节点特征**：frame 的全局旋转向量。于是节点通道**不是 SE(3) 不变
  的**，尽管 IPA 是。照搬了（`packer_embed_rotvecs`），代价见下。
- ESM-2 650M（`apm/configs/base.yaml: PLM: faESM2-650M`）、bf16、冻结、
  取 34 层（含 embedding 层）做可学 softmax 加权。

## 有意偏离的部分

1. **`seq_cond` 可插拔**（默认 `a_token`）。APM 的 packer 没有 a_token，序列信息
   只走 PLM。我们的实验就是要量这件事，所以做成 `none / a_token / plm / both`，
   插入点是 APM 的那一个点：投影到 `c_node` 后**加到** node embedding 上，就是
   `init_node_embed += plm_s` 的位置。
2. **四臂共用一套参数**。两个投影（`a_proj` 与 `plm_conditioner`）无论哪一臂都
   构造，关掉的那个乘 0。于是四臂的参数名、形状、初值**逐位相同**，只有信息不同。
   APM 是条件构造 PLM 分支的；`seq_cond="plm"` 时本模块算的东西和 APM 一样，多出
   来的投影只是闲置。
3. **`embed_aatype=True`**（APM 的 packing 配置是 `False`）。APM 靠 PLM 带序列
   信息；我们的 `none` / `a_token` 两臂没有 PLM，关掉它就等于没有任何残基类型
   通道。res_type 是 Stage 2 的显式输入，所以留着。
4. **坐标由 `BuildSC` 摆**（本仓库的理想几何 builder），不是 openfold 的 rigid
   group。同一个映射，不同的表。
5. **坐标损失不是 FAPE**：复用 `losses.sidechain_global_frame_aligned_loss`
   （stop-grad frame 上的 masked MSE）。APM 用 AF2 的 all-atom FAPE。这是一项
   独立的改动，不是疏忽。

## 损失

$$
\mathcal L=\lambda_{\rm coord}\,\mathcal L_{\rm coord}+\lambda_\chi\,\mathcal L_\chi
$$

三个边界，都在代码和测试里钉死：

1. **π 周期**表 ≠ `metrics.SWAPS`。后者还含 ARG NH1/NH2、LEU CD1/CD2、
   VAL CG1/CG2，那些是**原子命名歧义**，由坐标损失的原子置换解决，不是扭转
   周期性。两张表分开放，是刻意的。
2. **PRO 不监督 χ**。吡咯环闭合回主链 N，绕 CA–CB / CB–CG 的刚性旋转会拆环，
   `BuildSC` 因此保留 CCD 构象（`CHI_ROTATABLE=False`）。监督一个解码器无法
   实现的角度，只会教 head 输出一个不动任何原子的数。PRO 的**原子**仍由坐标
   损失监督，只是固定不动。
3. **χ 的参考系**：target χ 对着残基**自己的** N 量（与 `diagnose_packing` 报告
   口径一致），BuildSC 摆原子用的是**理想** N。理想骨架下两者完全相同；N 被扰动
   0.02 Å（真实量级）时差 < 2°。有测试钉住这个数。

## 不变性

- **IPA 是精确不变的**（测试里差 0.0）。
- **节点通道不是**，因为 `rotvecs` 是全局量——这是 APM 的设定，我们照搬。
  `packer_embed_rotvecs=False` 可以换回严格不变，有测试同时钉住两种行为，免得
  有人把「不变」当成默认配置的性质。
- **a_token 也不是**：它来自 AF3 式 DiffusionModule，读的是增强坐标系下的原始
  坐标，靠随机旋转增强而非等变性。
- **没有侧链坐标泄漏**：模块不读任何侧链坐标，连 template 都不读（有测试：把
  `noisy_coords` 换成 10 Å 量级随机数，输出逐位不变）。

## 2×2 实验

```bash
mkdir -p logs/training/sc_torsion_packer
bash scripts/training/slurm_sc_torsion_packer_hai.sh --dry-run   # 登录节点先跑
sbatch scripts/training/slurm_sc_torsion_packer_hai.sh           # array 0-3
```

| task | ARM | 序列通道 |
| ---: | --- | --- |
| 0 | `none` | 只有 BB + res_type（几何参照） |
| 1 | `a_token` | + Stage 1 的结构感知 token |
| 2 | `plm` | + 冻结 ESM-2 650M（**APM 自己的设定**） |
| 3 | `both` | 两个都有 |

课程直接复用 `slurm_official_sc_scratch_hai.sh`：monomer-only、native types、
native frames、frozen backbone、crop 384、bf16、accumulation 8、LR 5e-5、
warmup 2000、最多 50k step，每 2000 step 在 ≤491 条 held-out recent-PDB monomer
上评估。launcher 自己 `export AA_BACKEND=sc_only`，**不需要 FaMPNN 权重**。
`--seed 0` 显式传给四臂。

这个 2×2 能回答一个比原计划更准的问题：**a_token 携带的信息，PLM 是不是已经
有了**。sc_warmup 里 res_type 本来就是硬 one-hot 给进去的，PLM 多给的是同一条
序列的进化/上下文信息，a_token 多给的是结构感知的 trunk 信息。

### 预注册的验收标准（先绝对，后相对）

1. `bond_sc` / `angle_sc` 违例率 ≈ 0 —— BuildSC 的结构性保证。不为 0 说明接线
   错了，别读别的数。
2. 打得过 Dunbrack-mode 模板基线（不用模型，违例率 ~0）：局部 RMSD **1.277 Å**、
   χ1 recovery **68.7%**。打不过就不该进 V1。
3. 然后才看四臂差异：`symmetry_rmsd`、`chi_recovery_20deg`/`_40deg`、
   `chi1_accuracy_*`、`rotamer_recovery`、`torsion/chi_mae_deg`、clash 率。

判据：差异要在同一 held-out 集、同一 step 数上，**且大于臂间噪声**才算数。
四臂同 seed、同参数初值（有测试），但只有一个 seed，所以 <1% 的差异不构成结论。
另外 **`packer_random_torsion_input=True` 让前向是随机的**，评估数字本身带一点
方差——这是照搬 APM 的代价，读数时记得。

## 状态

**已验证**

- 全量 `pytest tests`：**755 passed, 2 skipped**（新增 `test_sc_ipa.py` 6 个、
  `test_sc_torsion_packer.py` 19 个、`test_sc_only_aa_backend.py` 7 个）。
- IPA 全局刚体不变性差 **0.0**；rotvec 的 log map 在 θ≈0 和 θ≈π 两个奇点都回环。
- 四臂 GPU smoke（job **117026**，30 步）**全部 COMPLETED**：loss 有限、
  `global_grad_norm` 6–10、ESM 只在 plm/both 两臂加载（日志各 1 行、另两臂 0 行）。
- **键长是常数，真实数据上验证**：随机初始化下 `bond_mae = 7.2e-4 Å`、
  `bad_bond_fraction = 0`。验收标准第 1 条由构造保证。
- 速度/显存：crop 384、bf16、accumulation 8 下约 **1.4 s/optimizer step**
  （smoke 117026 的第 1–30 步）。ESM-2 650M 前向 12–17 ms、2.7 GB，占比 ~1%。

**没有验证**

- packing 质量与任何 a-token / PLM 的结论。smoke 是 30 步随机初始化。
- 50k 步的 wall clock：按 1.4 s/step 外推 **~19.4 小时**，在 23:50 的额度内但
  余量不大。每 500 步存档，超时续跑代价可控。这是外推，不是承诺。

## 这次踩到并修掉的坑

按「后来才发作」的程度排序，前四条都是 CPU 全绿、GPU 才死的：

1. **`BuildSC` 的退化二面角**。没有 χ_k 的残基把 slot 0 索引四次，`atan2(0,0)`
   前向 0、反向 **NaN**，而 `torch.where` 消不掉（NaN × 0 仍是 NaN）。
   **这条同样存在于既有的 `chi_output` 路径上**，因为共用 `BuildSC`；这次一起修了。
2. **AngleResnet 对被 mask 的残基输出恰好 (0,0)**（`node_embed` 被 mask 乘零，
   残差分支 `init="final"` 零初始化，bias 零）→ `atan2(0,0)` 反向 NaN；
   `torsion_loss` 里的 `vector_norm(0)` 同理。前者在 atan2 之前替换输入，后者
   改成 openfold 的 `sqrt(sum+eps)` 形式。
3. **`_apply` 与 `nn.Module._apply` 撞名** —— PyTorch 在 `.to(device)` 时调它，
   于是每次搬设备都 TypeError。CPU 测试不搬设备所以看不见。
4. **新 worktree 的子模块补丁**：`patches/pxdesign-embedders-protenix-2.0.patch`
   不会自动应用，不应用则 dry-run 和 preflight 全过、GPU 第一步炸。新建 worktree
   后先 `cd $PROTEOAA_REPO/PXDesign && git apply ../patches/...`。
5. **`train_protenix_monomer.py` 不 seed**。`--seed` 只到数据采样器，模型初始化
   没被 seed；也就是说只差一个架构开关的两臂，**初值也不一样**——正好是 A/B 不能
   容忍的混淆。（`train_sc_adaptation.py` 一直是 seed 的，这个入口不是。）现在在
   建模型前 `seed_all(configs.seed)`，而且必须包含 numpy：AF2 式的
   `Linear(init=...)` 走 scipy 的 truncnorm，用的是 numpy 全局 RNG。
6. `sc_only` 下不可读的 fampnn 目录仍挂在 PYTHONPATH 上，`pip freeze` 遍历
   sys.path 时 PermissionError —— preflight 通过之后才死。
7. smoke launcher 用 `$(dirname "${BASH_SOURCE[0]}")` 找兄弟脚本，而 sbatch 跑的
   是 Slurm spool 里的副本。改成从 `$PROTEOAA_REPO` 解析。

## 已知限制

1. **侧链之间互相看不见**：所有 χ 在同一次前向里同时输出。one-shot packer 的
   共性限制，DiffPack 用自回归、FlowPacker 用迭代缓解；V0 用 clash 率量化代价。
2. **PRO 只能是 CCD 构象**，且被排除在 χ 指标之外（分母变了）。
3. **前向是随机的**（照搬 APM 的随机扭转输入）。
4. 坐标损失不是 FAPE（见上）。
5. 只有一个 seed。
