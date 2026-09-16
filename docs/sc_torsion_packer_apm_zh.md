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
   > **这条预注册写错了，原样留着以存档。** 1.277 Å 是另一个估计量在另一批链上的
   > 数，和 packer 被打分的 `symmetry_rmsd` 不可比。同口径下的正确门槛是
   > **2.086 Å / χ1 0.703**，见下面「基线口径」一节。
3. 然后才看四臂差异：`symmetry_rmsd`、`chi_recovery_20deg`/`_40deg`、
   `chi1_accuracy_*`、`rotamer_recovery`、`torsion/chi_mae_deg`、clash 率。

判据：差异要在同一 held-out 集、同一 step 数上，**且大于臂间噪声**才算数。
四臂同 seed、同参数初值（有测试），但只有一个 seed，所以 <1% 的差异不构成结论。
另外 **`packer_random_torsion_input=True` 让前向是随机的**，评估数字本身带一点
方差——这是照搬 APM 的代价，读数时记得。

## 结果（2026-09-15，job 117035，四臂各 50000 步）

四臂全部 COMPLETED，16.6–18.5 h。step 50000、308 条 held-out monomer：

| arm | symmetry_rmsd ↓ | χ1 acc 40° ↑ | χ1+χ2 acc ↑ | chi_rec 40° ↑ | rotamer_rec ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Dunbrack-mode 模板基线** | **2.086** | **0.703** | — | — | — |
| none | **1.509** | 0.765 | 0.590 | 0.696 | 0.565 |
| a_token | 1.529 | 0.765 | 0.586 | 0.696 | 0.566 |
| plm | 1.520 | **0.768** | **0.591** | **0.699** | **0.567** |
| both | 1.537 | 0.764 | 0.587 | 0.697 | 0.565 |

`bond_mae` 5.78e-4 Å、违例率 2.06e-5，四臂逐位相同。

### 验收标准：第 1 条通过，第 2 条通过，第 3 条无结论

1. **键长/键角违例率 ≈ 0** —— 成立，且由 BuildSC 结构性保证，不是学出来的。
2. **打得过 Dunbrack 模板基线** —— 成立，**2.086 → 1.509 Å（−28%）**，χ1 recovery
   0.703 → 0.765。
3. **四臂差异** —— 极差 0.028 Å，而单臂在最后 5 个验证点上的自身波动就有
   0.008–0.021 Å。**同量级，不构成结论。**

### 基线口径：一个必须先排掉的陷阱

我一开始拿 `docs/sidechain_config_notes.md` 里的 **1.277 Å** 当验收门槛，据此得出
"打不过模板基线"——**那是错的**。仓库里有三个模板基线数字，它们是三个不同的量：

| 估计量 | 模板基线（同一批 308 条链上重测） | 出处 |
| --- | ---: | --- |
| mean per-residue RMSD（每残基 RMSD 再平均，不对齐） | 1.172 Å | `eval_template_quality.py` 报 1.277（33 条手挑链） |
| atom-weighted RMSE（sqrt 全局 MSE，不对齐） | 2.174 Å | `eval_sidechain_template_baseline.py` 报 2.18（491 条） |
| **`symmetry_rmsd`（sqrt 全局 MSE + 对称对齐）** | **2.086 Å** | **packer 被打分用的就是这个** |

sqrt-of-mean ≥ mean-of-sqrt（Jensen），加上链集不同、对称对齐只有一边做 —— 三者
本来就不可比。`scripts/evaluation/eval_template_baseline_matched.py` 在**同一批 308
条链、同一 mask、同一估计量**下把三个口径一次算全，并交叉验证了 mask：它数出每条链
**820.43** 个受监督原子，与训练 eval 日志里的 `val_sc_observed_atoms=820.4` 一致。

**教训**：拿一个文档里的数字当验收门槛之前，先确认它和你要比的东西是同一个估计量。

### a_token 为什么没用：最可能的解释

这个相位喂给 packer 的是 **native 骨架 + native frame + native 序列**
（`predicted_frame=False`、`force_gt_type_logits=True`），而 a_token 是**同一份
native 骨架**在 σ=0.4 条件下过一遍冻结 trunk 得到的结构感知 embedding。IPA 已经
直接读着这份几何了，所以 **a_token 在这里与几何输入高度冗余**。

关于那个 σ：坐标上**没有加噪**（`x_noisy` 就是 native 坐标）。σ 只进两处，
`c_in=1/√(σ²+σ_data²)` 在 σ∈[0.01,1] 区间几乎不变（0.06250→0.06238，σ_data=16
主导），真正起作用的只有时间嵌入 `ln(σ/σ_data)/4`。所以 **σ=0 不可表示**
（ln 0 = −∞），而且 trunk 是个 denoiser、σ 是它输入契约的一部分：训练噪声是
log-normal(−1.2, 1.5)，中位数 σ=0.301，`feature_sigma=0.4` 正落在 57.5 分位。
σ→0 是分布外外推，不是"更干净的表示"。**这个 trunk 上不存在"σ=0 的 a_token"。**

要证伪"冗余"这个解释，该做的是让 a_token 携带几何以外的信息 —— 即把骨架换成
**预测的/带噪的**（`predicted_frame=True`，Stage III/IV），而不是调 σ。

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

---

# 第二轮：完全沿用 APM 的 architecture + code 逻辑，换成 yfsun 新生成的 data

（2026-09-15 起。上面那一轮跑在 PXDesign 的数据和我们自己的损失上；这一轮的目标是让
**除了序列条件这一路以外，一切都是 APM 的**。）

## 先说结论：这一轮不"移植"损失，而是直接调 APM 的代码

上一轮的损失是我们自己的 `sidechain_global_frame_aligned_loss`。APM 在
`train_packing_only` 下的损失是（`flow_module.py:model_step`）：

```
train_loss = supervised_chi_loss(chi_weight=1, angle_norm_weight=0.02)   # openfold
           + cal_sidechain_fape_loss(...)                                # flow_module:461
```

两项权重都是 1.0——`torsions_loss_weight: 0.5` 在 `pretrain_sidechain.yaml` 里确实写了，
但它只作用于 `line 415` 的 multiflow 式 `torsions_vf_loss`，走不到这个分支。

`cal_sidechain_fape_loss` 依赖 openfold 的 rigid-group 表、`atom14_*` ground truth 和
原子改名（`compute_renamed_ground_truth`）。本来打算移植，后来发现**不需要**：
apm_reference 里 vendored 了完整的 openfold，缺的只有三个导入期依赖。补上之后，
APM 的 featuriser 和 loss 在我们的 env 里直接可调。

```
pip install --no-deps --target /hai/scratch/shenjm/pyextra dm-tree lightning-utilities torchmetrics
/hai/scratch/shenjm/pyextra/torch_scatter.py    # 纯 torch 的 scatter/scatter_add 垫片
PYTHONPATH=...:/hai/scratch/shenjm/apm_reference:/hai/scratch/shenjm/pyextra
```

装在 env **旁边**而不是 env 里，所以 proteoaa 本身没被改动。
（`torch_scatter` 只在 `apm/data/utils.py` 的模块级 import 和 batch-OT 代码里用到，
走不到 featurisation 路径；垫片仍然写了真实实现，因为一个"能 import 但算错"的
stub 比 import 失败更糟。）

## 这一轮各部分的来源

| 部分 | 来源 | 文件 |
|---|---|---|
| 网络 | 我们的移植（已证明等价，max\|diff\| 1.0e-05 rad） | `pxdesign_train/sidechain/packer.py` |
| featurisation | **APM 本人** `_process_csv_row_FAESM` | `pxdesign_train/sidechain/apm_dataset.py`（只是包一层） |
| L_chi | **openfold 本人** `supervised_chi_loss` | `pxdesign_train/sidechain/apm_loss.py` |
| sidechain FAPE | APM 的 glue 逐行誊写，每个张量操作仍是 openfold 的函数 | 同上 |
| batch sampler | `LengthBatcher_nonRep` 的单副本复刻 | `scripts/training/train_packer_apm_data.py` |
| 过滤 | **APM 本人** `_length_filter` / `_max_coil_filter` / `_rog_filter` | 同上 |
| 优化器 | AdamW(1e-4, β=0.95/0.999)，clip 5.0 norm | 同上 |

## 一个刻意保留的 APM bug

`apm/data/datasets.py:247`：

```python
torsions_1    = torsion_angles[:, -4:]     # chi1..chi4
bb_torsion_1  = torsions_1[:, :3]          # 于是 bb_torsions_1 = chi1..chi3
```

而 `cal_sidechain_fape_loss` 把 `bb_torsions_1` 拼到 `torsion_angles_to_frames`
读作 omega/phi/psi 的前三个槽位。也就是说：**APM 把 chi1-3 当成 omega/phi/psi 喂进去了。**

影响范围有限：这三个角只驱动 rigid group 1-3，atom14 里只有 O 原子属于 group 3(psi)，
所以后果是主链 O 的 frame 错了、给 FAPE 贡献一个近似常数项，侧链本身没被污染。

**我们照抄了这个 bug。** 理由：released checkpoint 就是在这个目标函数下训出来的，
在这里"修好"就意味着新的 run 优化的东西和 reference 从来不是同一个。

## 数据

`/hai/scratch/yfsun/apm/extracted/data_APM`

| 目录 | 数量 | 这一轮是否用 |
|---|---|---|
| `pdb_monomer` | 18,684 | **是** |
| `pdb_multimer` | 11,620 | 否 |
| `afdb` | 235,692 | 否 |

`meta_data.csv` 里 `train_set` 那 18,684 行和 `pdb_monomer` 一一对应，且已经是
APM 的过滤结果（60–384 残基、coil ≤ 0.5、monomeric、单链）。跑一遍 APM 自己的
过滤函数确认，只有 rog_quantile 会再掉一点。

**AFDB 和 multimer 是有意排除的偏离**，APM 的 `pdb_dataset` 里 `use_AFDB: True` /
`use_multimer: True`。排除理由：AFDB 的侧链是 AlphaFold 的预测，用它训 packer 学到的是
"复现 AF2 的 packing"而不是晶体学；multimer 排除是为了让训练集和 monomeric 的
post-2021 验证集口径一致。四个 arm 用同一份数据，所以 arm 之间的比较不受影响。

一个 epoch = 每个 cluster 采 1 条 = 3,694 条 = 540 个 batch（单副本）。
APM 用 8 卡、accumulate 1，所以一次 optimizer step 吃 8 个 batch；我们单卡 `--accum 8`
对齐有效 batch，代价是梯度是累积出来的而不是 all-reduce 出来的。

**一个 batch 里全是同长度的链**（按 `modeled_seq_len` 分组），所以全程没有 padding。
batch size = `min(64, 400000 // L² + 1)`，因此 L=384 时只有 3 条、L≤79 时才吃到 64。

还有一条 APM 的行为值得记下来：当某个长度组凑不满 batch 时，
`batch_repeats = bs // len(batch)` 会把这些样本**重复**填满。实测一个真实 epoch：
3,694 条 unique、6,373 个 forward slot，**平均重复 1.73 倍**。
也就是说名义 batch size 高估了每步看到的不同蛋白数。这是他们的代码，照抄。

sampler 是这一轮唯一"复刻"而非"调用"的部分（`LengthBatcher_nonRep` 是按分布式
写的），也因此是最可能悄悄跑偏的一块——batch 组成不一样不会报错，只是变成另一个实验。
所以做了逐 batch 比对：在真实过滤后的 index 上，epoch 0/1/5 各 526/528/528 个 batch，
**和 APM 本人的输出 index 对 index 完全相同**。固化成 `tests/test_apm_sampler_parity.py`
（apm_reference 不在 PYTHONPATH 上时跳过）。

## 参数量的实话

移植对齐到 released 配置之后：

| | 参数 |
|---|---|
| 我们的模块，合计（四臂相同） | **17,498,890** |
| 其中和 APM checkpoint 同名同形状 | 16,896,244（546 张量，逐个核对过） |
| 我们自己多出来的 | `a_proj` 196,864 + `plm_conditioner` 396,322 + `atom_embed` 9,472 |
| checkpoint 里在我们这边是 buffer 而非参数的 | 两个 `torsion_embedding.freq_bands`，各 6 个常数 |

（之前记的 10.28M 是对齐之前那个更小的默认配置，不是这个。）

四个 arm 构造**同一套参数**是刻意的——消融要改的是信息不是容量，
所以 `none` 臂上 `a_proj` 也照样构造，只是乘 0。代价是 `plm_conditioner`
在非 PLM 臂上永远拿不到梯度。一度把它改成按需构造，被
`test_the_four_arms_share_one_parameter_set` 拦下来了；那个测试是对的，已回退。
改成训练时**实测**并打印：

```
PARAMS {"total": 17498890, "receiving_gradient": 17093096,
        "inert_modules": ["atom_embed", "plm_conditioner"]}
```

让容量报告诚实，而不是为了报告好看去改模型。
（`atom_embed` 不吃梯度是因为这一轮的损失只读扭转角，不读它喂给
Protenix diffusion head 的那两路原子特征。）

## a_token 的特征桥：noise 到底是多少

用户要求 noise = 0 而不是 0.4。这里必须把两件都叫"noise"的事分开：

1. **加在坐标上的高斯扰动** —— 设成**严格的 0**。用的是已经存在的
   `residue_type.clean_coordinate_input`，trunk 看到的是原生 backbone。

2. **写进 time channel 的那个数** —— EDM 的条件量是
   `c_noise = ln(σ/σ_data)/4`（`Protenix/protenix/model/modules/diffusion.py:162`），
   **σ=0 时是 −∞**。所以 σ=0 不是"要不要"的问题，是这个网络根本无法表示的输入。
   条件 σ 因此钉在 `--sigma-floor`，默认 **4e-4**，即 Protenix 自己的 `s_min`
   （`configs_base.py:172`）——采样 schedule 的终点噪声，也就是 trunk 被要求
   条件化过的最干净状态。它带来的 `c_in = 1/sqrt(σ²+σ_data²)` 与 σ=0 处相差
   3e-10（相对）。

实际用的数字写进 cache manifest，不让任何读者去猜。

`--check` 模式在训练前验四件事，因为它们各有各的"静默出错"方式：

- **长度**：a_token 的行数 == packer 看到的残基数
- **对齐**：第 i 行是不是 packer 的第 i 个残基（按三字母名比对；光看长度会漏掉
  "一个插入配一个删除"）
- **确定性**：重复 forward 是否复现
- **σ**：regime A（坐标零噪声 + σ_floor）和 regime B（有噪声 + σ=0.4，即上一轮
  a_token 臂训练时的设定）之间 a_token 到底差多少 —— 这是测出来的，不是断言的

### 桥为什么要绕 mmCIF

trunk 吃的是 Protenix 的 feature dict，APM 的 pkl 只有裸数组。走
`APM pkl → mmCIF → CifFileProvider → DesignSourceDataset` 是因为这条路
CASP benchmark 和 design run 已经在用，命名/顺序/entity 的约定都被跑过了。

一个必须处理的细节：`_process_csv_row_FAESM` 切的是
`modeled_idx.min() .. .max()` 这个**连续区间**，不是 `modeled_idx` 本身。
抽样 300 条，11% 的链区间里有没有坐标的残基（占残基总数 0.19%）。
只写有坐标的残基会让 a_token 比其它张量短几行，并且每个 gap 之后整条链错位一格。
所以这些残基通过 `entity.full_sequence` 进 `_entity_poly_seq`，被 Protenix
tokenise 成 unresolved —— 它们本来就是 unresolved，`res_mask` 也已经这么说了。

坐标先过 APM 的 `parse_chain_feats`（CA 质心平移、未观测原子置零），
让 trunk 读到的数和 packer 建 frame 用的数是同一份，而不是它的一个平移。

## 现在的状态

- `none` / `plm` 两臂：已具备提交条件，等 smoke（job 117525）确认后提交正式 run。
- a_token 桥：job 117532 在跑一致性检查；通过之后才会预算 a_token / both 两臂。
- 评测：`eval_on_apm_testset.py` 新增 `apmdata_<arm>` 方法名，和旧的 `ours_<arm>`
  分开列——它们是不同的模型，合在一个标签下会分不清。新臂在评测时拿 APM 的
  `diffuse_mask` 语义（每个 modelled 残基都是 1），旧臂拿它们训练时的语义
  （侧链归属，GLY=0），因为这是网络输入不是记账掩码。

## 代码在哪里

| | 路径 |
|---|---|
| 我们的代码 | `/hai/scratch/shenjm/wt_torsion_packer`（branch `sjm/sc-apm-torsion-packer`） |
| APM 原始代码 | `/hai/scratch/shenjm/apm_reference`（`github.com/bytedance/apm` @ `a98e59b1`） |
| APM 权重 / 测试集 | `/hai/scratch/shenjm/apm_weights` |
| 新数据 | `/hai/scratch/yfsun/apm/extracted/data_APM` |
| 旁装依赖 | `/hai/scratch/shenjm/pyextra` |
