# AlphaProteo10 designability：第一次全量结果

运行日期 2026-09-11。协议见
[`alphaproteo10_designability_validation_zh.md`](alphaproteo10_designability_validation_zh.md)，
这份只写结果和解读。

## 摘要

| Backbone | Sequence | 打分条数 | coverage | 严格通过 | designability |
| --- | --- | --- | --- | --- | --- |
| 官方 PXDesign v0.1.0 | ProteinMPNN | 3280 | 1.0 | 271 | **8.26%** |
| Proteo-AA `111408/step6000` | ProteinMPNN | 3280 | 1.0 | **0** | **0.0%** |

两组 coverage 都是 1.0，3280 条全部真实打分完成。**这不是上一轮那种「任务没提交」的假零**
（上一轮只提交了 task 0–9，Proteo-AA 那 40 个 task 从未运行，汇总里的 0.0 是
`missing_scores_count_as_failures=true` 造成的假象）。

**但这个 0% 不能读作「Proteo-AA 针对靶点的设计能力为零」，因为 Stage III
从来没有在这个任务上训练过。** 它的训练目标是整个复合体一起加噪去噪，靶点
不是固定条件；benchmark 忠实地反映了这一点。详见
[采样器与协议不匹配](#采样器与协议不匹配)。这和 AlphaProteo / PXDesign 的
协议（靶点已知）不是同一个任务。

## 逐靶点

| target | PXDesign 通过 | PXDesign | Proteo-AA 通过 | Proteo-AA |
| --- | --- | --- | --- | --- |
| bhrf1 | 85 | 0.259 | 0 | 0.000 |
| pdl1 | 70 | 0.213 | 0 | 0.000 |
| ir | 63 | 0.192 | 0 | 0.000 |
| trka | 30 | 0.091 | 0 | 0.000 |
| sc2rbd | 11 | 0.034 | 0 | 0.000 |
| vegfa | 6 | 0.018 | 0 | 0.000 |
| h1 | 3 | 0.009 | 0 | 0.000 |
| tnfa | 3 | 0.009 | 0 | 0.000 |
| il17a | 0 | 0.000 | 0 | 0.000 |
| il7ra | 0 | 0.000 | 0 | 0.000 |

每个靶点 328 条（length 80–130 扫描，seed 42，n_step 400）。

## 0% 的直接原因：主链几何不成立

严格判据的三项分布：

| 指标 | 阈值 | PXDesign 中位 | 通过率 | Proteo-AA 中位 | 通过率 |
| --- | --- | --- | --- | --- | --- |
| `af2_binder_pred_design_rmsd` | < 1.5 Å | 0.69 | 0.801 | **16.00** | 0.000 |
| `pLDDT` | > 0.9 | 0.91 | 0.613 | **0.51** | 0.001 |
| `unscaled_i_pAE` | < 7.0 | 23.95 | 0.102 | **28.52** | 0.000 |

三项全部大幅失败，不是边缘擦过。往上游看一层，根源是生成的主链不是蛋白：

| | 连续 CA–CA 距离 | 偏离 3.8 ± 0.3 Å 的比例 |
| --- | --- | --- |
| PXDesign binder | median **3.83**（3.76–3.89） | **0.0%** |
| Proteo-AA binder | median **2.75**（0.84–5.02） | **93.9%** |

相邻 CA 最近到 **0.84 Å**，物理上不可能。连带现象全部自洽：

* 二级结构 `alpha 0.00 / beta 0.00 / loop 0.96`（PXDesign 是 `0.85 / 0.00 / 0.14`）
* ProteinMPNN 因为几何无法承载侧链而退化成多聚甘氨酸/丙氨酸 —— G+A 占比
  p90 达 **0.83**，**17% 的样本超过一半是 G/A**（PXDesign 侧 p90 只有 0.32，
  没有任何样本超过一半）
* 于是 AF2 折不出来：pLDDT 0.51、binder RMSD 16 Å

典型序列长这样：

```
SGGGEGGAPLPPEPGGAAAAALAALGAAGGAGGGGGGGGGAGAAALAGAAAGLAGGGGGGALGG...
```

`Rg = 12.95 Å` 对 100 残基是正常的紧凑值，端到端 22 Å —— 链没有炸开，是
**塌缩成了没有二级结构的团**，同时键长普遍偏短。

## 打分链路本身是可信的

Baseline 走**完全相同**的打分代码，几何完美、分布合理、拿到 8.26%。所以
0% 不是打分的问题。

顺带一个对解读 baseline 有用的观察：**PXDesign 自己的瓶颈是 `i_pAE`** ——
RMSD 过 80.1%、pLDDT 过 61.3%，但 `i_pAE < 7.0` 只过 **10.2%**。也就是说
这个 benchmark 卡的主要是「AF2 认为界面靠不靠谱」，不是「binder 自己折不折得好」。

## 采样器与协议不匹配

这是这次运行最重要的发现，也是 0% 不能当结论读的原因。

追查时我先注意到一个反常：**Proteo-AA 那一组的靶点链也是坏的。**

| | 靶点链 A（157 残基，无残基编号跳跃） |
| --- | --- |
| PXDesign 组 | 坏键 **0.6%**，min 3.15 Å |
| Proteo-AA 组 | 坏键 **48.7%**，min 2.20 Å |

靶点是 2wh6 的晶体坐标，同样的 crop、同样的编号、同名文件。模型本不该碰它。

**我因此一度判断是 harness 把该原样保留的坐标写坏了。核对后这个判断是错的。**
两组的 manifest 记录：

```
pxdesign_official       model_mode=pxdesign   sampler_mode=pxdesign_native  n_step=400
stage3_111408_step6000  model_mode=proteoaa   sampler_mode=pxdesign_native  n_step=400
```

`--mode` 只决定写哪些 sequence arm，**不切换采样路径**
（`generate_alphaproteo_designs.py:160-200`）。同一个脚本、同一个采样器、
同样 400 步，**唯一差别是 checkpoint**。

真正的原因是：`pxdesign_train/cogenerate.py` 的 `cogenerate()` 是
**自由共生成**采样器，它的 docstring 自己写明了：

> Co-generate (backbone coordinates, residue sequence) **from noise**. …
> its GT coordinates are **NOT used** — structure starts from noise.

它**没有任何固定原子参数**，也没有 `torch.where(atom_design, …)` 之类的还原逻辑。
整个复合体（含靶点）都是从噪声里重建出来的。

于是两组靶点链质量的差异就有解释了：官方 PXDesign 权重强到能把靶点近乎完美地
重建回来（0.6% 坏键），Proteo-AA `step6000` 做不到（48.7%）。

**后果**：这个 harness 测的是「从噪声同时重建靶点 + 设计 binder」，而
AlphaProteo 和 PXDesign 的协议是**把靶点作为已知条件给定**。任务被显著加难，
而加难的那部分（靶点重建）不是我们要测的能力。官方模型强到不受影响，
`step6000` 被这一项直接归零。

### 这不是 benchmark 接错函数，而是 Stage III 本身没有靶点条件

一开始我以为换个目标条件化的采样器就能修。**不对。** 训练侧的固定靶点条件化在
`pxdesign_train/generator.py:101`：

```python
if input_feature_dict.get("stage4_fixed_context", False):
    design = input_feature_dict["design_token_mask"].bool()
    atom_design = design[input_feature_dict["atom_to_token_idx"].long()]
    x_input = torch.where(atom_design[..., None], x_input, x_gt_aug)   # 靶点压回真值
```

而 `stage4_fixed_context` 只有 Stage IV 会设（`model.py:999`，门控在
`uses_codesign()` 上）。三条独立证据指向同一结论：

* `fixed_atom_xyz` / `fixed_atom_mask` 的**唯一消费者**是 `codesign.py` 和
  `stage4.py`，Stage III 没有任何消费者
* 坐标 loss 用 `coordinate_mask`（全部已解析原子），不是 binder-only ——
  Stage III 被监督去重建整个复合体
* Stage III 的 slurm 脚本不传 `--use-template` / `--use-msa`，没有 template
  通道能把靶点结构干净地送进去

**Stage III 训练时整个复合体一起加噪，靶点从来不是干净的给定条件。**
模型唯一知道「哪些残基是设计区」的途径是 `design_token_mask` 和被设成 XPB 的
`restype`，几何上它没有任何锚点。

所以 `cogenerate()` 的自由共生成**恰恰是匹配 Stage III 训练的推理路径**。
这个 benchmark 是忠实的，不是接错了。**把它换成目标条件化的采样器会让
Stage III 落到分布外，只会更差，不会更好。**

目标条件化是 **Stage IV 才引入的**（`stage4_fixed_context`，以及
`stage4.py:277,284,285` 每步去噪后的 `torch.where(atom_design, xyz, fixed_xyz)`）。
换句话说，Stage IV 是第一个真正在训练「针对给定靶点做设计」这个任务的阶段。

## 能得出和不能得出的结论

**能**：

* 打分流水线（ProteinMPNN + AF2-IG + 严格过滤 + 汇总）端到端可用，20 个 task
  全部完成、coverage 1.0。
* 官方 PXDesign 在这个 harness 下拿到 8.26%，主链几何完美，瓶颈在 `i_pAE`。
* Proteo-AA `111408/step6000` 在**自由共生成**设定下无法产出几何成立的主链
  —— 93.9% 的 CA–CA 键长越界。这一条本身是真实的、值得重视的。

**不能**：

* 不能说 Proteo-AA 的 binder 设计能力是 0。靶点没有被固定，测的不是这件事。
* 不能把 8.26% 和 AlphaProteo / PXDesign 论文里的数字直接比 —— 协议不同
  （靶点非固定），而且我们的 hotspot 集合对 tnfa 用的是 PXDesign 的宽集合
  （见 `benchmarks/alphaproteo10/targets/tnfa.yaml` 的注释）。
* 不能把这次的 0% 当作 `step6000` 这个 checkpoint 的最终评价。它本身也不是
  完成的 Stage III —— job 111408 在 30000 步里 6650 步超时，step6000 是它的
  最后一个 checkpoint。

## 复现

产物根目录：

```
/hai/scratch/shenjm/proteo_aa_runs/alphaproteo10_designability/full_111408_step6000/
├── generation/{pxdesign_official,stage3_111408_step6000}/<target>/proteinmpnn/*.cif
├── scores/<model>/<target>/proteinmpnn/sample_level_output.csv
└── summary/designability_{summary,by_target}.csv
```

生成：job 114026（PXDesign，8h58m）+ 114027（Proteo-AA，8h23m），
各 10 targets × 328 条。打分：队列管理器 job 114314，20 个 task，
每个约 38 分钟，4 并发，06:03 完成。

主链几何可以这样复查：

```python
import gemmi, numpy as np
st = gemmi.read_structure(cif); st.setup_entities()
ch = next(c for c in st[0] if c.name == "Z")          # Z = binder
ca = np.array([r["CA"][0].pos.tolist() for r in ch if r.find_atom("CA", "*")])
d  = np.linalg.norm(np.diff(ca, axis=0), axis=1)
print((np.abs(d - 3.8) > 0.3).mean())                  # 坏键比例
```

注意 `sample_level_output.csv` 里 AF2 指标是**带方括号的列表字符串**
（`[0.88]`），直接 `pd.to_numeric` 会整列变 NaN；要先 `.str.strip("[] ")`。
`designability_by_target.csv` 的 `mean_*` 列目前是空的，汇总脚本没有填。

## 下一步

1. **先确认研究计划的意图**，这是个方向问题不是修 bug：Stage III 到底要不要
   做「针对给定靶点的 binder 设计」？
   * 如果要 —— 训练目标缺了靶点条件化，`generator.py:101` 那个分支应该对
     Stage III 也打开（代价是改变训练任务，之前所有 Stage III 结果不可直接续比）。
   * 如果不要、由 Stage IV 承担 —— 那么 **Stage III 不该用 AlphaProteo-10
     评测**，或者只能作为「整复合体自由共生成」的基线来读，不能和论文数字并列。
2. **不要**只把 benchmark 换成目标条件化采样器。那会让 Stage III 落到分布外，
   得到一个既不反映训练也不反映论文协议的数。
3. Stage III 的一步去噪指标看起来是正常的（111408 报告里 crop 448 下
   val Cα RMSD ≈ 3.2 Å），而 400 步自由生成塌成 93.9% 坏键。扩散模型训练指标
   与自由采样质量脱节本身常见，但这个量级值得单独查：去噪终点、sigma 调度、
   以及 `pxdesign_native` 采样器和这个 checkpoint 的约定是否一致。
4. 顺手补两个小坑：汇总脚本的 `mean_*` 列没填；AF2 指标的列表字符串格式
   容易让人整列读成 NaN。
