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

**但这个 0% 也不能读作「Proteo-AA 的设计能力为零」。** 原因见
[采样器与协议不匹配](#采样器与协议不匹配)：这个 harness 不把靶点当固定条件，
所测任务比 AlphaProteo / PXDesign 的协议难得多，而变难的那部分不是我们想测的东西。

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

**目标条件化的采样器在代码库里是存在的** —— `pxdesign_train/stage4.py` 的
`generate()` 有明确的固定原子还原：

```python
xyz = torch.where(atom_design[None, :, None], xyz, fixed_xyz)
```

它在每一步去噪后都把非设计原子压回给定坐标。所以这不是缺功能，是这个
benchmark 接错了函数。

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

1. **把 benchmark 接到目标条件化的采样器上**，然后重测 Proteo-AA 这一组。
   在靶点固定之前，Proteo-AA 的 designability 没有意义。参照
   `stage4.generate()` 的固定原子还原做法。
2. 重测后如果几何仍然不成立，那才是模型本身的结论，值得单独查（键长普遍偏短
   这个特征很具体，可能指向去噪终点或 sigma 处理）。
3. 顺手补两个小坑：汇总脚本的 `mean_*` 列没填；AF2 指标的列表字符串格式
   容易让人整列读成 NaN。
