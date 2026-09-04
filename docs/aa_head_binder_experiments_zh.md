# Stage III Binder 联合训练与 AA Head 实验总结

> 整理日期：2026-09-02
>
> 分支：`sjm/binder-design-training`
>
> 范围：Stage III binder 联合训练、AA head 代码审计、PINDER validation、sigma 对照、轨迹 readout 和 tiny-overfit 诊断。

## 1. 核心结论

目前的证据不支持“AA label 接错”“binder/receptor mask 反了”“AA head 被冻结”或“两个 checkpoint 嫁接不兼容”是主因。AA head 的参数确实在更新，监督对象也是 binder 上的 native residue type。

AA head 也不是完全没有学习能力：在 PINDER native backbone 的固定噪声评估中，Stage III checkpoint 在中等 sigma 下可以达到约 **15.8% accuracy**；在固定 32 个 PINDER 样本上的 head-only tiny-overfit 中，训练准确率可以从约 8% 上升到 **20.1%**。但是，这个学习信号很弱，并且没有继续转化为真实联合生成轨迹上的序列恢复能力。

目前最一致的解释是以下问题叠加：

1. AA head 本身很浅，主要依赖 diffusion trunk 已经编码好的局部结构表示；现有表示对 residue identity 的可分性不足。
2. 训练使用的是 native structure 加单步噪声，而推理使用从 Gaussian noise 开始的多步 rollout，存在明显的 train–inference state distribution shift。
3. 所有 binder residue 在训练和推理时都被同时 mask，模型没有已知 binder sequence context，也没有把已预测序列自回馈给 trunk。
4. 低 sigma 并不是唯一问题：强制 low-sigma 训练没有带来可见提升；固定 native backbone 上中等 sigma 更好，但同一生成轨迹上选择中间 sigma readout 反而没有改善。
5. 类别分布和预测存在明显塌缩，balanced accuracy 和 macro-F1 显著低于表面 accuracy。

因此，现在还不能简单下结论说“这个网络结构一定学不到”；更准确的说法是：**当前 shallow AA readout + one-step noisy-native supervision 的组合，没有学到足以迁移到 free rollout 的 sequence–structure mapping**。下一步应先完成严格匹配的 last-block tiny-overfit，再决定是做 rollout-aware training，还是直接引入更强的 inverse-folding head。

## 2. 不同评估的含义

几类指标不能混在一起解释：

| 评估 | 输入结构 | 能回答的问题 | 不能直接回答的问题 |
| --- | --- | --- | --- |
| 训练日志中的 `val_*` | monomer validation | Stage III 是否保留/改善 monomer fold prior | binder 生成质量、PINDER AA recovery |
| PINDER fixed-sigma | native binder backbone 加指定噪声 | 给定接近 native 的结构表示时，AA head 能否识别 native AA | free generation 是否能生成 native binder |
| PINDER free co-generation readout | 从 Gaussian noise 多步生成 | 实际 rollout 上 AA readout 的行为 | AA head 的纯分类上限；这里还混入了生成 backbone 偏离 native 的影响 |
| Tiny-overfit | 重复固定的少量训练样本 | 参数、梯度和模型容量是否至少能记忆训练集 | 泛化能力和真实 binder 设计能力 |

尤其要注意：PINDER free-generation 的 native sequence recovery 同时受 backbone 是否接近 native 构象影响。一个生成出来但不同于 native 的合理 binder，未必应该恢复出 native sequence，所以该指标不能单独等价为 AA head accuracy。

## 3. 代码逻辑审计

### 3.1 已确认没有发现硬错误的部分

- Native residue type 在结构加噪前保存为 `aa_clean`，不是拿被 mask/扰动后的 token 当 label。
- AA CE 只在目标 binder residue 上计算；PINDER 使用 chain B 作为 binder，Protenix PPI 使用第二条 chain，未发现 binder/receptor selector 反转。
- 严格重建检查按 `(chain_id, res_id)` 对齐，未发现 label 在 crop 或重排后整体错位。
- AA logits 来自每个 diffusion sigma 对应的 `a_token`，CE 正确覆盖 diffusion samples，并进入总 backbone loss。
- alternating optimization 下 AA head 属于 backbone optimizer，未被冻结。
- Crop-448 step 6000 与初始 donor 相比，AA head 的 781k 个参数元素发生变化，相对参数变化约 **3.63%**，进一步排除了“没有更新”。
- 对用于嫁接的 checkpoint 做过逐 tensor 核对：diffusion trunk 为 **636/636** 一致，design-condition embedder 为 **96/96** 一致。因此目前可以排除 backbone 不兼容导致的静默错配。
- `complete_unmask` 与当前训练中的 all-binder masking 语义一致，没有发现训练/推理 mask 定义相反。

### 3.2 代码上仍存在的建模限制

- AA head 基本是逐 token 的浅层 MLP：`LayerNorm -> Linear -> ReLU -> Linear`。跨 residue 和几何上下文必须预先由 diffusion trunk 写入 `a_token`。
- 当前 AA 监督是 one-step noisy-native supervision：对 native coordinates 加指定 sigma 的噪声后直接预测 residue type。
- 实际推理是从随机坐标开始的多步 EDM rollout，状态分布与训练输入不同。
- 在 `mask_mode=all`、`aa_t=1` 下，整个 binder sequence 同时未知；已预测 AA 不会在后续 denoising step 中反馈给 trunk。
- 主要 AA logits 位于 side-chain/refinement 后处理之前，缺少利用最终 `S_phi` 或 refined backbone 表示的直接 AA supervision。

### 3.3 数据侧的限制

- PINDER 当前按条目抽样，同一 cluster 内的相似复合物可能重复出现，实际结构与序列多样性可能低于样本条数显示的规模。
- 目前还没有一套与 monomer validation 完全同规模、同 crop、同采样规则的 binder validation，因此不同阶段之间应优先比较同一 PINDER protocol 下的结果。

## 4. 已完成实验

### 4.1 AA donor 的严格 monomer 评估

对 AA donor step 9000 做了 fixed-sigma、固定 channel 的严格评估：

| Sigma | AA CE | Accuracy |
| ---: | ---: | ---: |
| 0.04 | 2.878 | 10.624% |
| 0.4 | 2.794 | 13.398% |
| 4.0 | 2.804 | 13.278% |

- 数据集的 majority-class baseline（LEU）为 **9.154%**。
- 在 sigma 0.4 随机化结构输入后，accuracy 降到 **8.331%**，macro AUROC 约 0.499，说明模型确实使用了一些结构信号，不是只输出类别先验。
- 但预测严重集中在 LEU、VAL、GLY、GLU、ALA，多个低频类别 recall 为 0，说明 donor AA classifier 本身已经存在类别塌缩。

结果文件：

`/hai/scratch/yfsun/proteo_aa_runs/eval_aa_head_strict_backbone/aa_head_on_stage2_step9000_fixedchannels/strict_aa_summary.csv`

### 4.2 Stage III mixed-data binder 联合训练

主实验：

- Job 107903：crop 512，约运行到 step 5800，因 23:50 Slurm walltime 结束。
- Job 107904：crop 448，运行超过 step 6200，获得 step 6000 validation。
- step 6000 左右的数据比例约为 monomer 42.3%、Protenix PPI 17.3%、PINDER 40.4%。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后没有再发生之前的显存碎片 OOM。

#### Crop 448：内部 monomer validation

| Step | Val loss | AA CE / acc. | SC local | BB post | MSE | Cα / BB RMSD | TM-score | lDDT loss | Distogram CE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 111.0 | 2.805 / 0.1285 | 1.816 | 21.15 | 21.25 | 3.671 / 3.621 | 0.7705 | 0.1833 | 1.818 |
| 4,000 | 107.3 | 2.801 / 0.1302 | 1.836 | 19.90 | 20.64 | 3.566 / 3.518 | 0.7751 | 0.1885 | 1.684 |
| 6,000 | 99.93 | 2.799 / 0.1310 | 1.818 | 17.69 | 19.35 | 3.446 / 3.398 | 0.7866 | 0.1858 | 1.473 |

#### Crop 512：内部 monomer validation

| Step | Val loss | AA CE / acc. | SC local | BB post | MSE | Cα / BB RMSD | TM-score | lDDT loss | Distogram CE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 113.6 | 2.805 / 0.1279 | 1.820 | 21.68 | 21.76 | 3.625 / 3.574 | 0.7741 | 0.1874 | 1.869 |
| 4,000 | 115.4 | 2.803 / 0.1273 | 1.806 | 21.55 | 22.24 | 3.736 / 3.687 | 0.7650 | 0.1832 | 1.609 |

结论：

- Crop 448 的 backbone validation 持续改善，step 6000 没有出现旧版 Stage III 在 step 4000 之后的回退。
- Crop 512 吞吐更低，并在 step 2000 到 4000 间出现几何指标回退；现阶段 crop 448 更稳定。
- AA CE 和 accuracy 基本不变，始终约为 2.80 / 13%。但这里是 monomer validation，不能用来判断 PINDER binder recovery。
- Crop 448 step 6000 相比 monomer-only Stage III v1 的 backbone 指标更好，但两者 validation 样本数和 crop 不同，只能作为趋势参考。

与 monomer-only Stage III v1 的 step 6000 数值对照如下：

| Metric | Monomer Stage III v1 | Binder crop 448 |
| --- | ---: | ---: |
| Val loss | 105.7 | 99.93 |
| AA CE / accuracy | 2.796 / 0.1310 | 2.799 / 0.1310 |
| SC local | 3.295 | 1.818 |
| BB post | 18.11 | 17.69 |
| MSE | 20.33 | 19.35 |
| Cα / BB RMSD | 3.581 / 3.533 | 3.446 / 3.398 |
| TM-score | 0.7628 | 0.7866 |
| lDDT loss | 0.1806 | 0.1858 |
| Distogram CE | 1.342 | 1.473 |

这里 Stage III v1 使用 308 个 proteins、crop 384，而 binder run 使用 128 个 proteins、crop 448，所以不能把差值解释为严格的模型胜负。

主日志：

- [`proteo-aa-stage3-binder-107903.out`](../logs/training/stage3_binder/proteo-aa-stage3-binder-107903.out)
- [`proteo-aa-stage3-c448-107904.out`](../logs/training/stage3_binder/proteo-aa-stage3-c448-107904.out)

### 4.3 AA learning-rate 与 detach 对照

两个 1000-step mixed-data 对照：

| Job | 配置 | Step 1000 overall AA CE / acc. | PINDER AA CE / acc. |
| --- | --- | ---: | ---: |
| 108695 | AA head LR `1e-4`，AA→`S_phi` 保持连接 | 2.817 / 12.34% | 2.867 / 10.10% |
| 108696 | 旧 AA LR `1e-5`，detach AA→`S_phi` | 2.819 / 11.99% | 2.876 / 10.20% |

两者没有实质差异。需要注意，这个实验后来发现受到全局 gradient clipping 干扰：

- LR `1e-4` run 的 raw AA grad norm 约 0.678，但被整体 grad norm 38.45 缩放后仅约 0.0176。
- detach run 的 raw AA grad norm 约 0.507，clip 后约 0.0056。

因此这个对照不能证明提高 head LR 无效；它主要促成了后续“AA head 独立 clipping”的修复。

日志：

- [`aa-lr1e4-c448-108695.out`](../logs/training/stage3_binder/aa-lr1e4-c448-108695.out)
- [`aa-detach-c448-108696.out`](../logs/training/stage3_binder/aa-detach-c448-108696.out)

### 4.4 纯 PINDER low-sigma warmup

完成了三个 1000-step、AA-head-only、冻结 trunk/`S_phi` 的纯 PINDER 实验，均使用 AA LR `1e-4` 和独立 head clipping：

| Job | Sigma 策略 | Step 1000 PINDER AA CE / acc. |
| --- | --- | ---: |
| 109122 | 原始 EDM sigma 分布，uniform AA loss | 2.892 / 9.60% |
| 109123 | 每 8 个样本强制加入 0.04、0.4，并做 inverse-quadratic weighting | 2.896 / 9.50% |
| 109126 | 所有样本在 0.04、0.4 间交替 | 2.895 / 9.51% |

对应 held-out PINDER 128 complexes fixed-sigma evaluation：

| 训练策略 | Sigma | CE | Accuracy | Balanced acc. | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 0.04 | 2.888 | 9.735% | 5.527% | 0.0288 |
| Uniform | 0.4 | 2.846 | 11.321% | — | — |
| Uniform | 4.0 | 2.843 | 11.448% | — | — |
| Partial-low | 0.04 | 2.880 | 10.144% | — | — |
| Partial-low | 0.4 | 2.842 | 11.206% | — | — |
| Partial-low | 4.0 | 2.843 | 11.253% | — | — |
| All-low | 0.04 | 2.879 | 10.260% | — | — |
| All-low | 0.4 | 2.840 | 11.277% | — | — |
| All-low | 4.0 | 2.849 | 11.527% | — | — |

差异很小。结论是：**仅把监督集中到 low sigma，并冻结 trunk 只训练 AA head，不能解决问题。**

日志：

- [`aa-uniform-c448-109122.out`](../logs/training/stage3_binder/aa-uniform-c448-109122.out)
- [`aa-lowsigma-c448-109123.out`](../logs/training/stage3_binder/aa-lowsigma-c448-109123.out)
- [`aa-alllow-c448-109126.out`](../logs/training/stage3_binder/aa-alllow-c448-109126.out)
- [`eval-aa-sigma-compare-109151.out`](../logs/validation/pinder_binder_backbone_inputs/eval-aa-sigma-compare-109151.out)

### 4.5 两个 Stage III 主 checkpoint 的 PINDER fixed-sigma validation

对 job 107903 和 107904 的 step 5000 checkpoint，在同一套 PINDER validation（128 complexes、crop 448）上评估：

| Checkpoint | Sigma | CE | Accuracy | Balanced acc. | Macro-F1 | Cα RMSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 107903 step 5000 | 0.04 | 2.869 | 10.458% | 6.718% | 0.0422 | 0.124 Å |
| 107903 step 5000 | 0.4 | 2.710 | 15.837% | 10.766% | 0.0813 | 0.465 Å |
| 107903 step 5000 | 4.0 | 2.768 | 14.095% | 9.477% | 0.0710 | 1.947 Å |
| 107904 step 5000 | 0.04 | 2.869 | 10.512% | 6.663% | 0.0405 | 0.126 Å |
| 107904 step 5000 | 0.4 | 2.713 | 15.842% | 10.878% | 0.0813 | 0.463 Å |
| 107904 step 5000 | 4.0 | 2.770 | 14.079% | 9.554% | 0.0712 | 1.944 Å |

两个 checkpoint 的 AA 结果几乎相同，说明 crop 448/512 的选择没有在 step 5000 产生可见 AA 差异。中等 sigma 明显优于接近最终推理端的 sigma 0.04。

日志：

- [`ppi-binder-inputs-109153.out`](../logs/validation/pinder_binder_backbone_inputs/ppi-binder-inputs-109153.out)
- [`ppi-binder-inputs-109154.out`](../logs/validation/pinder_binder_backbone_inputs/ppi-binder-inputs-109154.out)

### 4.6 Crop-448 step 4000/6000 的完整 sigma sweep

对 job 107904 的两个 checkpoint 做七个 sigma 的 PINDER fixed-sigma sweep：

| Sigma | Step 4000 CE | Step 4000 acc. | Step 6000 CE | Step 6000 acc. |
| ---: | ---: | ---: | ---: | ---: |
| 0.04 | 2.867 | 10.704% | 2.869 | 10.591% |
| 0.10 | 2.845 | 11.651% | 2.842 | 11.602% |
| 0.20 | 2.721 | 15.524% | 2.717 | 15.652% |
| 0.40 | 2.713 | 15.869% | 2.714 | 15.821% |
| 0.80 | 2.714 | **15.912%** | 2.717 | **15.873%** |
| 1.00 | 2.717 | 15.891% | 2.720 | 15.835% |
| 4.00 | 2.770 | 14.123% | 2.770 | 14.099% |

结论：

- AA head 有明显 sigma dependence，最佳区间约为 0.4–1.0。
- sigma 0.04 比中等 sigma 低约 5.2 个百分点。
- step 4000 到 6000 几乎完全没有提升，虽然同期 backbone validation 明显改善。
- balanced accuracy 和 macro-F1 仍低，说明 15.8% 的表面 accuracy 不能掩盖类别不均衡问题。

日志：

- [`pinder-s4000-sigma-109339.out`](../logs/validation/pinder_binder_backbone_inputs/pinder-s4000-sigma-109339.out)
- [`pinder-s6000-sigma-109340.out`](../logs/validation/pinder_binder_backbone_inputs/pinder-s6000-sigma-109340.out)

### 4.7 同一 free-generation 轨迹上的 AA readout

Job 109341 在 64 个 PINDER complexes、20-step Gaussian-start co-generation 轨迹上比较三个 readout。没有把 ground-truth coordinates 输入生成器，使用 `complete_unmask`：

| Readout | 对应 sigma | CE | Accuracy | Top-5 accuracy |
| --- | ---: | ---: | ---: | ---: |
| 最终 step | 0.0345 | 2.914 | **8.262%** | **35.619%** |
| 最接近 target sigma | 0.3974 | 2.999 | 7.591% | 33.717% |
| 每 token 选择最高置信度 step | 不固定 | 3.016 | 7.809% | 33.019% |

中间 sigma 和 per-token confidence-best 都没有优于最终 readout。因此“只需在推理时读取 sigma≈0.4 的 logits”不成立。

fixed-native-backbone 的 sigma 0.4 可达到约 15.8%，而 free rollout 同一区间只有约 7.6%，提示明显的表示分布偏移。不过 free rollout 的 backbone 未必与 native binder 一致，所以这部分差距不能全部归因于分类 head。

日志：[`pinder-aa-readout-109341.out`](../logs/validation/pinder_binder_backbone_inputs/pinder-aa-readout-109341.out)

### 4.8 Tiny-overfit 容量诊断

#### Head-only，固定 32 个 PINDER 样本

Job 109152：

- 仅训练 AA head，约 1.30M trainable parameters。
- 8 个 diffusion samples 全部为 sigma 0.04。
- 无 reference-position augmentation。
- AA LR `3e-4`，独立 clipping，共 1500 steps。

| Step | PINDER train-window accuracy |
| ---: | ---: |
| 10 | 7.62% |
| 100 | 10.80% |
| 200 | 12.78% |
| 500 | 13.36% |
| 750 | 15.74% |
| 1,000 | 18.30% |
| 1,250 | 18.52% |
| 1,500 | **20.09%** |

最终 window CE 为 2.492；最后一个 batch 为 CE 2.418、accuracy 21.50%。这证明梯度和 optimizer 确实有效，但一个 shallow head 对 32 个固定样本也远未达到记忆。

日志：[`aa-overfit32-c448-109152.out`](../logs/training/stage3_binder/aa-overfit32-c448-109152.out)

#### Head + 最后一个 diffusion transformer block

Job 109342：

- 训练 AA head 和最后一个 diffusion block，共约 9.56M trainable parameters。
- sigma 在 0.04、0.4 间交替。
- head LR `3e-4`，trunk LR `1e-5`，trunk gradient scale 0.1。
- 共 1500 steps。

最终 train-window CE 为 2.632、accuracy 为 **17.22%**；最后 batch 为 CE 2.565、accuracy 18.83%。其中最后 batch 的 low-sigma accuracy 为 16.18%，mid-sigma 为 21.48%。

这个实验低于 head-only，但不能据此断言“解冻最后一层无效”，因为两个实验并不严格匹配：sigma schedule 不同，并且 trunk 的有效更新强度约比 head 小 300 倍。

日志：[`aa-lastblk32-c448-109342.out`](../logs/training/stage3_binder/aa-lastblk32-c448-109342.out)

## 5. 实验失败与工程修复

这些失败属于运行或配置问题，不是模型结果：

| Job/现象 | 原因 | 修复 |
| --- | --- | --- |
| 108607/108608 | 多个 DataLoader worker 向共享 PINDER pdb 路径写 `.tmp`，触发 `PermissionError` | 改用用户可写的 `PINDER_PDB_CACHE=/hai/scratch/shenjm/pinder/2024-02/pdbs`，并配置独立 CIF cache |
| 108897/108898 | forced-sigma 空列表进入 config manager，触发 `IndexError` | 显式区分未设置与空列表 |
| 109120/109121 | `ml_collections.ConfigDict` 把 list 字段锁定成 string 类型，覆盖时报 `TypeError` | config 中先存 CSV string，构建后再解析为 float list |
| 107381 | CUDA allocator 有大量 reserved-but-unallocated memory，出现碎片 OOM | 使用 expandable CUDA segments；107903/107904 未再因此 OOM |
| 107902 | 调用了错误的 external code path | 修正运行环境/路径 |
| `MaxSubmitJobsPerAccount` | Slurm account/QOS 的同时提交上限 | 取消不需要的 job 或等待队列名额；与训练代码无关 |

另外，PXDesign atom-attention embedder 的 API 兼容性已经在 submodule commit `2202ad0` 修复。

## 6. 当前证据支持和排除的假设

| 假设 | 当前判断 | 证据 |
| --- | --- | --- |
| 两个 checkpoint 对不上 | 基本排除 | backbone 和 design-condition embedder 逐 tensor 一致 |
| AA label 或 binder mask 错 | 未发现 | label 保存时机、selector 和严格重建检查均正确 |
| AA head 被冻结/未进入 optimizer | 排除 | 参数有约 3.63% 相对变化；tiny-overfit 可持续提升 |
| 单纯 AA LR 太低 | 不是完整解释 | 早期对照被 global clipping 混淆；独立 clipping 后 low-sigma 实验仍无改善 |
| 只因最终 sigma 训练不足 | 不充分 | all-low warmup 无显著收益；free trajectory 中 sigma 0.4 readout 也更差 |
| backbone 联合训练完全破坏 AA 表示 | 不支持 | fixed-native sigma 0.4 仍有约 15.8%，且 crop 448/512 结果一致 |
| 表示能力/训练目标与 rollout 不匹配 | 强烈怀疑 | fixed-native 与 free rollout 差距大；step 4k→6k backbone 变好而 AA 不变 |
| shallow AA head 容量不足 | 有证据但未完全证明 | 32-sample head-only 只能到约 20%；last-block 实验尚未做严格匹配 |

## 7. 下一步实验优先级

### P0：严格匹配的 last-block tiny-overfit

这是当前最关键的诊断实验。它必须与 job 109152 保持相同的 32 samples、全 sigma 0.04、无 position augmentation 和 1500 steps，只改变：

- 解冻最后一个 diffusion transformer block；
- trunk LR 提到 `1e-4`；
- trunk gradient scale 从 0.1 提到 1.0；
- head LR 保持 `3e-4`。

解释标准：

- 若 accuracy 明显超过 30% 且仍在上升，说明 trunk representation 可以被 AA supervision 改造；下一步应做 rollout-aware/on-policy AA training。
- 若仍停在约 20%，说明单个最后 block 加 shallow head 仍不足，应优先设计独立的 inverse-folding head，而不是继续堆低 sigma warmup。

截至本文整理时，这个严格匹配实验是“计划运行/等待结果”，不能与已完成的 job 109342 混为一谈。

还需要注意：当前未提交的 `submit_aa_lastblock_tiny_overfit.sh` 仍硬编码 sigma 0.04/0.4 交替和 `--trunk-grad-scale 0.1`，对应的是 job 109342 的旧配置；在提交 P0 前必须把它改成全 sigma 0.04 和 trunk gradient scale 1.0，不能只设置 `TRUNK_LR=1e-4` 就当作严格匹配实验。

### P1：1-sample memorization sanity check

如果 P0 仍然弱，把 `TINY_SAMPLES` 降到 1。若单样本都不能接近完全记忆，应继续查表示、重复采样是否真正固定、loss normalization 和 optimizer update；若单样本可记忆而 32 样本不行，则更支持容量或表示可分性不足。

### P2：Rollout-aware AA supervision

在训练中周期性使用模型自己 rollout 得到的中间/后期结构状态，再对这些 on-policy states 计算 AA CE。目标是缩小 noisy-native 与 Gaussian-start rollout 之间的状态分布差距。该实验应先从短程、冻结大部分 backbone 的版本开始，控制显存和训练不稳定性。

### P3：更强的 inverse-folding head

如果 tiny-overfit 表明浅层 readout 是瓶颈，应增加显式几何/邻域建模，例如 residue-level graph/message passing 或小型 sequence transformer，并直接以最终/中间 backbone 几何为条件预测 AA。ProteinMPNN 在已有 monomer benchmark 上，GT-backbone recovery 为 46.25%，predicted-backbone recovery 为 33.38%，远高于当前 donor 的约 13%，说明专门的 inverse-folding architecture 值得作为基线；但该数字来自 monomer，不可直接当成 PINDER binder 对照。

### P4：类别不均衡处理与诊断

继续报告 per-class recall、balanced accuracy、macro-F1 和预测频率。可以在保证固定评估协议的前提下比较 class-weighted CE、label smoothing 或 balanced sampler，但它们应排在 representation/rollout mismatch 诊断之后。

## 8. 已完成的代码改动与 Git 状态

已提交到 `sjm/binder-design-training` 的主要 commit：

- `148eb18`：增加 source-specific AA metrics、AA head optimizer group/LR、gradient/update norm、PINDER evaluation、用户可写 cache 和 Stage III 对照提交脚本。
- `2b2c5e2`：增加 AA head 独立 gradient clipping、forced-sigma sampler、sigma weighting/per-sigma diagnostics、PINDER fixed-sigma comparison 和 low-sigma/tiny-overfit 脚本。
- `f1560e1`：更新 PXDesign embedder compatibility；对应 PXDesign submodule commit 为 `2202ad0`。

截至整理时仍有未提交的实验性改动：

- `pxdesign_train/cogenerate.py`
- `scripts/training/train_protenix_monomer.py`
- `tests/test_complete_unmask.py`
- `scripts/evaluation/infer_aa_readouts_pinder.py`
- `scripts/evaluation/slurm_infer_pinder_aa_readouts.sh`
- `scripts/evaluation/submit_stage3_pinder_sigma_sweep.sh`
- `scripts/training/submit_aa_lastblock_tiny_overfit.sh`

这些改动包含三个同轨迹 AA readout、`--unfreeze-last-diffusion-blocks`、PINDER sigma sweep/trajectory evaluation 脚本和相应测试。已有测试记录为 **59 passed**，PINDER dry-run 通过，并确认配置可以只解冻 AA head 与最后一个 diffusion block。工作区中的 `tmp.sh` 未纳入本次实验改动。

## 9. 最终判断

Stage III mixed-data 训练对 crop-448 的 backbone 学习是有效的，而且没有损坏 monomer fold prior；真正没有随训练改善的是 AA prediction。现有证据表明 AA 通路能学到少量结构信号，但 donor 起点已经较弱、类别预测塌缩明显，并且 one-step noisy-native training 与 free rollout 之间存在较大的表示分布差异。low-sigma warmup、提高 head LR、detach 和简单更换 readout 都没有解决问题。

最合理的下一步不是继续盲目延长主训练，而是先用严格匹配的 last-block tiny-overfit 判断 representation 是否可被局部调整；随后根据结果选择 rollout-aware supervision 或更强的 inverse-folding head。
