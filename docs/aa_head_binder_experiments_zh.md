# Stage III Binder 联合训练与 AA Head 实验总结

> 整理日期：2026-09-05
>
> 分支：`sjm/binder-design-training`
>
> 范围：从 Stage III binder training 启动开始的训练、失败任务、AA head 代码审计、PINDER validation、sigma/优化器/解冻深度对照、free-generation 采样与 tiny-overfit 诊断，以及 2026 年 9 月会议提出的后续问题。

## 1. 核心结论

目前的证据不支持“AA label 接错”“binder/receptor mask 反了”“AA head 被冻结”或“两个 checkpoint 嫁接不兼容”是主因。AA head 的参数确实更新，监督对象也是 binder native residue type。

最关键的新结果来自固定 32 个 PINDER 样本的 clean-backbone tiny-overfit：只训练 AA head 最终为 **22.81%**，训练 head 加最后一个 diffusion block 为 **20.57%**；而训练 head 加全部 16 个 diffusion blocks 后，PINDER train-window accuracy 达到 **99.95%**、CE 降到 **0.0051**。这证明整套模型有能力记住 backbone-conditioned AA 任务，也排除了 loss、label、mask、梯度或 optimizer 完全断开的假设。

这项结果同时把问题进一步定位为：

1. 原始冻结 diffusion representation 中的 AA identity 对浅层 head 不够可分；只允许最后一个 block 调整仍不够，深层 representation adaptation 才能完成训练集记忆。
2. 99.95% 只是在 32 个训练样本上的 memorization，尚未证明模型真正使用 backbone geometry，也尚未证明能泛化到 held-out PINDER。
3. 主 Stage III 使用 one-step noisy-native supervision，而真实推理从 Gaussian noise 开始多步 rollout，train–inference state distribution shift 仍然存在。
4. 所有 binder residue 同时被 mask，预测 AA 不反馈给后续 diffusion step；这是比逐步 unmask 更困难的一步到位任务。
5. AA 类别分布仍有塌缩迹象，balanced accuracy 和 macro-F1 明显低于表面 accuracy。
6. 原生 PXDesign sampler 的第一次 validation 因兼容层漏传 `pair_z`、`p_lm`、`c_l` 而对 8/8 样本全部失败，所以目前仍没有有效的原生 sampler free-generation 指标。

因此，当前不能说“NN 结构本身学不到 AA”。更准确的结论是：**网络容量足以记忆任务，但 frozen/shallow readout、训练状态分布与 free rollout 不匹配、以及 all-at-once unmask 共同限制了主训练的泛化和生成端表现**。下一步应先证明 all-16 模型在 held-out 数据上是否有效、用 shuffled-backbone 验证其是否真的依赖几何，再决定解冻深度、rollout-aware training 和 iterative unmasking 的方案。

### 1.1 Job 总览与有效性

下表按实验演进顺序记录从 Stage III binder training 开始出现的主要 job。`无效` 表示任务因工程错误退出，其数值不能作为模型结论；`被替代` 表示后来发现评估实现有误，应该使用修正后的 job。

| Job | 实验 | 状态 | 主要结果或用途 |
| ---: | --- | --- | --- |
| 105659 | 最早 binder launch | 无效 | Slurm 环境找不到 `conda` |
| 107180、107296 | binder smoke tests | 有效 smoke | 6 steps 通过，确认数据、checkpoint 和前后向可运行 |
| 107381 | crop-512 正式尝试 | 无效 | step 50 左右因 CUDA allocator 碎片 OOM |
| 107902 | 修复 OOM 后重提 | 无效 | external Protenix 路径配置错误 |
| 107903 | Stage III binder，crop 512 | 有效 | 到约 step 5800，walltime 结束；后期 monomer geometry 不如 crop 448 稳定 |
| 107904 | Stage III binder，crop 448 | 有效 | 到 step 6200；step 6000 backbone validation 最好，AA 仍约 13% |
| 108572 | 最早 PINDER eval | 无效 | evaluation namespace 缺 `resume_lr` |
| 108607、108608 | LR/detach 首次提交 | 无效 | DataLoader 向共享 PINDER cache 写临时文件，触发权限错误 |
| 108695、108696 | AA LR 与 detach 对照 | 有效 | 1000 steps 内几乎无差异；暴露全局 gradient clipping 干扰 |
| 108897、108898 | low-sigma 首次提交 | 无效 | 空 `forced_sigmas` 触发 `ConfigDict`/`IndexError` |
| 109120、109121 | low-sigma 第二次提交 | 无效 | list/string 类型锁定导致 `TypeError` |
| 109122、109123、109126 | uniform / partial-low / all-low | 有效 | 仅训练 head 时均约 9.5–9.6%，low sigma 没解决问题 |
| 109151 | 三个 low-sigma 模型的 held-out eval | 有效 | 三者在 sigma 0.04/0.4/4.0 的差异都很小 |
| 109152 | 32-sample、head-only tiny-overfit | 有效 | 1500 steps，PINDER window acc 20.09% |
| 109153、109154 | 107903/107904 step5000 PINDER fixed-sigma | 有效 | 两 checkpoint 几乎相同；sigma 0.4 最好，约 15.84% |
| 109339、109340 | 107904 step4000/6000 sigma sweep | 有效 | 最佳 sigma 0.4–1.0；4k 到 6k 无 AA 提升 |
| 109341 | 20-step free-generation AA trajectory readout | 有效 | final 8.28%；sigma≈0.4 与 confidence-best 均更差 |
| 109342 | 早期 head + last block tiny-overfit | 有效但非严格对照 | acc 17.22%；sigma 与梯度强度和 head-only 不匹配 |
| 109451 | 严格 low-sigma head + last block | 有效 | 1500 steps，PINDER window acc 20.57%，仍未完全记忆 |
| 110407 | 首次带结构指标的 free-generation eval | 被替代 | Kabsch 实现方向错误，RMSD 数值不应继续引用 |
| 110421 | clean backbone + head-only，32 samples | 有效 | 3000 steps，PINDER window acc 22.81% |
| 110423 | Kabsch 修正后，20-step free generation | 有效 | 64 samples；Cα RMSD 18.884 Å，lDDT 0.2377，TM 0.1096 |
| 110424 | Kabsch 修正后，400-step free generation | 有效但小样本 | 8 samples；Cα RMSD 19.518 Å，lDDT 0.2086，TM 0.1154 |
| 110540 | clean backbone + all 16 blocks，32 samples | 有效 | 3000 steps，PINDER CE 0.0051、acc 99.95% |
| 110546 | PXDesign native sampler，400 steps | 无效 | 8/8 样本因缺少三个 forward 参数失败，没有 summary |

## 2. 不同评估的含义

几类指标不能混在一起解释：

| 评估 | 输入结构 | 能回答的问题 | 不能直接回答的问题 |
| --- | --- | --- | --- |
| 训练日志中的 `val_*` | monomer validation | Stage III 是否保留/改善 monomer fold prior | binder 生成质量、PINDER AA recovery |
| PINDER fixed-sigma | native binder backbone 加指定噪声 | 给定接近 native 的结构表示时，AA head 能否识别 native AA | free generation 是否能生成 native binder |
| PINDER free co-generation readout | 从 Gaussian noise 多步生成 | 实际 rollout 上 AA readout 的行为 | AA head 的纯分类上限；这里还混入了生成 backbone 偏离 native 的影响 |
| Tiny-overfit | 重复固定的少量训练样本 | 参数、梯度和模型容量是否至少能记忆训练集 | 泛化能力和真实 binder 设计能力 |

尤其要注意：PINDER free-generation 的 native sequence recovery 同时受 backbone 是否接近 native 构象影响。一个生成出来但不同于 native 的合理 binder，未必应该恢复出 native sequence，所以该指标不能单独等价为 AA head accuracy。

### 2.1 总体研究问题

Stage III binder 实验围绕四个逐层收窄的问题展开：

1. **工程与监督是否接通？** 检查 checkpoint、label、binder mask、梯度、optimizer 和数据 cache，确认 AA loss 是否真正作用于目标 residue。
2. **给定接近 native 的 backbone，AA head 能否预测 sequence？** 用 fixed-sigma 和 clean-coordinate 实验隔离 sequence readout，不让 free-generation backbone 质量混入结论。
3. **模型是否具备学习容量，瓶颈位于 head 还是 representation？** 用固定 32 个样本的 tiny-overfit，以及 head-only、last-block、all-16-block 解冻深度对照判断。
4. **训练能力能否迁移到真实生成？** 用 Gaussian-start rollout、不同轨迹 readout 和 PXDesign native sampler 评估 train–inference state distribution shift 与采样器影响。

这一顺序很重要：如果 tiny-overfit 都不能成功，应先查接线或容量；只有 tiny-overfit 成功后，held-out 与 free rollout 的差距才可以解释为泛化、表示分布或 decoding strategy 问题。

### 2.2 公共模型、数据和训练 protocol

除非单项实验明确说明，Stage III binder 实验共享以下设置：

| 项目 | 设置 | 目的 |
| --- | --- | --- |
| Backbone + side-chain warm start | Stage II `step52500.pt` | 保留已经训练的 backbone 与 `S_phi` |
| AA head donor | `aa_head_on_stage2/step9000.pt` | 避免使用 Stage II 中 chance-level 的随机 AA head |
| Binder 数据 | PINDER chain B；Protenix PPI 第二条 chain | 统一 binder selector 语义 |
| 主训练数据 | monomer + Protenix PPI + PINDER curriculum | 在学习 complex/binder 的同时维持 monomer fold prior |
| AA target | 加噪前保存的 native residue type | 防止把被 mask 或扰动后的 token 当 label |
| 默认 sequence corruption | binder 全部 mask，`mask_mode=all` | 与当前 one-shot `complete_unmask` 推理语义一致 |
| 默认 crop | 448 或 512 tokens；binder 完整保留 | 比较吞吐、显存与包含 interface 的能力 |
| 主训练优化 | backbone 与 side chain alternating；AA CE 进入 backbone loss | 联合优化 backbone、side chain 与 residue type |
| Fixed-sigma eval | native binder backbone 加指定 sigma 噪声 | 测量给定接近 native geometry 时的 AA readout 上限 |
| Free-generation eval | binder 坐标从 Gaussian noise 开始多步 rollout | 测量实际联合生成轨迹，而非 teacher-forced 分类 |
| 结构指标 | 每个 complex 对 binder 做 Kabsch 后计算 Cα/BB RMSD、lDDT、TM | 去除全局刚体平移和旋转；不包含 receptor-relative pose |

主训练 warm start 使用两个 checkpoint，但已经逐 tensor 检查二者共同部分：`diffusion_module` 636/636、`design_condition_embedder` 96/96 完全一致。因此两个 checkpoint 的组合不是后续 AA 停滞的解释。

### 2.3 实验设计总表：目的、方法、控制变量与判据

| 实验 | 目的/假设 | Method 与关键控制 | 预先判据 | 实际结果 |
| --- | --- | --- | --- | --- |
| AA donor strict evaluation | 建立 Stage III 起点，并判断 donor 是否使用结构而非只学类别先验 | 同一 donor、同一数据；比较 sigma 0.04/0.4/4.0，并在 sigma 0.4 随机化结构输入 | Native 明显优于 randomized，表示存在结构信号 | Native sigma 0.4 为 13.40%，randomized 为 8.33%；有弱结构信号但类别塌缩 |
| Main Stage III crop 448/512 | 检验 mixed binder training 是否能维持 backbone prior，并选择可稳定运行的 crop | 同一 warm start、loss 和 curriculum；主要改变 crop size；每 2000 steps 做 monomer validation | Backbone 指标持续改善且 AA 不退化；比较吞吐/OOM/稳定性 | Crop 448 到 step6000 持续改善；crop512 后期回退；两者 AA 都约 13% |
| AA LR vs detach | 检验 AA 不动是否来自 head LR 太低，或 AA→`S_phi` 耦合干扰 | 两个 1000-step mixed-data run；比较 head LR `1e-4` 与 detach 配置 | 若优化是主因，应看到明显 CE/acc 分离 | 几乎无差异；后发现全局 clipping 混淆，促成独立 head clipping |
| Uniform/partial-low/all-low | 检验 AA 几乎没在最终低 sigma 上训练是否为主因 | 纯 PINDER、head-only、相同 LR/clipping；只改变 sigma 采样与 weighting | Low-sigma arm 应显著优于 uniform，尤其在 sigma 0.04 eval | 三个 arm 均约 9.5–9.6%；held-out 差异很小，假设不充分 |
| 107903 vs 107904 fixed-sigma | 判断 crop 448/512 的 Stage III checkpoint 在 binder 上是否不同 | 同一 128 个 PINDER validation complexes、crop448、sigma 0.04/0.4/4.0 | 同 protocol 下直接比较 AA CE/acc 与类别指标 | 两者几乎完全相同；sigma 0.4 最好，约 15.84% |
| Step4000 vs step6000 sigma sweep | 判断 AA 是否随 Stage III 继续训练而改善，并定位最佳 sigma | 同一 crop448 run 的两个 checkpoint；七个 sigma；相同 PINDER protocol | 若 AA 随训练改善，6k 曲线应整体优于 4k | 曲线基本重合；最佳 sigma 约 0.4–1.0，主训练 AA 没继续改善 |
| Free trajectory readout | 判断 fixed-native 的中 sigma 优势能否通过改变读取时机恢复 | 64 个 PINDER complexes、Gaussian-start、20-step rollout；比较 final、sigma≈0.4、per-token confidence-best | 若只是 readout timing 错，非 final 策略应优于 final | Final 8.28% 反而最好；简单换 readout 无效 |
| Head-only tiny-overfit | 检查 label/loss/optimizer 是否接通，以及 shallow head 能否记忆小数据 | 固定重复 32 个 PINDER 样本；全 sigma 0.04；只训练 AA head | 完全接通且容量充足时应接近 100% train accuracy | 1500 steps 仅 20.09%；能学习但不能完全记忆 |
| Last-block tiny-overfit | 判断允许局部 representation adaptation 是否足够 | 与 head-only 匹配的 32 samples/低 sigma；训练 head + block15，head LR `3e-4`、block LR `1e-4` | 明显超过 head-only 表示最后一层是关键瓶颈 | 20.57%，仍与 head-only 同量级；单个末层不够 |
| Clean backbone + head-only | 隔离坐标噪声，测试 clean geometry 在 frozen representation 下是否容易读取 | 固定 32 samples；coordinates 不加噪；保留 sigma=0.04 conditioning；只训练 head | 若噪声是主因，应比普通 low-sigma 大幅提升并能记忆 | 3000 steps 22.81%；去噪仍不能让 shallow head 记忆 |
| Clean backbone + all 16 blocks | 判断整个 diffusion representation 允许适配后是否具备 AA 学习容量 | 与 clean head-only 相同数据/步数/loss；唯一核心变化是解冻全部16个 blocks | 接近 100% 表示容量与训练接线无根本问题 | CE 0.0051、acc 99.95%；训练集容量问题被排除 |
| Minimal-sampler structure eval | 测量实际 free-generated backbone 与 PINDER native 的差距 | Gaussian-start；Kabsch 修正后分别用20 steps/64 samples和400 steps/8 samples | RMSD/lDDT/TM 给出生成结构质量；不同样本集不得直接比较步数 | 两者均约19 Å RMSD、TM约0.11，当前 minimal sampler 表现差 |
| PXDesign native sampler | 检验极高 RMSD 是否主要由自写 minimal Euler sampler 导致 | 计划在相同 checkpoint/样本上调用 PXDesign `sample_diffusion`，400 steps | 必须完成全部样本并生成 summary 后才可比较 | Job110546 接口报错，0 个有效样本；假设仍未检验 |

### 2.4 结果解释与成功标准

- **Tiny-overfit 成功不等于泛化成功。** 训练 accuracy >95% 只说明容量和优化链路存在；还必须在未参与训练的 PINDER complexes 上提高，才能支持可泛化 inverse folding。
- **证明模型使用 backbone 需要因果 control。** 同一样本上 shuffled/null backbone 应显著降低 accuracy；否则模型可能用 sample identity、target context 或类别先验记忆。
- **比较 sampler 必须成对。** checkpoint、样本 ID、初始 noise、crop、步数和 metric implementation 应固定；110423 与 110424 样本数不同，只能用于量级判断。
- **比较 loss 必须固定 architecture。** class weighting、focal loss 或 label smoothing 不能与解冻深度、sigma schedule、unmask strategy 同时改变。
- **AA accuracy 不能单独使用。** 必须同时报告 CE、balanced accuracy、macro-F1、per-class recall、top-5 accuracy、prediction histogram 和 calibration。
- **Free-generation native recovery 不是唯一设计指标。** 当生成 backbone 与 native 不同但仍合理时，native sequence 未必是唯一正确答案；后续应补充 sequence plausibility、structure self-consistency 和 interface quality。

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

### 4.9 严格匹配的最后一层 low-sigma tiny-overfit

Job 109451 修正了 109342 中不匹配的变量：固定同一批 32 个 PINDER 样本、8 个 diffusion samples 全部为 sigma 0.04、head LR `3e-4`、最后一个 diffusion block LR `1e-4`，不再把 trunk gradient 缩小到 0.1。

| Step | PINDER train-window CE | PINDER train-window accuracy |
| ---: | ---: | ---: |
| 10 | 2.915 | 7.63% |
| 100 | 2.841 | 10.95% |
| 200 | 2.788 | 12.60% |
| 500 | 2.750 | 13.40% |
| 750 | 2.680 | 15.56% |
| 1,000 | 2.559 | 18.47% |
| 1,250 | 2.555 | 18.54% |
| 1,500 | **2.478** | **20.57%** |

最后一层确实更新，但最终只与 head-only 的约 20% 同量级，不能完成 32-sample memorization。这说明问题不是简单地“最后一层没允许动”，而是最后一层提供的适配深度仍不够。

日志：[`aa-lastblk32-low-c448-109451.out`](../logs/training/stage3_binder/aa-lastblk32-low-c448-109451.out)

### 4.10 Clean-backbone 容量对照：head-only 与 all-16 blocks

为了回答“只给 backbone、接近 sigma=0 时能否学 AA type”，增加了不对 native coordinates 加噪声、但保留 sigma=0.04 conditioning 的 clean-coordinate 模式。没有直接使用数学上的 sigma=0，是为了避免 EDM preconditioning/噪声嵌入在训练分布外出现数值或语义歧义；这项实验等价地隔离了 clean backbone 是否足够提供 AA 学习信号。

两个实验使用完全相同的 32 个 PINDER 样本、crop 448、3000 steps 和 AA-only loss：

| 配置 | Trainable params | Step 1000 acc | Step 1500 acc | Step 2000 acc | Step 3000 CE / acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Head-only，Job 110421 | 1.30M | — | — | — | 2.384 / 22.81% |
| Head + all 16 diffusion blocks，Job 110540 | 133.53M | 51.33% | 80.03% | 97.21% | **0.0051 / 99.95%** |

All-16 的完整关键轨迹如下：

| Step | PINDER train-window CE | PINDER train-window accuracy |
| ---: | ---: | ---: |
| 100 | 2.814 | 11.56% |
| 500 | 2.319 | 25.61% |
| 750 | 1.916 | 36.36% |
| 1,000 | 1.477 | 51.33% |
| 1,250 | 1.125 | 63.26% |
| 1,500 | 0.646 | 80.03% |
| 2,000 | 0.129 | 97.21% |
| 2,500 | 0.0288 | 99.54% |
| 3,000 | **0.0051** | **99.95%** |

结论分两层：

- 强结论：模型、AA loss、label、mask 和 optimizer 的组合具备完成训练集记忆的容量；“整个 NN 结构根本学不到 AA”被排除。
- 尚不能下的结论：该模型已经学到了可泛化的 inverse folding。因为这里只有 32 个重复样本，还需要 held-out validation 和 shuffled-backbone/null-coordinate control 排除按样本身份或非几何上下文记忆。

训练日志：

- [`aa-clean-head-32-c448-110421.out`](../logs/training/stage3_binder/aa-clean-head-32-c448-110421.out)
- [`aa-clean-all16-32-c448-110540.out`](../logs/training/stage3_binder/aa-clean-all16-32-c448-110540.out)

All-16 checkpoint：

```text
/hai/scratch/shenjm/proteo_aa_runs/aa_clean_backbone_tiny_overfit/stage3_binder_coevolution/110540/checkpoints/step3000.pt
```

训练日志中的 Cα RMSD 约 0.10–0.16 Å 是 clean-input 单步诊断，不是 free-generation 结构质量；这里的 `lddt` 也是训练 loss 字段，不能当作生成 Cα lDDT。

### 4.11 Kabsch 修正后的 free-generation 结构指标

Job 110407 首次为 Job 109341 的 Gaussian-start 轨迹加入 Cα RMSD/lDDT/TM-score，但随后发现 Kabsch 旋转方向实现错误，因此其中约 22.9 Å 的 RMSD 已被修正后的 jobs 取代。

修正结果：

| Job | Sampler | Samples | Cα RMSD | BB RMSD | Cα lDDT | TM-score | Final AA acc |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 110423 | minimal Euler，20 steps | 64 | 18.884 Å | 18.825 Å | 0.2377 | 0.1096 | 8.28% |
| 110424 | minimal Euler，400 steps | 8 | 19.518 Å | 19.461 Å | 0.2086 | 0.1154 | 9.46% |

这两个 job 的样本数和 retry 后实际样本集合不同，因此不能严格断言 400 steps 比 20 steps 更差；但它们共同说明当前 minimal Euler sampler 下的 free-generated binder 与 PINDER native backbone 相差很远。此前训练日志中约 3.4 Å 的 monomer validation RMSD 与这里也不能直接比较：数据集、任务、起点和评估对象均不同。

日志：

- [`pinder-kfix-n20-110423.out`](../logs/validation/pinder_binder_backbone_inputs/pinder-kfix-n20-110423.out)
- [`pinder-kfix-n400-110424.out`](../logs/validation/pinder_binder_backbone_inputs/pinder-kfix-n400-110424.out)

### 4.12 原生 PXDesign sampler validation：当前无有效结果

Job 110546 使用 Job 107904 step5000、8 个 PINDER 样本、400 steps，尝试恢复 PXDesign 的原生 `sample_diffusion`。8/8 样本全部在 wrapper 调用 Protenix diffusion module 时失败：

```text
TypeError: DiffusionModule.forward() missing 3 required positional arguments:
'pair_z', 'p_lm', and 'c_l'
ERROR: no PINDER inference samples completed
```

这是 native sampler compatibility wrapper 的接口错误，不是 checkpoint 性能。该 job 没有生成 `inference_summary.json`，所以不得引用任何结构或 AA 指标。另需注意，它评估的是旧主模型 `107904/step5000.pt`，并不是新完成的 all-16 tiny-overfit checkpoint。

日志：[`pinder-native-n400-110546.err`](../logs/validation/pinder_binder_backbone_inputs/pinder-native-n400-110546.err)

## 5. 实验失败与工程修复

这些失败属于运行或配置问题，不是模型结果：

| Job/现象 | 原因 | 修复 |
| --- | --- | --- |
| 105659 | Slurm 非交互环境找不到 `conda` | 改为显式 source conda 环境初始化脚本 |
| 108572 | 评估脚本构造的 argparse namespace 缺少 `resume_lr` | 让 evaluation config 与 training config 的必需字段同步 |
| 108607/108608 | 多个 DataLoader worker 向共享 PINDER pdb 路径写 `.tmp`，触发 `PermissionError` | 改用用户可写的 `PINDER_PDB_CACHE=/hai/scratch/shenjm/pinder/2024-02/pdbs`，并配置独立 CIF cache |
| 108897/108898 | forced-sigma 空列表进入 config manager，触发 `IndexError` | 显式区分未设置与空列表 |
| 109120/109121 | `ml_collections.ConfigDict` 把 list 字段锁定成 string 类型，覆盖时报 `TypeError` | config 中先存 CSV string，构建后再解析为 float list |
| 107381 | CUDA allocator 有大量 reserved-but-unallocated memory，出现碎片 OOM | 使用 expandable CUDA segments；107903/107904 未再因此 OOM |
| 107902 | 调用了错误的 external code path | 修正运行环境/路径 |
| 110407 | Kabsch 旋转方向写反，free-generation RMSD 偏高 | 修正实现并以 110423/110424 结果替代 |
| 110546 | native sampler wrapper 未提供 `pair_z`、`p_lm`、`c_l` | 待在 wrapper 中补默认值并增加能覆盖真实 forward 签名的测试 |
| `MaxSubmitJobsPerAccount` | Slurm account/QOS 的同时提交上限 | 取消不需要的 job 或等待队列名额；与训练代码无关 |

另外，PXDesign atom-attention embedder 的 API 兼容性已经在 submodule commit `2202ad0` 修复。

## 6. 当前证据支持和排除的假设

| 假设 | 当前判断 | 证据 |
| --- | --- | --- |
| 两个 checkpoint 对不上 | 基本排除 | backbone 和 design-condition embedder 逐 tensor 一致 |
| AA label 或 binder mask 错 | 未发现 | label 保存时机、selector 和严格重建检查均正确 |
| AA head 被冻结/未进入 optimizer | 排除 | 参数有约 3.63% 相对变化；all-16 tiny-overfit 达到 99.95% |
| 单纯 AA LR 太低 | 不是完整解释 | 早期对照被 global clipping 混淆；独立 clipping 后 low-sigma 实验仍无改善 |
| 只因最终 sigma 训练不足 | 不充分 | all-low warmup 无显著收益；free trajectory 中 sigma 0.4 readout 也更差 |
| backbone 联合训练完全破坏 AA 表示 | 不支持 | fixed-native sigma 0.4 仍有约 15.8%，且 crop 448/512 结果一致 |
| frozen representation 不足 | 强支持 | head-only 22.81%，严格 last-block 20.57%，all-16 99.95% |
| 整个网络容量不足/NN 结构绝对学不到 | 排除训练集层面的说法 | all-16 已能完全记住 32 个 PINDER 样本 |
| all-16 已学会可泛化 inverse folding | 未知 | 还没有用 110540 step3000 做 held-out 或 shuffled-backbone control |
| train–rollout distribution shift | 强烈怀疑 | fixed-native sigma 0.4 为 15.8%，Gaussian-start rollout 仅约 7.6–8.3% |
| 一步到位 all-unmask 过难 | 合理但未做因果验证 | 当前没有序列反馈；confidence-best 只改变 readout，没有真正逐步 unmask |
| 换 AA loss 就能解决主问题 | 暂不支持作为首要解释 | 标准 CE 在 all-16 overfit 中可降到 0.005；loss 本身并未阻止记忆 |
| 类别分布影响 AA accuracy | 支持 | donor 预测集中在少数高频 AA，balanced accuracy/macro-F1 显著低于 raw accuracy |
| free-generation Cα RMSD 已可靠评估 | 部分完成 | minimal sampler 的 Kabsch 指标已修正；原生 sampler 仍无有效结果 |

## 7. 上次会议讨论事项：现状与实验化定义

### 7.1 去掉 length 限制

当前并不只有一个“length 限制”：

- index 过滤默认 `COMPLEX_MAX_N_TOKEN=1536`；
- 实际模型输入仍受 `CROP_SIZE=448/512` 限制；
- binder 必须完整保留，且默认不能超过 crop 的 75%；
- 总 complex 超过 crop 时只保留 binder 和距 binder 最近的 receptor tokens。

因此只去掉 index 过滤并不等价于模型看见完整长复合物，反而可能让更多样本被 crop/retry。建议把会议项定义为一个独立数据实验：取消或显著提高 `complex-max-n-token`，保留 binder-complete crop，并按 binder length/complex length 分桶报告接受率、OOM、吞吐和 AA recovery。若目标是完整长复合物，则需要 dynamic batching、gradient accumulation 或更大 crop，而不是只删一行过滤。

状态：**未完成**。

### 7.2 “跃迁”的问题

这里建议把“跃迁”明确成两个可测量现象：

1. diffusion trajectory 中相邻 sigma 的 logits 是否突然改变，而不是平滑演化；
2. sequence 从 all-mask 一步变成完整序列时，是否因为缺少中间条件而发生困难的离散跃迁。

Job 109341 已记录 final、target-sigma 和 confidence-best 三种 readout，但没有完整量化每一步的 logits 动态。下一次评估应保存每一步的 entropy、top-1 margin、相邻步 KL divergence、top-1 flip rate、预测氨基酸频率和 native accuracy，并按 sigma 作图。这样可以区分“结构轨迹本身坏掉”和“AA logits 在末端突然塌缩”。

状态：**部分完成，缺完整 trajectory distribution 分析**。

### 7.3 允许 parameter 更新

已经完成从 frozen head-only 到 last-block，再到 all-16 blocks 的容量实验：

- head-only：22.81%；
- head + last block：20.57%；
- head + all 16 blocks：99.95%。

这个方向已经得到决定性结果：深层参数必须允许适配，至少“只动最后一层”不够。下一步不是重复 all-16，而是用完全相同设置做 4/8/12-block depth sweep，并在 held-out PINDER 上比较，以找到最小有效解冻深度并控制 backbone prior 被破坏的风险。

状态：**容量 sanity check 已完成，最小解冻深度未完成**。

### 7.4 只给 backbone、sigma 接近 0 能否学 AA type

Job 110421/110540 已通过 clean-coordinate 模式回答大部分问题：输入 native clean backbone，不添加 coordinate noise，但保留 sigma=0.04 conditioning。

- 只训练 shallow head：只能到 22.81%；
- 允许全部 diffusion blocks 适配：达到 99.95%。

所以 clean backbone 中存在足够信息，但 frozen representation 没有把它编码成容易被当前 head 读取的形式。严格数学意义的 sigma=0 尚未运行；考虑到 EDM preconditioning 通常不把 0 当作普通训练 sigma，当前 clean-coordinate + sigma=0.04 是更安全、也更能回答科学问题的对照。

状态：**训练集容量问题已回答；泛化未回答**。

### 7.5 Sanity check：backbone 是否只在最后一步起作用

目前还没有完成这个因果实验。现有 trajectory readout 只是在不同 step 读取 logits，并没有控制“什么时候把 backbone 信息提供给 AA 模块”。

建议做三个严格对照：

- `backbone_every_step`：每一步都使用当前 backbone；
- `backbone_final_only`：此前步骤给 null/shuffled backbone，只在最后一步给真实 backbone；
- `backbone_never`：所有步骤都给 null/shuffled backbone。

同时固定 noise、样本和 unmask schedule。如果 `final_only ≈ every_step`，说明序列通路几乎只利用最终 backbone；如果 `every_step` 更好，则说明应让中间 AA prediction 反馈到后续 diffusion。

状态：**未完成**。

### 7.6 修改 AA CE 或使用其他 loss

已做的 loss 变化主要是 sigma weighting，并未改善 frozen-head 结果；尚未系统比较类别相关 loss。All-16 用普通 CE 可以完全记忆训练集，因此“CE 数学形式导致完全学不到”已被排除，但类别塌缩和泛化仍可能受 loss 影响。

建议在固定 architecture、固定 sampler 和固定 held-out set 下依次比较：

1. plain CE 基线；
2. inverse-sqrt-frequency weighted CE；
3. logit-adjusted CE；
4. focal loss；
5. 小幅 label smoothing。

选择指标不能只看 accuracy，还要看 CE、balanced accuracy、macro-F1、per-class recall、top-5 accuracy 和预测分布。不要同时改 loss、解冻深度和 sampler，否则无法归因。

状态：**sigma weighting 已做；类别 loss sweep 未完成**。

### 7.7 Residue type 的训练与预测分布

已知 donor 存在明显塌缩：预测主要集中在 LEU、VAL、GLY、GLU、ALA，多个低频 residue recall 为 0。Stage III fixed-sigma 评估也显示 balanced accuracy/macro-F1 明显低于 raw accuracy。

仍需统一输出四组 histogram：训练 label、validation label、模型 top-1 prediction、模型 softmax probability mass，并按 source、sigma、trajectory step、binder length 和 interface/core residue 分层。若问题主要来自高频 AA，可以比较 balanced sampling、inverse-sqrt class weight 或 logit adjustment；激进 inverse-frequency weighting 容易让稀有类过补偿，不建议作为第一版。

状态：**已有 donor/per-class 证据，尚未覆盖最新 all-16 与完整 trajectory**。

### 7.8 PLM 如何学习 sequence probability

这部分尚未在当前代码中实验。可借鉴 masked language modeling / discrete denoising 的核心思想：模型学习的是条件分布，而不是一次性回归最终类别。对本项目最相关的用法有三种：

- 用冻结 PLM 给 binder sequence 提供先验 logits，与结构 AA logits 做可学习融合；
- 用 PLM teacher distribution 做 KL/distillation，而不是只对 one-hot native label 做 CE；
- 把当前 all-mask objective 改成多 mask-ratio 的 masked-token denoising，让模型训练时看到部分已知 sequence context。

必须避免把 native binder sequence 作为推理不可得输入泄漏给模型。PLM 应作为 prior、teacher 或已生成 token 的编码器，而不能直接读取完整答案。第一步应先建立 frozen-PLM prior baseline，比较 sequence NLL/perplexity 和结构条件加入后的增益。

状态：**未开始**。

### 7.9 按顺序 unmask，并从最高 confidence 开始

Job 109341 的 `confidence_best` 只是从各 diffusion step 选择一次最高置信度 readout，没有把选出的 residue 固定并反馈给后续网络，因此它不是完整的 confidence-first iterative unmask。它低于 final readout，不能否定真正的 iterative decoding。

建议采用 MaskGIT 风格实验：

1. 初始全部 binder residues masked；
2. 每轮按 confidence 固定一部分 residue；
3. 下一轮把已固定 AA 重新输入 trunk；
4. 低置信度 residue 保持 masked，必要时允许 remask；
5. 比较 linear/cosine unmask fraction schedule，以及 confidence、随机、N→C、interface-first 四种 order。

每轮记录 logits entropy、confidence calibration、AA frequency 和 token flip rate。这个实验需要训练时也包含 partial mask，否则推理会再次落入新的 distribution shift。

状态：**readout 版已做，真正的反馈式 iterative unmask 未完成**。

### 7.10 一步到位可能太难

当前证据支持这一担忧：训练和推理的 binder residue 全部同时 masked，`aa_t=1`，已预测 sequence 不会反馈给 trunk；free rollout 最终 accuracy 仅约 8%。但 clean all-16 能记住训练样本，说明一步到位不是绝对不可学习，而更可能是数据效率和泛化很差。

实验上应比较：

- all-mask one-shot；
- 随机 partial-mask denoising；
- confidence-first iterative unmask；
- teacher-forced 已知 token 比例 curriculum。

状态：**假设得到间接支持，直接 A/B 未完成**。

### 7.11 高频 residue 与 loss 改进

该问题与 7.6/7.7 相连。需要区分两件事：native 数据本身频率不均衡，与模型预测比 native 更严重地集中在少数类别。建议先做 logit-adjustment 和 inverse-sqrt-frequency CE 两个温和对照，并明确报告预测/标签 frequency ratio。若某些 AA 在合理 interface 环境中仍几乎从不出现，再考虑 class-balanced sampler 或 focal loss。

状态：**诊断部分完成，loss 对照未完成**。

## 8. 下一步实验优先级

### P0：修复并重跑原生 PXDesign sampler validation

先补齐 native sampler wrapper 的 `pair_z=None`、`p_lm=None`、`c_l=None`，加入真实签名覆盖测试，然后在完全相同的 8 个 PINDER complexes 上比较 minimal Euler 与 PXDesign native sampler。只有这一步完成后，才能判断 18–20 Å 的 free-generation RMSD 是模型能力还是采样器实现造成的。

### P1：验证 all-16 到底学了什么

对 `110540/step3000.pt` 做以下三项成对评估：

1. 同一 32 个训练样本上的 clean/low-sigma recovery；
2. held-out PINDER 上的 clean/low-sigma recovery；
3. shuffled/null backbone control。

只有 held-out 提升且 shuffled backbone 明显下降，才能说模型学到了可泛化的 geometry→AA mapping。

### P2：解冻深度 sweep

在相同 32-sample、clean-coordinate 设置下比较 4/8/12/16 blocks；根据训练集是否能记忆和 held-out 是否提升选择最小有效深度。同步检查 monomer backbone validation，避免 AA 适配破坏 fold prior。

### P3：partial-mask 与 confidence-first iterative unmask

先训练多 mask-ratio objective，再比较 one-shot 与反馈式 iterative decoding。完整保存 logits/entropy/KL/flip-rate trajectory，用于分析会议提到的“跃迁”。

### P4：rollout-aware AA supervision

在训练中周期性使用模型自身 rollout 的中间/后期状态计算 AA CE，缩小 noisy-native 与 Gaussian-start state distribution shift。先做短 rollout、小 batch 和有限解冻，控制显存与稳定性。

### P5：residue distribution 与 loss sweep

固定 P1/P2 选出的 architecture 后，比较 plain CE、inverse-sqrt weighted CE、logit-adjusted CE、focal loss 和 label smoothing。以 balanced accuracy、macro-F1、per-class recall 和 calibration 为主，不能只看 raw accuracy。

### P6：PLM sequence prior

先建立 frozen-PLM sequence prior 与 distillation baseline，再考虑结构 logits 融合。该方向改动更大，应该放在 representation、sampler 和 iterative unmask 的基本问题得到澄清之后。

### P7：长度泛化

去掉/提高 index length filter，按长度分桶验证 crop 接受率、吞吐、OOM、AA recovery 和 backbone quality；如需完整长复合物，再引入动态 token batching 或更大 crop。

## 9. 已完成的代码改动与 Git 状态

已提交到 `sjm/binder-design-training` 的主要 commit：

- `148eb18`：增加 source-specific AA metrics、AA head optimizer group/LR、gradient/update norm、PINDER evaluation、用户可写 cache 和 Stage III 对照提交脚本。
- `2b2c5e2`：增加 AA head 独立 gradient clipping、forced-sigma sampler、sigma weighting/per-sigma diagnostics、PINDER fixed-sigma comparison 和 low-sigma/tiny-overfit 脚本。
- `f1560e1`：更新 PXDesign embedder compatibility；对应 PXDesign submodule commit 为 `2202ad0`。
- `d434e8d`：增加 binder AA 诊断与 clean-coordinate/解冻实验支持，并修复结构指标实现。

截至 2026-09-05，分支 `sjm/binder-design-training` 比远端领先 1 个 commit；以下 native sampler 实验性改动仍未提交：

- `pxdesign_train/cogenerate.py`
- `tests/test_complete_unmask.py`
- `scripts/evaluation/infer_aa_readouts_pinder.py`
- `scripts/evaluation/slurm_infer_pinder_aa_readouts.sh`
- 本文档 `docs/aa_head_binder_experiments_zh.md`

这些未提交改动包含 PXDesign native sampler bridge、默认 native sampler 的 evaluation CLI 和相应测试。现有测试没有覆盖真实 Protenix forward 的三个必需参数，因此 Job 110546 暴露了接口缺口；修复并验证前不应提交该 bridge。工作区中的 `tmp.sh` 仍为用户的 untracked 文件，不纳入实验代码。

## 10. 最终判断

Stage III mixed-data crop-448 对 monomer backbone validation 有效，并且优于 crop-512 的稳定性；但主训练 AA prediction 从 step 4000 到 6000 基本不变。单纯提高 head LR、detach、low-sigma warmup、sigma reweighting和更换 trajectory readout 都没有解决该问题。

All-16 clean-backbone tiny-overfit 达到 99.95%，使当前判断发生了重要更新：模型不是绝对学不到，而是需要深层 representation adaptation；现有 shallow/frozen 设置不足。与此同时，这一结果仍可能只是 32-sample memorization，不能代表 held-out 泛化。

最合理的近期路线是：先修复原生 sampler、再对 all-16 checkpoint 做 held-out 与 shuffled-backbone 对照，然后定位最小解冻深度。完成这三步后，再进入 partial-mask/confidence-first unmask、rollout-aware supervision、类别 loss 和 PLM prior。这样每一步都能回答一个明确问题，避免同时改 architecture、loss 和 inference 后无法归因。
