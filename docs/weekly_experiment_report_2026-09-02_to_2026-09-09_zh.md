# Proteo-AA Binder Design 周实验总结（2026-09-02 至 2026-09-09）

## 1. Executive summary

本周工作围绕三个问题展开：为什么 Stage III 的 AA head 学不出 binder sequence；完整 diffusion rollout 的 backbone 质量到底如何；以及怎样建立与 AlphaProteo 接近的 de novo binder designability benchmark。

最重要的结论是：

1. **AA label、binder mask、loss、optimizer 和 checkpoint overlay 没有发现会让训练完全失效的硬错误。** AA head 参数确实在更新，数据中的 native residue type 也在 xpb/masking 前保存。
2. **旧 Stage III 的主要瓶颈更像是 backbone 内部表示和训练/推理状态分布，而不是 AA head 这两层 MLP 本身。** 旧 Stage III 在 PINDER native-neighbourhood、σ≈0.4 时 AA accuracy 约 15.9%，完整 free rollout 后只有 7.9%–9.2%，且出现明显 LEU collapse。
3. **官方 PXDesign backbone 直接接同一种 AA head 可以学习可泛化的 binder sequence signal。** 冻结官方 PXDesign trunk、仅训练新 AA head，在 held-out PINDER、σ=0.4 上由 step 1000 的 28.80% 提升到 step 5000 的 **33.68%**；CE 从 2.307 降到 **2.129**。这远高于旧 Stage III 的约 15.9%。
4. **“全解冻能记忆”不等于“能泛化”。** 32-sample clean-backbone all-16 tiny-overfit 的训练 accuracy 达到 99.95%，但 held-out PINDER 在 σ=0.4 只有 4.53%，说明该 run 严重过拟合，不能作为 inverse-folding 成功的证据。
5. **Stage III 的完整 backbone generation 明显退化。** 在相同 PINDER native sampler protocol 下，官方 PXDesign 的 Cα RMSD/lDDT/TM 为 15.97 Å/0.344/0.173；Stage III step 6000 为 21.85 Å/0.231/0.099，step 4000 更差。
6. **AlphaProteo10 benchmark 已完成生成端 smoke pipeline，但 scoring 尚未形成完整可比较结果。** 当前 chain-fixed run 已生成官方 PXDesign 10 条和 Proteo-AA 40 条候选；AF2InitialGuess/Protenix scoring 仍有环境与字段兼容问题，不能报告最终 designability rate。

## 2. 本周实验与状态总览

| 方向 | 代表 job | 状态 | 主要结论 |
| --- | --- | --- | --- |
| 最后一层解冻 tiny-overfit | 109342、109451 | 完成 | 约 17.2%–20.6%，只解冻最后一层不够 |
| Clean-backbone capacity control | 110421、110540 | 完成 | head-only 22.81%；all-16 训练集 99.95% |
| all-16 held-out validation | 112045 | 完成 | σ=0.4 仅 4.53%，证实严重过拟合 |
| Kabsch/free-generation 修正 | 110407、110423、110424 | 完成 | 修正旋转方向；minimal sampler 仍约 19 Å RMSD |
| PXDesign-native PINDER free rollout | 112086、112087、112438、112496 | 完成 | 官方明显优于 Stage III；Stage II 与官方 RMSD 接近但 lDDT/TM 较低 |
| Stage III 可重复性 | 111408 | 完成至 step 6650 后 walltime | backbone 比 107904 好，但 AA plateau 可重复 |
| 官方 PXDesign + AA head tiny control | 112509、112510 | 完成 | 8-sample clean/σ=.04 均可到约 50% train-window accuracy |
| 官方 PXDesign + AA head full PINDER | 112552、112711；eval 112712–112714 | 完成 | held-out σ=.4 达 33.68%，是本周最关键正结果 |
| Protenix-PPI-only Stage III | 112722 | walltime @ step 5900 | 无 monomer/PINDER；AA moving average 仍约 12.5%–13.0%，只有 step 2000/4000 checkpoint |
| Partial-mask 训练能力 | 代码与测试完成 | 尚无正式对照结果 | 已支持 partial/time-dependent mask，尚不能声称改善 |
| AlphaProteo10 benchmark | 112849、113008、113009 等 | 生成完成，scoring 未完成 | 10 targets 可跑；尚无完整 designability 比较 |

## 3. AA head 失效诊断

### 3.1 代码与数据审计

本周检查了 AA supervision 的完整路径：

- native residue type 在 xpb/masking 之前保存；
- PINDER chain B 和 Protenix complex 的 binder selector 能正确选择 binder token；
- AA label 与 token 数量有严格 alignment guard；
- Stage II backbone checkpoint 与 AA donor checkpoint 的参数能分别完整 overlay；
- AA head 参数有非零 gradient 和 update，并非 optimizer 漏参。

因此，没有证据支持“label 错位”“AA head 没被训练”或“checkpoint 根本没加载”这几类硬错误。

但发现了一个 **validation logging bug**：`evaluate()` 对每个动态出现的 per-sigma key 都除以全局样本数，而稀疏 sigma bucket 只在部分样本上出现。因此旧日志里的 `val_aa_*_sigma_*` 不能直接引用；需要按 per-key count 或 token count 重算。这个问题不影响总 AA CE/accuracy，也不影响独立 fixed-sigma evaluator 的结果。

### 3.2 Tiny-overfit：需要多深的表示适配

固定 32 个 PINDER 样本、clean backbone、AA-only loss 的结果如下：

| Trainable part | Step 3000 CE | Step 3000 accuracy |
| --- | ---: | ---: |
| AA head only（110421） | 2.384 | 22.81% |
| AA head + all 16 diffusion blocks（110540） | **0.0051** | **99.95%** |

严格匹配的 “AA head + 最后一个 diffusion block” run 109451 在 step 1500 只有 CE 2.478、accuracy 20.57%，与 head-only 同量级。由此可以排除“整个网络没有能力表达 AA 任务”，但也说明只让最后一层适配并不足够。

随后对 110540/step3000 做 held-out PINDER 验证：

| Sigma | AA CE | AA accuracy | Balanced accuracy | Cα RMSD | Cα lDDT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.04 | 11.175 | 7.91% | 6.18% | 0.10 Å | 1.000 |
| 0.4 | 15.424 | **4.53%** | 5.15% | 0.69 Å | 0.891 |
| 4.0 | 11.342 | 6.26% | 5.18% | 3.20 Å | 0.540 |

所以 all-16 run 学会的是 32-sample memorization，而不是可泛化的 sequence-from-structure mapping。

## 4. 关键对照：官方 PXDesign backbone 直接训练 AA head

### 4.1 Protocol

- backbone：官方 `pxdesign_v0.1.0.pt`；
- 数据：PINDER train split；held-out PINDER validation 128 个 eval positions；
- 只训练新 AA head，约 1.30M 参数，PXDesign trunk 冻结；
- binder sequence 全 mask；
- 训练 sigma 在 0.04 和 0.4 间交替；
- side-chain/co-evolution path 关闭；
- AA LR `3e-4`，独立 gradient clipping；
- 112552 因一个 PINDER CIF parse/DataLoader 错误中断，112711 从 step 2000 在同一 run directory 正常恢复到 step 5000。

### 4.2 Held-out PINDER AA 结果

| Checkpoint | σ=0.04 CE / acc | σ=0.4 CE / acc | σ=4.0 CE / acc |
| ---: | ---: | ---: | ---: |
| step 1000 | 2.416 / 25.95% | 2.307 / 28.80% | 2.743 / 18.57% |
| step 3000 | 2.251 / 29.88% | 2.169 / 32.31% | 2.821 / 16.74% |
| step 5000 | **2.200 / 31.33%** | **2.129 / 33.68%** | 3.184 / 16.26% |

step 5000、σ=0.4 的 balanced accuracy 为 26.05%，macro-F1 为 0.246，且预测覆盖全部 20 类。这个改善不是单纯通过猜高频 residue 获得的。

同时，σ=4 的表现随着训练继续反而下降。这与训练只覆盖 0.04/0.4 一致，说明 head 对 high-noise latent 有明显 distribution shift。下一步若要把 AA 预测用于完整 rollout，需要在训练中加入更宽 sigma 或 rollout-state supervision，而不能只优化 native neighbourhood。

### 4.3 对旧 Stage III 的含义

这一对照比 tiny-overfit 更关键，因为它在 held-out 数据上成立。它说明：

- 相同类型的浅层 AA head 能从官方 PXDesign 的内部 representation 读出有效 sequence signal；
- PINDER label/loss 管线本身能学；
- 旧 Stage III 的约 13%–16% plateau 不是“AA head 必然太弱”；
- 更可能的问题是 Stage I/II/III 后的 diffusion representation 被改变、联合训练时 AA gradient 太弱，或 noisy-native training 与 free rollout state 不匹配。

## 5. Backbone generation 评估

### 5.1 评估协议

使用 PINDER validation、crop 448、native receptor condition、binder sequence 全 mask，从 Gaussian coordinates 开始，以 PXDesign native sampler 运行 400 steps。Binder native coordinates 不作为输入。结构指标按 binder 单独 Kabsch 对齐后计算，因此这里评估的是 binder fold matching，不是 receptor-relative binding pose。

请求了 128 条轨迹，但 crop retry 导致只有 95 个 unique complexes。不同 checkpoint 使用相同 indices/seeds，paired comparison 仍有效；不能把它描述为 128 个独立 complex。

### 5.2 结果

| Checkpoint | Cα RMSD mean / median ↓ | BB RMSD mean / median ↓ | Cα lDDT mean / median ↑ | TM-score mean / median ↑ |
| --- | ---: | ---: | ---: | ---: |
| 官方 PXDesign | **15.97 / 16.48 Å** | **15.90 / 16.44 Å** | **0.344 / 0.289** | **0.173 / 0.135** |
| Stage II parent step 52500 | 15.95 / 15.65 Å | 15.92 / 15.63 Å | 0.275 / 0.272 | 0.135 / 0.140 |
| Stage III step 6000 | 21.85 / 20.37 Å | 21.76 / 20.26 Å | 0.231 / 0.201 | 0.099 / 0.098 |
| Stage III step 4000 | 40.77 / 22.96 Å | 40.35 / 22.91 Å | 0.200 / 0.161 | 0.074 / 0.078 |

官方 PXDesign 相对 Stage III step 6000，在 98/128 条轨迹上 Cα RMSD 更低、124/128 上 lDDT 更高、95/128 上 TM-score 更高。Stage II 的 RMSD 与官方接近，但 lDDT/TM 已下降；Stage III 进一步退化，尤其 step 4000 出现严重长尾。

Binder length 是明显混杂因素。官方 PXDesign 在长度 ≤80 aa 时平均 Cα RMSD 10.37 Å、lDDT 0.571；长度 >220 aa 时恶化到 21.55 Å、0.243。本 eval 的 binder 中位长度为 158 aa，因此它比典型短 de novo binder 更难。

### 5.3 Free rollout 上的 AA readout

| Stage III checkpoint | Readout | AA CE | Accuracy | Top-5 accuracy |
| --- | --- | ---: | ---: | ---: |
| step 4000 | final | 2.945 | 7.61% | 33.44% |
| step 6000 | final | 2.921 | 7.87% | 35.03% |
| step 6000 | target σ≈0.363 | 2.998 | 8.69% | 35.45% |
| step 6000 | confidence-best | 3.128 | 9.22% | 34.41% |

PINDER 的 majority-class baseline 约 9.78%，top-5 frequency baseline 约 40.0%。这些 readout 都没有超过对应 baseline；step 6000 的 confidence-best 约 70% token 被预测为 LEU。这里首先暴露的是 backbone rollout OOD：fixed-sigma native neighbourhood 上尚有约 15.9%，完整 rollout 后 representation 已落到训练分布之外。

## 6. Stage III 复现与数据源对照

### 6.1 Mixed monomer/complex reproduction：111408

111408 使用 crop 448、gradient accumulation 8、Stage II step52500 加独立 AA donor，运行到 step 6650 后因 walltime 正常结束。Monomer validation：

| Step | Val loss | AA CE / acc | SC local | Cα / BB RMSD | TM-score |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 107.8 | 2.805 / 12.85% | 1.823 | 3.286 / 3.233 Å | 0.7896 |
| 4000 | **97.76** | 2.798 / 13.08% | 1.823 | **3.100 / 3.047 Å** | **0.8036** |
| 6000 | 104.8 | 2.800 / 13.08% | **1.812** | 3.224 / 3.171 Å | 0.7973 |

Backbone 相比早期 107904 run 有改善，但 AA plateau 完全复现。按 source 的训练 moving average，PINDER AA accuracy 从约 9.67% 缓慢升至 10.85%，Protenix PPI 从 11.85% 到 12.77%，monomer 基本停在 13%。

### 6.2 Protenix-PPI-only：112722

为了隔离 PINDER provider/分布是否导致 AA 不学习，启动了 monomer=0、PINDER=0 的 Protenix complex-only run。它在 24 小时 walltime 到达 step 5900，保存了 step 2000 和 4000 checkpoint。训练 moving-average `protenix_ppi_complex_aa_acc` 仍约 12.5%–13.0%，没有显示清晰学习趋势。

这个 run 没有独立 validation，单 batch 的 backbone 指标波动很大，因此目前只能说：**把数据源换成纯 Protenix PPI 没有立即解除 AA plateau**；还不能由此比较泛化或生成质量。

## 7. Partial masking 与 progressive decoding

代码现已支持 `all`、`partial`、`none` 和 `time_dependent` 四种 AA mask mode，并记录真实 mask fraction；也已为逐步 unmask/co-generation 补充基础接口和测试。

但本周没有一组严格匹配、完成验证的 partial-mask vs all-mask 训练结果。因此当前只能把它记作“实验能力已经具备”，不能写成“partial masking 已改善 AA”。下一步需要固定 backbone、数据、sigma、LR 和随机种子，仅改变 masking/unmask schedule。

## 8. AlphaProteo10 de novo binder benchmark

### 8.1 已完成

- 整理并校验 AlphaProteo 的 10 个 target 配置和 target-chain mapping；
- 增加无需 native binder 的 de novo placeholder-binder generation；
- 为 checkpoint loading 增加强校验，避免模型权重未加载却继续 benchmark；
- 建立 paired generation：官方 PXDesign + ProteinMPNN，与 Proteo-AA 同 backbone 上的 ProteinMPNN、AA-final、AA-target-sigma、AA-confidence-best；
- 建立独立 `pxdbench` 环境和 ProteinMPNN/AlphaFold scoring 权重准备脚本；
- chain-fixed smoke generation 成功：官方 PXDesign 10 targets × 1 = 10 rows；Proteo-AA 10 targets × 4 sequence arms = 40 rows，binder length 105，native sampler 400 steps。

### 8.2 尚未完成

AF2InitialGuess smoke 目前只有官方 PXDesign/ProteinMPNN 的 3 个 target 留下 summary，三者 `af2_easy_success=0`、`af2_opt_success=0`。这些任务同时报告 JAX 找不到 cuDNN，且缺少 Protenix scoring 字段；其余 target/arms 没有形成完整结果。随后生成端又修正了 biological-assembly chain handling，旧 smoke score 也不应与最新 chain-fixed generation 混用。

所以截至本报告时间：

- 可以说 generation pipeline 已跑通；
- 可以记录 3 个 AF2 smoke case 均未通过 AF2 阈值；
- **不能**报告官方 PXDesign 与 Proteo-AA 的完整 designability rate，也不能与论文数字比较。

## 9. “PXDesign 直接接 AA head”的输入到底是什么？

短答案：**AA head 的直接输入是 diffusion trunk 内部的 `a_token`，不是 raw geometry backbone coordinates。** 更精确地说，是 `DiffusionModule.diffusion_transformer` 之后、`layernorm_a` 输出的 per-token latent，默认维度为 768；AA head 另外接收 `aa_t` 的 time embedding。

数据流是：

```text
noisy binder backbone coordinates + target context + pair features
                         ↓
PXDesign atom encoder / diffusion transformer
                         ↓
post-layernorm a_token  [N_sample, N_token, 768]
                         ↓
LayerNorm → Linear → ReLU → Linear(20)
                         ↓
AA logits
```

因此，如果同学的问题必须二选一，答案是 **`a_token`**。但 `a_token` 是 geometry-aware representation：backbone geometry 已在上游进入 diffusion network，所以从功能上说这是“从 geometry-conditioned latent 预测 sequence”，不是一个 structure-blind sequence head。

对应代码位置：

- 默认配置将 `input_source` 设为 `diffusion_internal`：[`configs_train.py`](../pxdesign_train/configs/configs_train.py#L37)
- forward hook 从 `layernorm_a` 缓存 `a_token`：[`model.py`](../pxdesign_train/model.py#L587)
- AA loss 使用保留 sigma/sample 轴的 `a_full` 调用 head：[`model.py`](../pxdesign_train/model.py#L1086)
- `a_token` 位于 diffusion transformer 之后、atom decoder 之前：[`diffusion.py`](../Protenix/protenix/model/modules/diffusion.py#L470)
- AA head 本体是 time embedding 加两层 MLP：[`heads.py`](../pxdesign_train/heads.py#L107)

只有在 hook 没有捕获到 latent 的异常 fallback 情况下，代码才退回 `s_inputs` 并打印 warning；正常的本周官方 PXDesign + AA-head 实验走的是 `diffusion_internal/a_token` 路径。

## 10. 当前判断与下一步优先级

1. **先保护官方 PXDesign representation。** 以官方 backbone + trained AA head step5000 作为新的 sequence-readout baseline；不要直接从已退化的 Stage II/III trunk 推断 AA architecture 无效。
2. **补 high-sigma/rollout-state supervision。** 当前官方-head run 在 σ=0.04/0.4 学得很好，却在 σ=4 随训练恶化；训练分布必须覆盖真正 reverse diffusion trajectory。
3. **做最小解冻深度和 backbone-retention sweep。** 比较冻结、最后 4/8/12/16 blocks，同时监控 official free-generation 指标，避免得到 110540 式记忆但摧毁泛化。
4. **修复 validation per-key aggregation。** 在继续依赖内置 per-sigma 日志前，让 `evaluate()` 使用 per-key count；独立 fixed-sigma evaluator 可继续作为可信基准。
5. **完成 chain-fixed AlphaProteo scoring。** 先修好 cuDNN/JAX 和 Protenix score columns，再跑 10-target smoke；确认 coverage=100% 后才扩到多长度、多 seed。
6. **之后再比较 iterative unmask。** Partial masking、confidence-first unmask 和 sequence feedback 应在 backbone/latent baseline 稳定后做严格 A/B。

## 11. 主要证据位置

- Stage III/AA-head 历史汇总：[`aa_head_binder_experiments_zh.md`](aa_head_binder_experiments_zh.md)
- PINDER native-sampler 结果：[`pinder_free_generation_validation_zh.md`](pinder_free_generation_validation_zh.md)
- 111408 复现报告：[`stage3_binder_run_111408_report.md`](stage3_binder_run_111408_report.md)
- AlphaProteo10 protocol：[`alphaproteo10_designability_validation_zh.md`](alphaproteo10_designability_validation_zh.md)
- 官方 AA-head held-out eval：[`eval-offaa-s5000-112714.out`](../logs/validation/pinder_binder_backbone_inputs/eval-offaa-s5000-112714.out)
- all-16 held-out eval：[`pinder-all16-val-112045.out`](../logs/validation/pinder_binder_backbone_inputs/pinder-all16-val-112045.out)
- 官方/Stage III free generation logs：[`pinder-free-official-112438.out`](../logs/validation/pinder_binder_backbone_inputs/pinder-free-official-112438.out)、[`pinder-free-s6k-112087.out`](../logs/validation/pinder_binder_backbone_inputs/pinder-free-s6k-112087.out)
- chain-fixed AlphaProteo generation logs：[`alpha10-pxdesign-113008.out`](../logs/validation/alphaproteo10/alpha10-pxdesign-113008.out)、[`alpha10-proteoaa-113009.out`](../logs/validation/alphaproteo10/alpha10-proteoaa-113009.out)

