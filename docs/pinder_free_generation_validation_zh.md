# PINDER Binder Free-generation 阶段性结果

> 更新时间：2026-09-08  
> 目的：检查 Stage III binder joint training 是否保留了从高噪声生成 binder backbone 的能力，并与官方 PXDesign checkpoint 做同协议对照。

## 1. 问题背景

此前的 PINDER validation 主要是 **native backbone 附近的单步去噪**：将 native complex 加入指定强度的噪声，再让模型预测去噪后的结构。这类实验能够判断模型在给定结构附近是否会 denoise，但不能说明模型能否从 Gaussian noise 完成真正的 reverse-diffusion generation。

因此，本轮增加了真正的 PINDER binder free-generation：

- native receptor 结构作为条件；
- native binder 坐标不进入模型；
- binder residue identity 全部 mask；
- 完整复合物坐标从 Gaussian noise 初始化；
- 使用 PXDesign 原生 sampler 完成 400-step reverse diffusion；
- 最终对生成 binder backbone 和三个 AA readout 进行统计。

## 2. Evaluation protocol

| 项目 | 设置 |
| --- | --- |
| 数据 | PINDER validation split |
| Crop size | 448 |
| Sampler | `pxdesign_native` |
| Reverse-diffusion steps | 400 |
| Seed | 42 |
| 请求轨迹数 | 128 |
| 实际完成 | 128/128，零失败 |
| 实际 unique complex | 95 |
| Binder AA input | 全部 masked |
| Binder coordinate input | 不使用 native binder coordinates |
| Receptor condition | 使用 native receptor structure |
| 结构对齐 | 每个 complex 对 binder 单独做 Kabsch alignment |
| Pose metric | 当前没有 receptor-relative binding-pose metric |

128 条轨迹中只有 95 个 unique complex，是因为 dataset crop retry 会在前 128 个 provider rows 内重新取样，从而产生重复 complex。三个 checkpoint 使用相同 dataset indices 和 seeds，因此 checkpoint 之间的配对比较仍然有效，但不能把结果写成“128 个独立 PINDER complexes”。

## 3. Evaluated checkpoints

| 名称 | Checkpoint |
| --- | --- |
| Stage III step 4000 | `/hai/scratch/yfsun/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step4000.pt` |
| Stage III step 6000 | `/hai/scratch/yfsun/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt` |
| 官方 PXDesign | `/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt` |

官方 checkpoint 与当前 backbone 架构兼容。加载结果为：

```text
missing=11, unexpected=0
```

缺少的 11 个参数全部属于本项目后来增加的 AA/distogram heads；官方 checkpoint 的原生 backbone 参数完整载入。

## 4. Backbone free-generation results

### 4.1 Aggregate metrics

| Checkpoint | Cα RMSD mean / median ↓ | BB RMSD mean / median ↓ | Cα lDDT mean / median ↑ | TM-score mean / median ↑ |
| --- | ---: | ---: | ---: | ---: |
| 官方 PXDesign | **15.97 / 16.48 Å** | **15.90 / 16.44 Å** | **0.344 / 0.289** | **0.173 / 0.135** |
| Stage III step 6000 | 21.85 / 20.37 Å | 21.76 / 20.26 Å | 0.231 / 0.201 | 0.099 / 0.098 |
| Stage III step 4000 | 40.77 / 22.96 Å | 40.35 / 22.91 Å | 0.200 / 0.161 | 0.074 / 0.078 |

结论：官方 PXDesign 在相同 PINDER 样本和采样设置下明显优于 Stage III。Stage III step 6000 相比 step 4000 有明显改善，但绝对生成质量仍然很低。

### 4.2 Threshold statistics

| Checkpoint | Cα RMSD < 5 Å | Cα RMSD < 10 Å | TM > 0.3 | TM > 0.5 | lDDT > 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方 PXDesign | 6/128 | 20/128 | 13/128 | 7/128 | 18/128 |
| Stage III step 6000 | 0/128 | 3/128 | 0/128 | 0/128 | 4/128 |
| Stage III step 4000 | 0/128 | 3/128 | 0/128 | 0/128 | 4/128 |

在逐轨迹配对比较中，官方模型相对 Stage III step 6000：

- 98/128 个轨迹具有更低的 Cα RMSD；
- 124/128 个轨迹具有更高的 Cα lDDT；
- 95/128 个轨迹具有更高的 TM-score。

因此，官方模型的优势不是由少数异常值造成的。

### 4.3 Binder length effect

当前样本的 binder 较长：平均 164 aa，中位数 158 aa，范围为 21–333 aa。官方 PXDesign 的结果随 binder length 增长明显变差。

| Binder length | N | 官方 RMSD ↓ | 官方 lDDT ↑ | Stage III step 6000 RMSD ↓ | Stage III step 6000 lDDT ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤80 aa | 23 | 10.37 Å | 0.571 | 15.48 Å | 0.372 |
| 81–120 aa | 20 | 11.83 Å | 0.403 | 26.55 Å | 0.230 |
| 121–160 aa | 27 | 15.36 Å | 0.285 | 20.45 Å | 0.186 |
| 161–220 aa | 23 | 17.38 Å | 0.290 | 21.50 Å | 0.205 |
| >220 aa | 35 | 21.55 Å | 0.243 | 24.64 Å | 0.190 |

官方模型在短 binder 上最好，但即使限制到 ≤80 aa，其平均 native-matching Cα RMSD 仍为 10.37 Å。Stage III 在所有主要长度区间内总体弱于官方 checkpoint。

## 5. Stage III AA readout results

同一条 free-generation trajectory 记录三种 AA readout：

- `final`：最后一个 diffusion denoiser step；
- `target_sigma`：最接近请求值 σ=0.4 的 step，实际 scheduler sigma 约为 0.363；
- `confidence_best`：每个 token 选择整条 trajectory 中置信度最高的预测。

| Checkpoint | Readout | AA CE ↓ | Accuracy ↑ | Top-5 accuracy ↑ |
| --- | --- | ---: | ---: | ---: |
| step 4000 | final | 2.945 | 7.61% | 33.44% |
| step 4000 | target σ≈0.363 | 2.985 | 7.88% | 33.86% |
| step 4000 | confidence-best | 3.096 | 7.97% | 33.08% |
| step 6000 | final | **2.921** | 7.87% | 35.03% |
| step 6000 | target σ≈0.363 | 2.998 | 8.69% | **35.45%** |
| step 6000 | confidence-best | 3.128 | **9.22%** | 34.41% |

PINDER native AA distribution 的 majority-class accuracy 为约 9.78%，固定预测频率最高的五种 AA 可得到约 40.0% 的 top-5 baseline。Stage III 的 top-1 和 top-5 均未超过这两个简单 frequency baselines。

此外，step 6000 的 `confidence_best` 有约 70% 的 token 被预测为 LEU。它虽然获得三种 readout 中最高的 accuracy，但 CE 最差，说明该策略主要放大了错误高置信度和类别坍缩，而不是提取出更好的 sequence signal。

官方 PXDesign 不包含本项目新增的 AA head。它在加载后使用的是随机初始化 AA head，因此官方模型输出的 AA CE/accuracy 不具备比较意义，本报告只使用其 backbone metrics。

## 6. 与 fixed-sigma validation 的关系

Stage III step 6000 此前在同类 PINDER 样本上进行过 native-neighbourhood 单步去噪：

| Evaluation condition | Cα RMSD ↓ | Cα lDDT ↑ | TM-score ↑ | AA accuracy ↑ |
| --- | ---: | ---: | ---: | ---: |
| Native + noise，σ=0.4，单步 denoise | 0.45 Å | 0.959 | 0.985 | 15.9% |
| Native + noise，σ=4，单步 denoise | 1.83 Å | 0.697 | 0.852 | 14.1% |
| Gaussian noise，完整 400-step rollout | **21.85 Å** | **0.231** | **0.099** | 7.9–9.2% |

这说明模型能够在 native backbone 附近做局部去噪，但无法稳定地从高噪声区域通过完整 reverse diffusion 进入 native-like binder backbone basin。free-generation 中 AA head 接收到的是明显偏离训练数据分布的 backbone representation，因此这里观察到的 AA 失败不能单独归因于 AA head architecture；首先失效的是完整 backbone rollout。

## 7. 当前可以下的结论

1. **Stage III step 6000 优于 step 4000，但仍不具备可靠的 full free-generation 能力。**
2. **官方 PXDesign 的 high-noise generation 明显强于 Stage III checkpoint。** 这说明 evaluator/native sampler 本身不是完全失效的。
3. **目前还不能证明退化一定发生在 Stage III。** Stage III 并非直接从官方 checkpoint 开始，而是从本项目的 Stage I/II checkpoint warm-start；需要评估直接 parent checkpoint 才能定位退化阶段。
4. **当前 AA free-generation 结果接近或低于简单 residue-frequency baseline，并存在类别坍缩。** 但 malformed/OOD backbone 是重要混杂因素。
5. **绝对 native RMSD 不能完整代表 binder design quality。** 给定 receptor 时可能存在多个不同于 PINDER native binder 的有效设计；当前评估也没有测 receptor-relative pose、interface quality、clash 或 sequence–structure self-consistency。

## 8. 下一步实验

### P0：评估 Stage III 的直接 parent checkpoint

运行 Stage II `step52500` 的同协议 free-generation：

```bash
cd /hai/users/s/h/shenjm/Proteo-AA

CHECKPOINT=/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt \
RUN_ROOT=/hai/scratch/shenjm/proteo_aa_runs/pinder_free_generation/stage2_step52500_native_n400 \
MAX_SAMPLES=128 \
N_STEP=400 \
SAMPLER_MODE=pxdesign_native \
CROP_SIZE=448 \
SEED=42 \
sbatch --job-name=pinder-free-s2 \
scripts/evaluation/slurm_infer_pinder_aa_readouts.sh
```

判别逻辑：

- 如果 Stage II 接近官方、明显优于 Stage III，说明 Stage III joint training 损坏了 high-noise generative prior；
- 如果 Stage II 已经接近 Stage III 的低水平，问题更早出现在 Stage I/II backbone training，而不是由 AA head 或 Stage III 单独造成。

### P1：Teacher-started rollout

分别从 `native + noise` 的 σ=4、16、64、256 开始，并连续 rollout 到 0。该实验可以定位 trajectory 从哪个 noise range 开始偏离 data manifold，区分：

- 高噪声 denoiser 本身未学好；
- 多步误差累积；
- sampler schedule/训练 noise distribution 不匹配。

### P2：补充真正的 binder-design quality evaluation

后续至少应加入：

- receptor-aligned binder RMSD / ligand RMSD；
- interface contacts、clash rate 和界面几何；
- 生成序列重新折叠后的 self-consistency RMSD；
- Protenix/AF2 的 pLDDT、ipTM 或同类 complex confidence；
- 每个 receptor 生成多个 binder 后的 success rate 与 diversity。

## 9. Result locations

```text
# Official PXDesign
/hai/scratch/shenjm/proteo_aa_runs/pinder_free_generation/
  official_pxdesign_v0.1.0_native_n400/inference_summary.json

# Stage III step 4000
/hai/scratch/shenjm/proteo_aa_runs/pinder_free_generation/
  111408_step4000_native_n400/inference_summary.json

# Stage III step 6000
/hai/scratch/shenjm/proteo_aa_runs/pinder_free_generation/
  111408_step6000_native_n400/inference_summary.json
```

