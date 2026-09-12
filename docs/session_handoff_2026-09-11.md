# 交接：2026-09-11 这个 session 的全部注意事项

给下一个 session 看。**先读这份，再动手。** 分三部分：当前状态、已确立的结论
（含我改过三次的判断）、以及踩过的坑。

---

## 1. 当前状态

### 在跑的作业（23:50 时限，2026-09-11 15:37 时的快照）

| job | phase | crop | step | eval 次数 |
| --- | --- | --- | --- | --- |
| 114341 | Stage IV IV-A | 384 | 10950 | 5 |
| 114342 | Stage IV IV-F | 256 | 6100 | 3 |
| 114345 | Stage IV IV-F | 384 | 6700 | 3 |

按当前步速三个都到不了 `--max-steps 30000`。114345 存在的目的是取代 114342
（crop 256 的硬过滤砍掉 45% 训练数据，而显存实测只用到 47/144 GiB）。

**这三个的 checkpoint 已经不可 full resume** —— `implementation_identity()` 是
`@lru_cache` 的，各进程冻结了启动时的 HEAD（114341/114342 是 `bd5ddad`，
114345 是 `b74be19`），而 HEAD 早已前移。要保留可续跑性，规则是
**在启动 run 和它第一个 checkpoint 之间冻结分支**。

### 四个分支，都已推送

| 分支 | 内容 | worktree |
| --- | --- | --- |
| `sjm/stage4-ligandmpnn` | Stage IV 的 LigandMPNN backend、IV-F phase | `~/Proteo-AA-ligandmpnn` |
| `sjm/alphaproteo10_eval` | AlphaProteo-10 生成/打分链路、结果文档 | 主 checkout `~/Proteo-AA` |
| `sjm/stage3-binder-design` | **eval 分支的超集** + 单体探针 + PXDesign 调研 | `~/Proteo-AA-stage3-binder` |
| `sjm/binder-design-training` | session 之前就有的 | — |

**给别人看就给 `sjm/stage3-binder-design`**，它信息最全。

阅读顺序：`docs/alphaproteo10_designability_result_zh.md` →
`docs/pxdesign_target_conditioning_zh.md`。

### 存储约定

大文件一律放 `/hai/scratch/shenjm`。home 只有 50 GiB 且已用 54%，而一个
Stage IV checkpoint 是 **1.9 GiB**（单 run 约 57 GiB）。源码和 worktree 留 home。

---

## 2. 已确立的结论

### AlphaProteo-10 全量结果

| | 打分条数 | coverage | 严格通过 | designability |
| --- | --- | --- | --- | --- |
| 官方 PXDesign v0.1.0 | 3280 | 1.0 | 271 | **8.26%** |
| Proteo-AA 111408/step6000 | 3280 | 1.0 | **0** | **0.0%** |

coverage 都是 1.0，**不是上一轮那种「任务没提交」的假零**。

### 血缘定位：两个失效叠加

无条件单体（12 样本，80–200 残基，无靶点混淆）vs 复合体 binder 链：

| checkpoint | 单体坏键 | 复合体坏键 |
| --- | --- | --- |
| PXDesign official | **8.0%** | **0.0%** |
| Stage II 52500 | 60.1% | 98.1% |
| Stage III 111408 s6000 | 60.5% | 85.2% |

（坏键 = 连续 CA–CA 偏离 3.8 ± 0.3 Å 的比例）

1. **基础主链几何不合格** —— 60% vs 官方 8%
2. **复合体再叠一层退化** —— 同 checkpoint 从 60% 掉到 85–98%

Stage III 相对 Stage II **在单体上没改善**（60.1 → 60.5）、**在复合体上明显改善**
（98.1 → 85.2）—— 和它 75% PINDER 数据一致：那 6650 步学的是复合体/条件化。

### PXDesign 原文怎么做（technical report p24，PDF 在 `PXDesign/assets/`）

* **不固定 receptor 坐标**，原文说「we find it unnecessary to freeze」
* 所有原子一起加噪去噪
* 条件化走 **pair 表示 `z`**：靶点 binned 成对距离（64 bin + 1 特殊 bin）
* 明确对立于 RFdiffusion 的 inpainting

**Proteo-AA 已经忠实实现了这个**（`_helpers.py:105` 自述 byte-for-byte；运行时
验证 target-target pair 100% 覆盖、bin 0–63），采样器参数也逐项一致
（γ0=1.0 / γmin=0.01 / λ=1.003 / N_step=400 / η piecewise_65，继承官方
`configs_base`）。

### 我改过三次的判断 —— 不要重新推导

| 我说过 | 实际 |
| --- | --- |
| 「benchmark 接错了函数，换目标条件化采样器就能修」 | **错**。PXDesign 自己就是自由生成所有原子，`cogenerate` 是对的路径 |
| 「Stage III 缺靶点条件化」 | **错**。它有，我当时只查了坐标冻结那条路（`stage4_fixed_context`），漏掉 pair 表示这条真正的路径 |
| 「缺的是无条件主链阶段，不是靶点条件化」 | **只对一半**。单体数据出来后是「两个失效叠加」 |

推论一条，值得告诉 yfsun：**Stage IV 的坐标冻结（`generator.py:101` 的
`stage4_fixed_context`，只有 Stage IV 会设）是相对 PXDesign 的一处偏离**，
不是补齐缺失。

### GPT 的分析里哪些成立

| 论点 | 裁决 | 依据 |
| --- | --- | --- |
| 调用路径未核实、`cogenerate` 只是 minimal Euler | ✅ **对**（当时代码在未推送分支） | 已推送修复 |
| 默认配置 + `strict=False` 导致静默随机初始化 | ❌ | 运行日志 `1172/1180, missing=8, unexpected=0` |
| SC→BB refinement 引入错误 | ❌ | PXDesign 缺 448 个（SC 全随机）仍 0.0% 坏键 |
| 靶点条件可能为空/错误 | ❌ | pair 100% 覆盖 |
| Stage I s96000 vs Stage II s52500 一致性对照 | 👍 好建议，**还没做** | — |

### Stage IV 用 PXDesign ckpt 当 donor

**直接用不行**：官方 ckpt 只有 `diffusion_module`(636) + `design_condition_embedder`(96)，
缺 `sidechain_module`(400) 和全部反馈模块，`trainer.py:1047` 会硬拒（不会静默退化）。

**但拼接可行**：骨架部分和 Stage III **形状完全一致**（636+96 张量零不符）。
可以造 PXDesign 骨架 + Stage II/III packer 的混合 donor。顾虑是 packer 在
Stage II/III 的骨架表示上训练，换骨架后是分布外 —— IV-F 那个 phase（冻结序列头、
训 packer）正好是适配它的形状。考虑到官方骨架是手上**唯一**单体 8% / 复合体 0%
的，这个方向现在最有价值。

---

## 3. 踩过的坑

### 这个 session 最大的教训

**重建参考脚本的等价物，而不是照抄它的调用。** 同一个探针脚本因此失败 5 次，
每次烧一个 GPU 作业去发现一行错误：

| 次 | 错误 | 本质 |
| --- | --- | --- |
| 1 | `could not parse 2wh6.cif` | worktree 缺 gitignore 掉的结构文件 |
| 2 | `build_monomer_index() missing 'limit'` | 凭签名猜参数 |
| 3 | `Namespace has no 'allow_binder_sidechain_leakage'` | 没调 `fill_missing_args` |
| 4 | `DesignSourceDataset has no 'source_names'` | 把 curriculum wrapper 当数据集 |
| 5 | `cannot import _source_item_with_crop` | 从错误的模块 import |

`fill_missing_args()` 的 docstring 早就记了这个教训（「五个 eval 脚本一个属性
一个属性地死过」）。**照抄参考脚本的整段 setup，包括它的 dry-run。**

### 先跑 `--dry-run`，别拿 GPU 作业当语法检查

给探针加了 `--dry-run`（建模型前走完 imports → 回填 args → 建索引 → 取样本 →
解析 CA 行）。它随后拦下两个**不会报错**的问题：过滤后只剩 2 行、挑到 29 残基的
蛋白。这两个会以「跑完了、数字很小」的形式进入结论。

### Slurm / 环境

* **`sbatch` 从交互分配里提交会继承 shell 的 CPU/内存请求**，压过脚本的
  `#SBATCH`。剥掉 `SLURM_*`（保留 `SLURM_CONF`），然后
  `scontrol show job <id> | grep ReqTRES` 确认是 `cpu=8,mem=192G,gres/gpu:h200=1`。
* **`sbatch` 会把脚本复制到 `/var/lib/slurm/slurmd/job<ID>/slurm_script`**，
  所以 `BASH_SOURCE` 在 Slurm 下指向 slurmd 的目录。要定位 repo 就先试
  `SLURM_SUBMIT_DIR` 再退回 `BASH_SOURCE`，且候选目录必须含标志文件，否则 exit 2
  —— **绝不回退到硬编码默认值**（那会静默跑另一个 checkout 的代码）。
* **计算节点看不到 `/tmp`**（节点本地）。worktree 必须在共享存储。
* `yejin-interactive` 上限 2 小时，但 `yejin` 是 24 小时、`yejin-lo` 7 天。
  纯 CPU 的 `srun --partition=yejin --time=24:00:00 --pty bash` 不占 GPU 配额。
* **训练作业设了 `Requeue=1` 但 `--open-mode` 默认截断** —— 已在 ligandmpnn 的
  launcher 修成 append，其它 launcher 还没修。
* `LAYERNORM_TYPE` 不设会走 fused CUDA LayerNorm，需要 ninja；用 `torch` 或
  `openfold`。
* 别手搓已有 launcher 的等价命令 ——
  `slurm_generate_alphaproteo_designability.sh` 传了
  `--sampler-mode pxdesign_native`，漏掉会静默变成 `minimal_euler`。

### 数据/代码的暗坑

* **`benchmarks/alphaproteo10/structures/` 被 gitignore**，新 worktree 里那 10 个
  PDB 不存在，而报错是 `could not parse <f>.cif` 不是「文件缺失」。软链主仓库那份。
  （`.gitignore` 的 `structures/` 带斜杠不匹配软链，已改成 `structures`。）
* **`sample_level_output.csv` 里 AF2 指标是带方括号的列表字符串**（`[0.88]`），
  直接 `pd.to_numeric` 会**整列变 NaN**。要先 `.str.strip("[] ")`。
* `designability_by_target.csv` 的 `mean_*` 列是空的，汇总脚本没填。
* **共享 PINDER 树约 1/3 是 mode 600**（yfsun 的 run 用严格 umask 解出来的）。
  `is_file()` 只 stat，会选中不可读文件并跳过 archive 兜底。已修成
  `is_file() and os.access(R_OK)`，且 `--pinder-root` 指自己的 scratch。
* 日志里 `aa_ce=0 aa_acc=0` 是**旧 MLP head** 的指标，在 LigandMPNN backend 下
  恒为 0（那个 head 不存在）。序列 loss 是 `stage4/aa_pre`。IV-A 的
  `loss_bb=0` 是 phase 清零了结构损失，不是 backbone loss 真为 0。
* **IV-A 和 IV-F 的 `aa_pre` 不能横向比** —— IV-A 训 head（这个数会动），
  IV-F 冻 head 训 backbone（变化体现在 `loss_bb`/`mse`）。动的是等式两边。

### 工作方式

* **`pkill -f <脚本名>` 会匹配到自己那条命令并杀掉 shell。**
* 登录节点上同时 `torch.load` 两个 2 GiB checkpoint 会被 OOM kill；逐个 + `mmap=True`。
* 工作区里出现过**不是本 session 写的**内容（可能是并发 session）。提交别人写的
  东西之前逐条核对 —— 那次 8 个 validation 数字全对，但「不要提交」的警告前提
  不成立、参数匹配论断漏了一项、计数已过期。

---

## 4. 下一步（按价值排序）

1. **补 monomer 加权的主链生成阶段**。PXDesign 第一阶段上调 monomer-only 蒸馏
   数据学无条件主链几何，再转复合体；我们是恒定 0.25，而 Stage II 练的是侧链。
   判据用**无条件 monomer designability**，不要用 binder —— 主链没达标前 binder
   数字读不出信息（0/3280 就是例子）。
2. **PXDesign 骨架 + Stage II/III packer 的拼接 donor**，跑 Stage IV。理由见上。
   先用 IV-F smoke 的三条梯度（`aa_to_sc` / `bb_to_sc` / `sc_aux_to_sc`）验收。
3. **Stage I s96000 vs Stage II s52500 一致性对照**（GPT 的建议）：若骨架张量
   逐个一致，同配置同种子关 SC 下应产出一致输出，否则怀疑调用链。
4. `decode_blocks` 扫描（4 → 8 → 16），FaMPNN 和 LigandMPNN 都扫。几乎零成本，
   能回答「更细粒度顺序解码值不值」，是决定要不要做原生 AR 实现的前置证据。
5. 收窄 Stage IV 的 resume identity gate（`implementation_identity()` 哈希 git
   HEAD + 全部 `pxdesign_train/**/*.py`，任何 commit 都让旧 checkpoint 不可续）。
   **这是 yfsun 的决定**，他刻意设严的。

## 5. 不要做的事

* 不要把 benchmark 换成目标条件化采样器 —— 会让 Stage III 落到分布外。
* 不要把 8.26% 和 AlphaProteo / PXDesign 论文数字直接比（协议不同，且 tnfa 用的是
  PXDesign 的宽 hotspot 集合）。
* 不要把 0/3280 当作 `step6000` 的最终评价 —— 它本身不是完成的 Stage III
  （30000 步里 6650 步超时）。
* 不要在有 run 在跑时提交到那个分支（见第 1 节 resume）。
* **不要凭未验证的假设下结论** —— 这个 session 我这么做了三次，每次都被数据推翻。
