# 蛋白设计的测试集与评估方式：现状梳理

调研范围：market 上 co-design 及相关蛋白设计工作的测试集怎么建、靶点怎么选、用什么工具打分、什么阈值算成功。
不限 binder，覆盖 enzyme、antibody、nanobody、peptide、motif scaffolding、无条件生成。

**日期**：2026-09-10
**状态**：广度调研完成。**binder / ProtDBench / A-CODE（§9，2026-09-12）与 单体 / 无条件 / motif scaffolding（§13，2026-09-13）已做过源码级核实**，事实逐条带证据标记，见文首。抗体、纳米抗体、酶三节仍以第一轮广度调研为主，标记从保守。

## 事实标记与适用性标记

每条事实后面跟一个标记，说明它的证据强度。**不要把 INFERRED 当成论文写明的 protocol 用。**

| 标记 | 含义 |
|---|---|
| `EXPLICIT_SOURCE` | 论文正文、附录或 SI 里明确写出的 **published protocol**。引用时可以说"论文规定" |
| `CODE_VERIFIED` | 从放出的 **released implementation**（源码或数据）里直接读到/算出来的 |
| `SOURCE_CODE_CONFLICT` | 论文写的和放出的代码不一致。**两边都记下来，不自动以哪边为准**——放出的实现可能是重写版或有版本漂移，论文描述的也可能是当时真跑的那版 |
| `INFERRED` | 由现有证据推出，但没有任何一处明确这么写。**不能当 protocol 引用** |
| `UNRESOLVED` | 查过但没查到，或两处来源冲突未能裁决。明确标出来，不要填一个看着合理的数 |

> `EXPLICIT_SOURCE` 和 `CODE_VERIFIED` 是**两类不同的事实**，不是强弱两档：前者是作者声明的协议，后者是这份放出的代码实际做的事。要复现论文数字时以前者为准并注明版本；要跑这份代码时以后者为准。

对 Proteo-AA 的适用性另用一组标记：

| 标记 | 含义 |
|---|---|
| `DIRECT_NOW` | 靶点/输入格式和我们现有管线对得上，拿到权重和打分环境就能跑 |
| `NEEDS_ADAPTER` | 任务本身我们能做，但要写接入代码（输入格式、条件化方式或打分接口不同） |
| `FUTURE` | 依赖我们目前没有的能力（配体条件化、抗体框架等），训练侧补上之后才谈得上 |
| `OUT_OF_SCOPE` | 不是设计任务（性质预测、对接、打分器评测），只作参考 |

---

## 覆盖对照：mentor 点名的范围

> "包括但不限于 binder design/enzyme design，可以是 miniprotein, peptide 也可以是 antibody/nanobody"

| 范围 | 收了几条 | 在哪 |
|---|---:|---|
| **binder design** | 12 | §3.1、§9.1–9.7、§9.12 |
| **enzyme design** | 9 | §11.1–11.11 |
| **miniprotein** | 10 | 即 binder 那条线（AlphaProteo / ProtDBench / A-CODE / PXDesign / BindCraft / BoltzGen / RFdiffusion / Cao / Latent-X / Adaptyv） |
| **peptide** | 7 | §9.8–9.11、§9.5（大环肽）、§9.1（肽预设）、§9.2（肽协议） |
| **antibody** | 7 | §10.1–10.6、§10.12 |
| **nanobody** | 8 | §10.15 汇总，详见 §10.6–10.11、§10.13、§9.2 |

"不限于"的部分另外收了：**motif scaffolding**（MotifBench、La-Proteina、RFdiffusion motif 集、Genie2）、**无条件单体生成**（FrameFlow 约定俗成协议、ProteinBench）、**逆折叠**（LigandMPNN、ProteinBench、CDR 逆折叠基准）、**口袋重设计**（PocketGen / PocketFlow）、**合法性检查器**（PoseBusters），以及 **§6 的评估方法学专章**（9 篇专门研究"该信哪个指标"的工作）。

---

## 一、先说结论

不是"市面上有 N 个 benchmark"，而是：

> **这个领域对自己这套 in-silico 评估标准的信任正在崩塌，而且证据来自至少九项彼此独立的工作。**

| 证据 | 说了什么 |
|---|---|
| Rocklin 组，2025 | 11 项研究、**614 个做过实验的从头设计单体**回测："很多失败的设计，置信度指标比成功的还好看"。最好的单一指标是 ESMFold 平均 pLDDT，把所有指标做逻辑回归"只有很小的改进" |
| Korbeld 等，Protein Science 2026 | refolding 指标会被**进化信息污染**——设计与天然序列同源时分数虚高，对实验成功的预测力下降 |
| MotifBench，2025 | 把 RFdiffusion 集合里的 **1QJG、1PRW 两个案例直接删除**，理由是 ESMFold 与无 MSA 的 AF2 在这两个案例上成功率差异过大。有名有姓的"换打分器排名就变" |
| ProtDBench，ICML 2026 | 八个打分器横比 + 湿实验锚定，发现"substantial verifier-dependent bias and limited agreement under identical filtering protocols" |
| Overath 等，2025 | **15 个靶点、3766 个做过实验的 binder**：AF3 的 **ipSAE 比 ipAE 平均精度高 1.4 倍** |
| Dunbrack，2025 | 标题即结论——"AlphaFold 的 ipTM 分数错在哪、怎么修" |
| Adaptyv EGFR 竞赛，2025 | 真做实验后暴露"缺乏标准化 benchmark、实验设计靶点和稳健的计算指标" |
| GPCR 多肽评估，2026 | 三个预测方法**全部**对摆错位置的肽给出过高置信度；"scoring problem 至今未解决" |
| Riff-Diff（酶），Nature 2026 | "预测的酶排名与实际活性之间相关性可忽略" |
| "Is Retrieval All You Need?"，2026 | 所谓"新颖"的生成骨架**大多能检索到已知结构域**，整链看着新而已。八个模型全中，含 PXDesign |

**三条线（binder / 单体 / 酶）都有公开证据说 in-silico 排名预测不了实验结果。**

需要和这条分开说的是**复现性**：论文之间其实复现得很好。ProtDBench 的 `af2_easy` 逐个靶点复现 A-CODE Table 4 到小数点后两位（§6.5.3，`CODE_VERIFIED`）。真正让数字剧烈变化的是**换过滤档**——同一批设计换一档，均值差 2.2 倍、个别靶点差 300 倍。所以"数字对不上"通常不是谁算错了，是没说清用的哪把尺子。

这直接决定我们该交付什么：不只是一张 benchmark 清单，而是**我们该在哪个测试集上、用哪个打分器报数**的一个明确建议。

---

## 二、三种评估范式，互相不通

把所有 benchmark 放一张表横向比大小本身就是错的——它们测的不是同一件事。

| | 有没有标准答案 | 怎么判 | 典型 |
|---|---|---|---|
| **范式一 自洽类** | 没有，从零设计 | 序列单独折回来，看跟设计的结构像不像；binder 另加"会不会结合" | binder、单体生成、motif scaffolding |
| **范式二 恢复类** | **有**，本质是把已知结构挖掉一段重设计 | 跟原生序列比恢复率、跟原生结构比 RMSD、Rosetta 能量、DockQ | 抗体 CDR 设计 |
| **范式三 活性类** | 没有 | 配体几何合不合法 + 真做实验测活性 | 酶、配体条件设计 |

抗体那条尤其要注意：**它是 redesign 不是 de novo**，能直接跟真值比，压根不用自洽那一套。这是为什么"多测几个 benchmark"汇不出一张可比的表。

---

## 三、范式一：自洽类

### 3.1 binder（miniprotein / peptide）

**详细 recipe 见 §9**，本表只作索引。数字均已与 §9 对齐。

一个反常识的发现：**binder 这边看着有三个 benchmark，实际上是同一批靶点套了三层壳。**
ProtDBench、PXDesignBench、A-CODE 跑的都是 AlphaProteo 那十个，只是过滤档不同。

| 名称 | 年 | 靶点数 | 打分器 | 代码 | Proteo-AA | recipe |
|---|---|---|---|---|---|---|
| **AlphaProteo** `arXiv:2409.08022` | 2024 | **10**（Table S1 全部）/ 8 做湿实验 | AF2 类 | 无 | `DIRECT_NOW` | §9.1 |
| **ProtDBench** `arXiv:2605.04118` | 2026 | 10 + 5 个 Cao 靶点 | AF2-IG、ColabFold、Protenix(-Mini)、Boltz-1/2、Chai-1、ESMFold | `congliuUvA/ProtDBench` | `DIRECT_NOW` | §9.1 |
| **A-CODE** `arXiv:2605.03360` | 2026 | 10 | AF2-IG（`af2_easy` 档） | 无 | `DIRECT_NOW` | §9.1 |
| **PXDesign / PXDesignBench** bioRxiv 2025.08.15.670450 | 2025 | 10 蛋白 + 12 环肽 | AF2-IG、Protenix；单体用 ESMFold | `bytedance/PXDesignBench` | `DIRECT_NOW` | §9.1 |
| **BindCraft** Nature 2025 | 2025 | **12**（v1 预印本 10） | AF2-multimer 设计 + AF2 单体重预测 | `martinpacesa/BindCraft` | `NEEDS_ADAPTER` | §9.2 |
| **BoltzGen** bioRxiv 2025.11.20.689494 | 2025 | **10** 个低同源新靶点 + 5 简单；8 场湿实验共 26 靶点 | **Boltz-2** | `HannesStark/boltzgen` | `NEEDS_ADAPTER` | §9.3 |
| **RFdiffusion** Nature 2023 | 2023 | 5 | AF2 带 initial guess + 靶点模板 | `RosettaCommons/RFdiffusion` | `NEEDS_ADAPTER` | §9.4 |
| **Latent-X** `arXiv:2507.19375` | 2025 | 7 湿实验 + 200 结构 in-silico 集 | Chai-1（主）、Boltz-2 | **无**（商业） | `OUT_OF_SCOPE` | §9.5 |
| **ODesign** `arXiv:2510.22304` | 2025 | 10 或 11（`UNRESOLVED`） | AlphaFold3，只给靶点 MSA | `UNRESOLVED` | `NEEDS_ADAPTER` | §9.6 |
| **Cao et al.** Nature 2022 | 2022 | **12 蛋白 / 13 位点** | RIFDock + Rosetta，酵母展示筛选 | IPD 六个 tar.gz | 打分器校准 `DIRECT_NOW` | §9.7 |
| **Adaptyv EGFR** bioRxiv 2025.04.17.648362 | 2025 | 1 靶点；R1 201 条 / R2 402 条 | ColabFold（两轮设置不同） | `adaptyvbio/egfr_competition_{1,2}` | 标签数据 `DIRECT_NOW` | §9.8 |
| **BoltzDesign1** bioRxiv 2025.04.06.647261 | 2025 | `UNRESOLVED` | Boltz-1 设计环 + AF3 验证 | `yehlincho/BoltzDesign1` | `NEEDS_ADAPTER` | §9.9 |
| **BindEnergyCraft** `arXiv:2505.21241` | 2025 | 8 | AF2-Multimer、Rosetta、Boltz-1 | 未放出 | `NEEDS_ADAPTER` | 未做 recipe |
| **Proteína-Complexa** `arXiv:2603.27950` | 2026 | **19**（12 easy + 7 hard）+ 4 小分子 + 41 AME | AF2-Multimer、RoseTTAFold3 | 承诺放出 | `NEEDS_ADAPTER` | 未做 recipe |

**peptide 单独一支**（§9.10）：PepBench / PepGLAD（测试 93）、BOND-PEP（193 对）、DiffPepBuilder（30 个重建）、RFpeptides（4 个大环肽靶点）。

### 3.2 单体 / motif scaffolding

**详细 recipe 与核实见 §13**，本表只作索引。**注意 §13.0 先分清了 designability / co-designability / all-atom co-designability 三个词——只有最后一个测得到我们的全原子 claim。**

| 名称 | 年 | 规模 | 打分器 |
|---|---|---|---|
| **MotifBench** `arXiv:2502.12479` | 2025 | 30 个问题；每 motif 100 个骨架 × 8 条序列 | ProteinMPNN → ESMFold → Kabsch RMSD；Foldseek 查唯一性和新颖性 |
| RFdiffusion motif 集 | 2023 | 25 | ProteinMPNN + AF2 |
| Genie2 变体 | 2024 | 24（去掉 6VW1） | ProteinMPNN + ESMFold |
| **无条件生成的约定俗成协议**（FrameFlow/FoldFlow） | 2024 | 长度 100/150/200/250/300 | 每骨架 ProteinMPNN 出 8 条序列（T=0.1），ESMFold 折回，取最小 scRMSD |
| **ProteinBench** `arXiv:2409.06744` | 2024 | 七类任务；抗体 55、CAMEO2022 183、apo-holo 91、ATLAS 82 | TM-score、RMSD、pLDDT、scTM；AF2 |
| La-Proteina | 2025 | 26 个全原子 motif 任务 | 全原子 co-designability |

> 注意：**MotifBench 和 La-Proteina 都不含配体**，只是蛋白原子的 motif。不要归到酶那一类。

---

## 四、范式二：恢复类（抗体 / 纳米抗体）

事实标准很集中：**RAbD 55 个案例**。ProteinBench、dyMEAN、DiffAb、AbDPO 全跑这个。
（原始 benchmark 是 60 个，ML 圈用筛过的 55 个，为什么砍掉 5 个尚未查清。）

| 名称 | 年 | 规模 | 指标 |
|---|---|---|---|
| **RAbD** | 2018 | 60（ML 用 55） | Rosetta 能量、序列恢复率 |
| **ProteinBench 抗体分支** | 2024 | 55 | AAR、RMSD、TM-score、Rosetta 结合/总能量、CN-score、AntiBERTy、IgFold 算 scRMSD |
| **CHIMERA-Bench** `arXiv:2603.13431` | 2026 | 2922 个复合物，3 种划分，11 个方法 | AAR、CAAR、Cα-RMSD、DockQ、表位 P/R/F1 |
| **dyMEAN 三任务协议** | 2023 | RAbD 55 + SKEMPI 53 | 被后续论文大量沿用 |
| **AbBiBench** | 2025 | 14 抗体 × 9 抗原，>184500 条测量 | 模型似然与实测亲和力的相关性 |
| **AIntibody** Nat Biotechnol | 2024/2026 | 29 家机构 511 个 AI 设计抗体 | CASP 式盲测，真做实验 |
| **abag-benchmark-set** | 2026 | 110 个低同源复合物 | DockQ、TM-score、ipSAE、pDockQ2；比 AF2.3/AF3/Boltz-1/Chai-1 |

### 纳米抗体：**这是一块空白**

查下来不存在 nanobody 版的标准设计 benchmark。

- 结构/对接方面有：Hitawala & Gray（60 个 Nb-Ag，独立分支）、ABAG-docking（14 个单域抗体）
- 序列设计有两个：某 IF benchmark 单独留了 **61 个 VHH**；**nanoFOLD**（43 个晶体 + 1064 个模型）
- **抗原条件下的 de novo 纳米抗体生成：基本没有共享测试集。** 最接近的是 IgGM 的 SAb-23H2-Nano（27 个纳米抗体），但那是单篇论文的时间划分
- NbBench 是唯一专门为纳米抗体做的套件，但 8 个任务里 7 个是预测，只有 CDR 填空沾边生成
- 注意 abag-benchmark-set **明确排除了纳米抗体**

> **要测纳米抗体设计，没有现成的可用，得自己建。** 这本身是个可交付项。

---

## 五、范式三：活性类（酶 / 配体条件设计）

**没有公认的 in-silico 酶设计 benchmark，主流仍是逐案例做湿实验。** 但正在变化。

| 名称 | 年 | 规模 | 打分器 | 湿实验 | 代码 |
|---|---|---|---|---|---|
| **AME（Atomic Motif Enzyme）** | 2025 | **41 个活性位点**，M-CSA × PARITY，EC 1–5 类 | LigandMPNN 出序列、Chai-1 折叠；催化重原子 RMSD + 配体撞车 | 有 | 随 RFdiffusion2 放出 |
| **Studio-179**（DISCO）`arXiv:2604.05181` | 2026 | 179 = **170 个配体 + 9 个多配体组合** | Chai-1（骨架 + 配体质心 RMSD < 2 Å）、AF3、ESMFold、PoseBusters | benchmark 部分无 | `DISCO-design/DISCO` |
| **SAM / OQO / FAD / IAI 四分子惯例**（RFdiffusionAA） | 2024 | 4 | 各家不同：AF2/AF3/Chai-1/RF3 | 原文有 | Baker lab |
| **LigandMPNN 测试集** | 2023/25 | 317 小分子 + 74 核酸 + 83 过渡金属 | 序列恢复率、χ1/χ2 恢复率 | 有（部分来自配套论文） | `dauparas/LigandMPNN` |
| **EnzyBench**（EnzyGen） | 2024 | 3157 个四级 EC 家族，1500 测试 | ESP score ≥ 0.6、AF2 pLDDT、Gnina docking | 无 | `LeiLiLab/EnzyGen` |
| **EnzyBind**（EnzyControl） | 2025 | 11100 对酶–底物 | ESMFold、ProteinMPNN、CLEAN、UniKP、Gnina | 无 | `Vecteur-libre/EnzyControl` |
| **CrossDocked2020 / Binding MOAD 口袋设计** | 2024 | 各 100 对 | AAR、scRMSD、Vina、PoseBusters | 无 | PocketGen / PocketFlow |
| **COMPSS** Nat Biotechnol 2025 | 2025 | 2 个酶家族，**500+ 条序列真表达纯化** | 拿 20 个 in-silico 指标去对实测活性 | 有 | `seanrjohnson/protein_scoring` |

**AME 是目前唯一有跨论文可比性的酶 benchmark**——RFdiffusion2 提出，一年内被 RFdiffusion3、Proteina-Complexa、ODesign 复用。但它是 41 个案例、一个实验室建的、没有排行榜也没有独立维护方。

**配体那边没有标准，只有惯例**：SAM/OQO/FAD/IAI 四个分子，各家打分器不同，数字不可横比。Studio-179 是想取代这个惯例的新选手，但目前**没有第三方复用**。

**没有酶设计的盲测社区挑战。** CASP16 有配体类别，但那是预测不是设计。

---

## 六、评估方法学：值得单独看的一批

这批比 benchmark 清单更有用，因为它们直接回答"该信哪个打分器"。

| 名称 | 年 | 结论 |
|---|---|---|
| Rocklin 组，零样本预测设计成功 | 2025 | 614 个实测单体；所有模型区分成败的能力都只是中等 |
| Overath 等，binder 成功预测 | 2025 | 3766 个 binder / 15 靶点；**ipSAE > ipAE，平均精度 1.4 倍** |
| Dunbrack，ipSAE | 2025 | ipTM 按整链算，无序区和附属域会压低分数；ipSAE 只看 pAE 低于阈值的残基对 |
| Korbeld 等，refolding 管线的局限 | 2026 | 进化信息污染自洽指标 |
| Protein FID | 2025/26 | 现有指标不衡量"是否覆盖了训练分布"；提出蛋白版 FID |
| Is Retrieval All You Need? | 2026 | 新颖性被高估；提出 Domain Retrieval Rate |
| GPCR 多肽评估 | 2026 | 三个验证方法全部对错误摆位过度自信 |
| COMPSS | 2025 | 酶版的指标校准，500+ 条实测 |
| 多状态蛋白预测偏差 | 2026 | 预测器倾向于给出 PDB 中的主导构象 |

---

## 六点五、我们自己算的（不是引别人的结论）

ProtDBench 把原始数据放出来了，所以它的核心主张我们可以直接验证，不用引摘要。
数据：`git clone https://github.com/congliuUvA/ProtDBench`，`data/` 目录 164 MB。

### 6.5.1 换个打分器，方法排名就变

`data/generative_benchmark/10_targets/` 里是七个方法在十个靶点上的**全部设计及其分数**（每个方法 2.2 万–3.1 万条）。同一批设计，按不同判定算通过率：

| 方法 | af2_easy | af2_opt | ptx_mini |
|---|---:|---:|---:|
| PXDesign | **19.88%** | 8.48% | 4.18% |
| BoltzDesign-1 | 13.08% | 2.06% | 0.16% |
| BoltzGen | 12.27% | 4.49% | **4.43%** |
| BindCraft | 无数据 | **12.24%** | 3.60% |
| RFDiffusion-3 | 8.61% | 3.07% | 1.14% |
| ODesign | 8.64% | 1.37% | 0.38% |
| Protpardelle-1c | 1.41% | 0.13% | 0.01% |

**排名随打分器变化：**
- af2_easy：PXDesign > BoltzDesign-1 > BoltzGen > …
- af2_opt：**BindCraft** > PXDesign > BoltzGen > RFDiffusion-3 > BoltzDesign-1
- ptx_mini：**BoltzGen** > PXDesign > BindCraft > RFDiffusion-3 > ODesign > BoltzDesign-1

**BoltzDesign-1 从第 2 名掉到第 6 名**，通过率 13.08% → 2.06% → 0.16%，相差 80 倍。
PXDesign（我们的骨架）在三种判定下都稳在第 1–2 名。

同一批 PXDesign 设计，判定之间的重叠（Jaccard）：
- af2_easy vs af2_opt：42.7%（嵌套关系，严格档是宽松档的子集）
- af2_opt vs ptx_mini：**25.1%**（1349 个只有 AF2 认可，372 个只有 Protenix 认可）
- af2_easy vs ptx_mini：**14.3%**

### 6.5.2 哪个打分器真的能预测湿实验？

`data/filter_benchmark/cao_verifier_scores.csv.gz`：**236,246 个设计 × 8 个打分器 × 4 个指标，外加一列真实湿实验标签**（1,485 个真结合，命中率 0.63%）。

取八个打分器**都有分数**的共同子集（140,904 条设计、8 个靶点、1,369 个真结合），算各自对真实结合的 AUC：

| 打分器 | ipAE | ipTM | pLDDT |
|---|---:|---:|---:|
| **ColabFold** | **0.801** | 0.792 | 0.765 |
| Protenix | 0.762 | 0.748 | 0.652 |
| Protenix-Mini | 0.755 | 0.740 | 0.675 |
| Chai-1 | 0.742 | 0.732 | 0.622 |
| Boltz-1 | 0.729 | 0.729 | 0.637 |
| **AF2-IG** | **0.727** | 0.688 | 0.571 |
| Boltz-2 | 0.703 | 0.710 | 0.601 |
| ESMFold | 0.607 | 0.581 | 0.667 |

**四条结论：**

1. **AF2-IG 在八个里排第 6。** 而它是全领域的事实标准，也是我们现在用的。ColabFold 高出 0.074 AUC。
2. **ipAE 一致优于 ipTM，更优于 pLDDT。** 这与 Overath 等人推荐 ipSAE（ipAE 的改进版）的方向一致。pLDDT 作为筛选依据很弱。
3. **新的不一定更好**：Boltz-2 (0.703) 低于 Boltz-1 (0.729)。
4. **没有一个超过 0.80。** 在 1% 的基础命中率下，这个区分度是弱的——与第一节里九项工作的结论完全吻合。

**必须同时说明的三点局限：**
- Cao 的设计是 Rosetta 时代的能量法设计，不是现代生成模型的产物，打分器排名未必能直接迁移到我们的设计上
- ESMFold 是单序列模型、本就不是为复合物设计的，这里 AUC 低属于预期，**不能据此否定它在单体自洽评估中的用途**
- 这是 8 个靶点，不是 AlphaProteo 那十个


### 6.5.3 同一批数字，换个过滤档相差 300 倍

**先纠正本文档早前的一个错误结论。** 此处原先写的是"同一个方法、同一批靶点、同一套过滤器，两篇论文差 302 倍"，并归因为复现失败。**那是错的**——是拿 A-CODE 的 `af2_easy` 数字去比 ProtDBench 的 `af2_opt`，档位对错了。

实际核对结果恰恰相反：

| 靶点 | A-CODE Table 4 | ProtDBench `af2_easy` | ProtDBench `af2_opt` |
|---|---:|---:|---:|
| BHRF1 | 43.90 | **43.90** | 32.09 |
| H1 | 12.08 | **12.08** | 0.04 |
| IL17A | 0.82 | **0.82** | 0.00 |
| IL7RA | 29.80 | **29.80** | 0.26 |
| IR | 25.04 | **25.04** | 12.31 |
| PDL1 | 45.33 | **45.33** | 31.16 |
| SC2RBD | 11.20 | **11.20** | 3.96 |
| TNFa | 3.43 | **3.43** | 0.26 |
| TrkA | 23.55 | **23.55** | 12.06 |
| VEGFA | 16.72 | **16.72** | 4.81 |
| **均值** | **21.19** | **21.19** | **9.70** |

**十个靶点逐个吻合到小数点后两位，平均绝对偏差 0.00，相关系数 1.000。** 对 `af2_opt` 则是偏差 11.49、相关 0.867。

正确的结论有两条，方向都和原先写的相反：

1. **复现性其实很好。** ProtDBench 的 `af2_easy` 就是 A-CODE Table 4 的那把尺子，数字分毫不差。
2. **巨大的差异来自换档，不是来自换论文。** 同一批设计，`af2_easy` 均值 21.19%、`af2_opt` 均值 9.70%，**均值差 2.2 倍，个别靶点差 300 倍**（H1 从 12.08 掉到 0.04，IL7RA 从 29.80 掉到 0.26）。

这依然是本文档主线最有力的实证——**只是它证明的是"过滤档决定数字"，不是"论文之间复现不了"**。

### 6.5.3a A-CODE Table 4 用的是哪一档（已核实）

A-CODE 附录 C.2 的阈值，与 ProtDBench 的 `af2_easy` 一致：

```
pLDDT > 0.80
ipTM  > 0.50
ipAE  < 10.85 Å           （= 归一化 0.35 × 31）
binder bound/unbound RMSD < 3.5 Å
```

A-CODE 自陈按 **PXDesign protocol** 做 Table 4。第三方佐证：ODesign 附录 C.1.2 原文写 *"Based on the **AF2-IG-easy filter definition in PXdesign**（ipAE<10.85, pLDDT>0.8, ipTM>0.5, binder bound/unbound RMSD < 3.5Å）"*。加上上面这张逐靶点吻合表，**三处独立证据指向同一组阈值**。

`af2_opt`（pLDDT>0.9、unscaled ipAE<7.0、binder RMSD<1.5）是**另一套更严的档**，ProtDBench 把两套都保留了。**它不是 A-CODE Table 4 用的那套。**

> **已修复**：`benchmarks/alphaproteo10/README.md` 和 `docs/binder_benchmark.md` 一度把 `ipAE<7.0 / pLDDT>0.9 / RMSD<1.5` 标为复现 Table 4 应该用的档，按此跑出的数与 Table 4 不可比。两处已改为 `af2_easy`，并附上逐靶点吻合的验证脚本（分支 `bench/alphaproteo-10-targets`，commit `d715f88`）。

### 6.5.4 十靶点的均值被四个简单靶点主导

同样从 `af2_opt` 档算出来的逐靶点通过率（%）：

| 靶点 | PXDesign | BindCraft | BoltzGen | RFDiff-3 | BoltzDesign-1 | ODesign | Protpardelle-1c |
|---|---:|---:|---:|---:|---:|---:|---:|
| BHRF1 | 32.09 | 12.66 | 12.88 | 16.62 | 0.30 | 9.98 | 1.14 |
| PDL1 | 31.16 | 37.17 | 7.13 | 6.07 | 6.82 | 2.07 | 0.57 |
| IR | 12.31 | 25.18 | 11.57 | 7.25 | 3.24 | 4.01 | 0.00 |
| TrkA | 12.06 | 25.03 | 15.40 | 5.55 | 9.42 | 0.75 | 0.04 |
| VEGFA | 4.81 | 1.43 | 1.20 | 0.24 | 0.21 | 0.24 | 0.00 |
| SC2RBD | 3.96 | 6.79 | 0.00 | 1.60 | 0.30 | 0.38 | 0.00 |
| **H1** | 0.04 | 1.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **IL17A** | 0.00 | 0.80 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **IL7RA** | 0.26 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **TNFa** | 0.26 | 0.20 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 |
| **均值** | **9.70** | **11.06** | **4.83** | **3.73** | **2.03** | **1.74** | **0.18** |

两点值得注意：

1. **四个靶点（H1、IL17A、IL7RA、TNFa）对所有方法都接近 0**，最好的也不到 1.1%。在这四个上做出改进很难被这套指标量到
2. **均值几乎完全由 BHRF1、PDL1、IR、TrkA 四个决定**。报"十靶点平均"时要意识到这一点——均值涨了可能只是这四个涨了


---

## 七、怎么用这份材料

**这份文档是配方册，不是选型报告。** 现阶段的目标是尽可能多地收集别人怎么做，而不是提前敲定一个"最严谨"的基准。等真正开跑时，看哪套合适就用哪套——A-CODE 那十个靶点只是其中一种做法。

所以每一条的记法统一成四样，够拿去直接跑：

1. **靶点/案例从哪来**——几个、PDB 或数据集出处、怎么裁、hotspot 怎么定
2. **用什么工具打分**——具体到模型和版本
3. **什么指标什么阈值算成功**——原样抄录，不做换算不做取舍
4. **代码数据能不能拿到**——链接，以及跑起来需要什么

### 已经能直接跑的（我们手上就有）

- **AlphaProteo 十靶点**：`benchmarks/alphaproteo10/`，十个结构已下载、裁剪范围和 hotspot 都过了校验脚本。当前按 A-CODE 口径统一 `binder_length: 105`，每个 yaml 的注释里另记了 AlphaProteo 逐靶点范围，随时可切
- **ProtDBench**：`git clone congliuUvA/ProtDBench`，`data/` 164 MB 已含七个方法的全部设计和分数、以及 23.6 万条带湿实验标签的打分器对比数据。过滤器定义在 `protdbench/protd_configs/eval.py`，五档（见 §7.1）
- **BindCraft**：`martinpacesa/BindCraft`，默认过滤器 18 类 54 条在 `settings_filters/default_filters.json`
- **MotifBench**：`blt2114/MotifBench`，30 题，有排行榜
- **Studio-179**：`DISCO-design/DISCO` 的 `studio-179/` 目录带 SDF

### 7.1 ProtDBench 的五档过滤器（原样抄录）

同一批设计可以同时按五档打分，这是目前见到的最完整的一套并行口径：

| 档 | 等价于谁的惯例 | 条件 |
|---|---|---|
| `af2_easy` | **BindCraft** | pLDDT > 0.8；i_pTM > 0.5；i_pAE < 0.35（归一化）；bound_unbound_RMSD < 3.5 |
| `af2_opt` | **A-CODE / PXDesign** | pLDDT > 0.9；**unscaled_i_pAE** < 7.0；binder RMSD < 1.5 |
| `ptx` | Protenix 全量 | iptm_binder > 0.85；ptm_binder > 0.88；RMSD < 2.5 |
| `ptx_mini` | Protenix-Mini | 同上 |
| `ptx_basic` | Protenix 放宽 | iptm_binder > 0.8；ptm_binder > 0.8；RMSD < 2.5 |

> **要和 A-CODE Table 4 横比，那一列必须锁死 `af2_easy`**——已核实它与 Table 4 逐个靶点分毫不差（§6.5.3）。`af2_opt` 是另一套更严的档，可以并列报，但**不能用它对标 Table 4**。

**MSA 只有 Protenix 那几档要**：`use_msa: True` 在 `eval.py` 里挂在 `ptx` 和 `ptx_mini` 下面，`af2` 块根本没有这个键（它只有 `use_initial_guess`、`use_binder_template`、`use_multimer: False`、`model_ids: [0]`）。PXDesign README 也是同样口径——MSA 是 "Required for 'Extended' mode (Protenix evaluation)"。跑评估需要 AF2、ESMFold、ProteinMPNN 三套权重。

### 7.2 真跑的时候要记住的三件事

不是建议用哪个，是三个已经量化过的坑：

1. **打分器换了，数字和排名都会变**（§6.5.1）。所以报数时必须写明用的是哪一档，不能只写"designability"
2. **ipAE 有两套量纲**（§8.3）。`i_pAE < 0.35` 和 `ipAE < 7.0` 差的是 31 倍归一化，前者其实更宽
3. **同一套口径，不同论文的数也可能对不上**（§6.5.3）。所以自己跑出来的数只跟自己的对照组比，别直接横比别人表里的数字

### 我们目前跑不了的（但照样收录）

训练数据里没有小分子、配体、金属簇，所以酶和配体条件那条线短期内跑不了；抗体/纳米抗体同理。
但按 mentor 的说法将来可能训，**这些照样按同样四栏记全**，只在表里单独标一下当前跑不了。

另外我们本地没有 GPU、模型权重在合作者服务器上，所以这边能交付的是**测试集定义 + 可运行的接入代码 + 跑法说明**，出数在服务器上做。

## 八、核实记录

**这一节是核实过程的流水账**（查了什么、结论是什么、推翻了什么）。
**binder 线当前仍未解决的项已整理到 §9.11**，那里是按 recipe 字段组织的；本节保留过程记录，便于追溯为什么某个结论变过。

1. ~~AlphaProteo 8 还是 10 个靶点~~ —— **已核实：Table S1 有十个，正文的八个是湿实验子集。原先的猜测是对的，见 §8.1**
2. **RAbD 60 vs 55** —— **仍未核实**，PMC 有人机验证挡住了。对我们眼下报数无影响（抗体不是当前主线），留待需要时再查
3. ~~Proteína-Complexa 14 vs 19~~ —— **已核实：19 个**（12 个 easy + 7 个 hard，hard 含 TNF-α、H1、IL17A、VEGFA）。另有 4 个小分子（SAM/OQO/FAD/IAI）和 41 个 AME 酶任务。验证器：蛋白用 AF2-Multimer（经 ColabDesign），小分子用 RoseTTAFold-3
4. BindCraft 12 靶点、RFdiffusion 5 靶点的清单**仍未核实**——bioRxiv 只给摘要、Nature 收费、GitHub README 不含清单。但 BindCraft 的**默认过滤阈值已从源码取得**，见 §8.3
5. ~~ProtDBench 的湿实验数据集规模~~ —— **已解决**：236,246 个设计、12 个靶点、1,485 个真实结合，见 `data/filter_benchmark/`
6. ~~PXDesignBench 与 ProtDBench 的关系~~ —— **已核实**：ProtDBench 的 python 包比 PXDesignBench 多 5 个文件（一个 `post_processing/` 模块），其余完全对应，配置命名空间由 `pxd_configs` 改名为 `protd_configs`，README 中引用 PXDesign。**ProtDBench 是在 PXDesignBench 基础上扩展的**，并额外附带 164 MB 数据
7. SKEMPI 抗体子集 = 53 这个数来自下游设计论文，不是 SKEMPI 本身
8. Riff-Diff 是否真的提出了新 benchmark——Nature 正文里没找到，可能只在 preprint 里

---

### 8.1 AlphaProteo 的"十个靶点"没有引错

**核实结论：AlphaProteo（`arXiv:2409.08022`）的附录 Table S1（p38）包含全部十个靶点**，表题是 "Binder design problem specifications for in silico benchmarking **and** experimental testing"——十个用于 in-silico 基准，其中八个做了湿实验。

正文里的 "we designed binders against eight target proteins" 说的是**湿实验子集**：成功 7 个（BHRF1、SC2RBD、IL-7RA、PD-L1、TrkA、IL-17A、VEGF-A），TNFα 失败。

所以 A-CODE 与 ProtDBench 写的 "as proposed in Zambaldi et al." / "Following prior work (Zambaldi et al., 2024)" **是准确的**，`benchmarks/alphaproteo10/README.md` 也不需要改。

佐证（三处独立一致）：
- 我们 `targets/h1.yaml` 里逐字转录了 Table S1 的原始规格串 `A1-50, A76-80, A107-111, A258-322, B1-68, B80-170`，hotspots `B21, B45, B52`
- 每个 yaml 都记了"AlphaProteo 自己给该靶点的 binder 长度范围"（BHRF1 80–120、H1 40–120、IR 40–120）
- 这些范围与 **ProtDBench Table 3 逐个一致**——说明 ProtDBench Table 3 就是 AlphaProteo Table S1

> 记录一处过程教训：抓 arXiv HTML 只拿到正文，附录表格没进来，据此一度误判为"误引传了三手"。**涉及靶点清单和规格的核实必须落到附录/补充材料**，摘要和正文不够。

### 8.2 binder 长度：不是错，是选了另一套口径

我们十个靶点统一写 `binder_length: 105`，配置注释里说明了原因——**A-CODE 对所有十个靶点统一采样 80–130，而复现 A-CODE Table 4 正是当时的目标**；同时每个 yaml 都把 AlphaProteo 自己的逐靶点范围记在注释里，以便随时切换。所以这是有意的口径选择，不是遗漏。

但要跟 **ProtDBench** 的数字可比，就得换成它那套（= AlphaProteo Table S1 的逐靶点范围）：

| 靶点 | PDB | 逐靶点范围 | A-CODE 口径 | 我们当前 |
|---|---|---|---|---|
| BHRF1 | 2wh6 | 80–120 | 80–130 | 105 |
| SC2RBD | 6m0j | 80–120 | 80–130 | 105 |
| IL-7RA | 3di3 | 50–120 | 80–130 | 105 |
| PD-L1 | 5o45 | 50–120 | 80–130 | 105 |
| TrkA | 1www | 50–120 | 80–130 | 105 |
| IR | 4zxb | 40–120 | 80–130 | 105 |
| H1 | 5vli | 40–120 | 80–130 | 105 |
| IL-17A | 4hsa | 50–140 | 80–130 | 105 |
| TNFα | 1tnf | 50–120 | 80–130 | 105 |
| VEGF-A | 1bj1 | 50–140 | 80–130 | 105 |

从 ProtDBench 放出的设计名（形如 `BHRF1_len100_sample_2`）可以还原它实际怎么扫的：**范围内每一个整数都跑**。BHRF1 是 80,81,…,120 共 41 个长度；IL-17A 和 VEGF-A 是 50…140 共 91 个。不是取几个采样点。

**结论**：现在的单点 105 用于复现 A-CODE 是合理的；若要报 ProtDBench 可比的数，需要加一个逐靶点范围的扫描模式。两套口径都该保留，由命令行选。

### 8.3 ipAE 有两套量纲，直接比数字会得到相反的结论

BindCraft 源码里的默认阈值是 `i_pAE < 0.35`，而 AF2-IG 的标准是 `ipAE < 7.0`。**同名指标，不同量纲**——前者是归一化的，后者是原始埃值。

ProtDBench 的数据同时存了两列（`i_pAE` 和 `unscaled_i_pAE`），实测比值恰好是 **31.000**，即 AlphaFold 的 PAE 上限。所以 `0.35 × 31 ≈ 10.85 Å`。

在同一批 22,720 条 PXDesign 设计上：

| 口径 | 换算成原始埃值 | 通过率 |
|---|---|---:|
| AF2-IG 标准 `ipAE < 7.0` | 7.0 Å | **10.63%** |
| BindCraft 默认 `i_pAE < 0.35` | ≈ 10.85 Å | **21.07%** |

**BindCraft 的默认门槛比 AF2-IG 标准宽了一倍**，通过率差两倍。如果照字面比较"0.35 vs 7.0"，会得出完全相反的结论。

**BindCraft 默认过滤器的完整阈值**（取自 `settings_filters/default_filters.json`，共 18 类、54 个条目，跨 model 1/2 和均值）：

| 指标 | 阈值 | 指标 | 阈值 |
|---|---|---|---|
| pLDDT | > 0.8 | Binder pLDDT | > 0.8 |
| pTM | > 0.55 | Binder RMSD | < 3.5 Å |
| i_pTM | > 0.5 | Hotspot RMSD | < 6 Å |
| i_pAE（归一化） | < 0.35 | Binder Loop 占比 | < 90% |
| Shape Complementarity | > 0.6（均值） | 表面疏水性 | < 0.35 |
| Rosetta dG | < 0 | dSASA | > 1 |
| 界面残基数 | > 7 | 界面氢键数 | > 3 |
| 未成键的埋藏氢键 | < 4 | Binder Energy Score | < 0 |

注意这跟 AF2-IG 的三条式过滤是**两种完全不同的哲学**：AF2-IG 只看结构预测的置信度，BindCraft 还叠加了一整套 Rosetta 界面能量项和物化性质。**同样的设计在两套过滤器下不可能得到可比的通过率。**

---

## 附：怎么来的

四条独立检索线（binder / 抗体与纳米抗体 / 酶与配体 / 通用与评估方法学），统一约束：不许猜数字、没读到就写 unknown、每条必须带可追溯链接。本文档只做广度，阈值等精确数字的逐篇核实是下一步。

---

# 九、配方明细：binder / peptide

每条按统一字段记录，字段缺失就写 `UNRESOLVED`，不填看着合理的数。证据标记见文首。

> **本节状态**：binder / ProtDBench / A-CODE 已做过一轮源码级核实（2026-09-12）。peptide 部分仍以第一轮广度调研为主，标记偏保守。

## 9.0 四条横跨全表的量纲坑

**坑一：ipAE 有两个量纲，差 31 倍。** `CODE_VERIFIED`

ProtDBench `protdbench/tools/af2/main_af2_complex.py:84-85`：

```python
"i_pAE": round(metrics["i_pae"], 2),               # i_pae divdied by 31
"unscaled_i_pAE": round(metrics["i_pae"] * 31, 2), # raw i_pae
```

即 **ColabDesign 交出来的 `i_pae` 本身就是归一化的**，除数是 AlphaFold 的 PAE 上限 31.0；ProtDBench 乘回 31 得到原始埃值。BindCraft 走同一条链路（`functions/colabdesign_utils.py:259` 直接取 `prediction_model.aux["log"]`），所以它的 `0.35` 是归一化值，等价原始 **≈10.85 Å**——**比 `7.0` 松，不是严**。

| 出处 | 写的值 | 原始 Å | 标记 |
|---|---|---|---|
| BindCraft `settings_filters/default_filters.json` | `i_pAE < 0.35` | ≈10.85 | `CODE_VERIFIED` |
| ProtDBench `af2_easy` | `i_pAE < 0.35` | ≈10.85 | `CODE_VERIFIED` |
| ProtDBench `af2_opt` | `unscaled_i_pAE < 7.0` | 7.0 | `CODE_VERIFIED` |
| RFdiffusion README | `pae_interaction < 10` | 10 | `CODE_VERIFIED` |
| RFpeptides（AfCycDesign） | `iPAE < 0.3` / GABARAP 档 `0.13` | ≈9.3 / ≈4.0 | `INFERRED`（除数按 ColabDesign 族推定，未见该文自陈） |

pLDDT 同样分裂：BindCraft / BoltzDesign1 用 0–1；RFdiffusion / Adaptyv / ODesign-AF3 用 0–100。`CODE_VERIFIED`

> ODesign 附录 C.1.2 自陈其阈值"基于 **PXdesign 里的 AF2-IG-easy 过滤器定义**（ipAE<10.85, pLDDT>0.8, ipTM>0.5, binder bound/unbound RMSD < 3.5Å）"。`EXPLICIT_SOURCE`

**坑二：min 和 mean 是两个量。** `EXPLICIT_SOURCE`
Latent-X 的 `min_ipae` 是界面残基对上的**最小值**（原文 "the minimum value across all interchain terms in the predicted alignment error matrix"），BoltzGen 的 `min_design_to_target_pae` 同理；RFdiffusion 和 Adaptyv 的 `pae_interaction` 是**均值**。`< 1` 和 `< 10` 不在同一根轴上。

**坑三：三个基准没有绝对阈值。** `CODE_VERIFIED` / `EXPLICIT_SOURCE`
BoltzGen 只有 RMSD 和氨基酸组成两条硬过滤，置信度全部走排名聚合（`src/boltzgen/task/filter/filter.py`）；BOND-PEP 的成功线是"超过该靶点天然肽自己的 ipTM"，逐靶点浮动；Cao 2022 取分布前 1%。带"阈值"列的表会歪曲这三个。

**坑四：RFdiffusion 的阈值有两套，别混。** README `CODE_VERIFIED`，论文数值 `EXPLICIT_SOURCE`（二手，未核原文）
README 原文：*"We have found that filtering at **pae_interaction < 10** is a good predictor of a binder working experimentally"*。论文 Extended Data Fig. 8F 另给 **单体 pLDDT > 80、interaction pAE < 10、单体 RMSD < 1 Å**。同一篇里**单体和 motif 用全局 `pAE < 5`**，与 binder 的**界面** `pAE < 10` 不是一个量。
> 流传的 `pLDDT>0.9 / ipAE<7.0 / RMSD<1.5` **不是 RFdiffusion 的**，那是 ProtDBench 的 `af2_opt`。

---

## 9.1 ProtDBench / A-CODE / AlphaProteo 十靶点（同一批靶点，三层壳）

这三者共用同一组靶点与规格，差别在过滤档和采样协议，合并成一条 recipe。

| 字段 | 内容 | 标记 |
|---|---|---|
| **task** | 条件式 binder 设计（给定靶点结构 + 裁剪范围 + hotspot，生成 binder 序列与结构） | `EXPLICIT_SOURCE` |
| **test cases** | 10 个靶点：BHRF1 `2wh6`、SC2RBD `6m0j`、IL-7RA `3di3`、PD-L1 `5o45`、TrkA `1www`、IR `4zxb`、H1 `5vli`、IL-17A `4hsa`、TNFα `1tnf`、VEGF-A `1bj1` | `CODE_VERIFIED`（我们仓库十个 cif + ProtDBench Table 3 一致） |
| **case selection** | 出自 AlphaProteo 附录 Table S1（p38），表题 "Binder design problem specifications for in silico benchmarking **and** experimental testing"。正文的"八个靶点"是**湿实验子集**（7 成功 + TNFα 失败） | `EXPLICIT_SOURCE` |
| **input conditioning** | 靶点结构**作为已知条件给定**（裁剪范围 + hotspot 残基列表），不重建靶点 | `EXPLICIT_SOURCE` |
| **expected output** | binder 序列 + 复合物坐标 | `EXPLICIT_SOURCE` |
| **sample count / length（ProtDBench）** | 范围内**每个整数长度** × **4 条骨架** × **8 条序列**；逐靶点 1,312–2,912 条，十靶点合计 22,720。长度范围逐靶点不同：BHRF1/SC2RBD 80–120、IL-7RA/PD-L1/TrkA/TNFα 50–120、IR/H1 40–120、IL-17A/VEGF-A 50–140 | `CODE_VERIFIED`（设计名 `BHRF1_len100_sample_2` 100% 可解析，长度数×4×8 = 实际条数对十个靶点逐个精确吻合） |
| **sample count / length（A-CODE）** | 原文只明确：**每靶点 328–728 条 binder，长度在 80–130 区间内**，同一靶点不同长度的成功数最后**汇总**（Appendix C.2）。<br>**没有**给出逐靶点长度排布、是否均匀采样、每个长度分配多少条 | 区间与汇总方式 `EXPLICIT_SOURCE`；**逐靶点长度排布与样本分配 `UNRESOLVED`** |
| **sequence design route（ProtDBench）** | 每条骨架 8 条序列（`seq_idx` 0–7） | `CODE_VERIFIED` |
| **sequence design route（A-CODE）** | 分两个 variant 各自报数：**模型共设计出的序列**，和 **ProteinMPNN 重设计的单条序列** | `EXPLICIT_SOURCE` |

> **不要因为三者共用十个靶点就把采样/序列协议合并。** ProtDBench 的"每长度 4 骨架 × 8 序列"是它自己复跑的协议，不是 A-CODE 的；A-CODE 的两个 variant 也不对应 ProtDBench 的 8 条序列。

#### 关于 A-CODE 的 328–728：一个算术对应，但只是 inference

`INFERRED`。AlphaProteo 逐靶点长度范围内的整数个数只有四个取值 {41, 71, 81, 91}，乘 8 得 **{328, 568, 648, 728}**——恰好覆盖 A-CODE 所述区间的端点：

| 长度个数 | 靶点 | ×8 |
|---:|---|---:|
| 41 | BHRF1, SC2RBD | **328** |
| 71 | IL7RA, PDL1, TNFa, TrkA | 568 |
| 81 | H1, IR | 648 |
| 91 | IL17A, VEGFA | **728** |

这**暗示** A-CODE 也走了逐靶点整数长度 × 8 条序列。但有反证：A-CODE 写的长度是 **80–130**，AlphaProteo 的逐靶点范围是 40–140，两者对不上。
**两种可能未能裁决**：(a) A-CODE 沿用了 AlphaProteo 的逐靶点范围，正文 80–130 是笼统描述；(b) 算术吻合是巧合。
**只作为 inference 记录，不能当 protocol 用。** `UNRESOLVED`
| **evaluator** | AF2，**单链非 multimer**，`model_ids: [0]`（只用 model 1），`use_initial_guess: True`，`use_binder_template: True`；**af2 块没有 `use_msa` 键** | `CODE_VERIFIED`（`protd_configs/eval.py`） |
| **metric / units** | `pLDDT` 0–1；`i_pTM` 0–1；`i_pAE` 归一化（原始/31）；`unscaled_i_pAE` 原始 Å；`bound_unbound_RMSD` = binder 单独预测 vs 复合物预测里的 binder 链；`af2_binder_pred_design_rmsd` = binder 单独预测 vs **原始设计** | `CODE_VERIFIED`（`tools/af2/main_af2_monomer.py:164,176`） |
| **threshold** | **`af2_easy`（= A-CODE Table 4 用的档）**：`pLDDT>0.8`、`i_pTM>0.5`、`i_pAE<0.35`（≈10.85 Å）、`bound_unbound_RMSD<3.5`<br>**`af2_opt`**：`pLDDT>0.9`、`unscaled_i_pAE<7.0`、`af2_binder_pred_design_rmsd<1.5`<br>**`ptx`/`ptx_mini`**：`iptm_binder>0.85`、`ptm_binder>0.88`、`RMSD<2.5`；**`ptx_basic`** 放宽到 0.8/0.8 | `CODE_VERIFIED` |
| **aggregation** | designability = 通过率（条数占比）；同一靶点不同 binder 长度的成功数汇总 | `EXPLICIT_SOURCE` |
| **diversity / novelty** | 簇级成功率：**用 TMalign 聚类（不是 Foldseek）**，只对**通过该过滤档**的设计聚类，TM 阈值 **[0.6, 0.8, 1.0]** 三档各算一次，`簇中心数 / total_backbones × 100`（**分母是骨架数不是设计数**），默认对 `af2_easy` 档做。另有 24 小时 GPU 预算下的吞吐量 | `CODE_VERIFIED`（`post_processing/config.py:33` 的 `CLUSTER_TM_THRESHOLDS = [0.6, 0.8, 1.0]`；`post_processing/binder.py:237-256, 520-570`） |
| **wet-lab labels** | 这十个靶点本身无逐设计标签。ProtDBench 另带 Cao 湿实验打分表：**236,246 条设计 × 8 打分器，1,485 条阳性**（`data/filter_benchmark/cao_verifier_scores.csv.gz`） | `CODE_VERIFIED` |
| **code / data** | `github.com/congliuUvA/ProtDBench`，`data/` 164 MB 随仓库发布（七个方法的逐条设计分数 + Cao 打分表）。构建于 `github.com/bytedance/PXDesignBench`（python 包多一个 `post_processing/` 模块，配置命名空间 `pxd_configs`→`protd_configs`） | `CODE_VERIFIED` |
| **compute** | 打分需 AF2 + ESMFold + ProteinMPNN 权重（`download_tool_weights.sh`）。生成 22,720 条设计是主要成本 | `EXPLICIT_SOURCE` |
| **Proteo-AA** | `DIRECT_NOW` —— 十个靶点已复现为可运行配置（`benchmarks/alphaproteo10/`，链/裁剪范围/hotspot 经 Latent-X 表 S3 独立比对逐字符一致）。缺的是权重和打分环境，两者都在服务器上 | — |

### 9.1a 哪一档对应 A-CODE Table 4 —— 已用数据裁决

`CODE_VERIFIED`。拿 ProtDBench 放出的逐条分数按两档各算一遍，与 A-CODE Table 4 的 PXDesign 行比：

| | 平均绝对偏差 | 相关系数 | 均值 |
|---|---:|---:|---:|
| **`af2_easy`** | **0.00** | **1.000** | 21.19% |
| `af2_opt` | 11.49 | 0.867 | 9.70% |
| A-CODE Table 4 | — | — | 21.19% |

十个靶点逐个吻合到小数点后两位。复算脚本：`benchmarks/alphaproteo10/verify_filter_protocol.py`（在 `bench/alphaproteo-10-targets` 分支上）。

**A-CODE 附录 C.2 的阈值** = `af2_easy`：`pLDDT>0.80、ipTM>0.50、ipAE<10.85 Å、bound/unbound RMSD<3.5`。`EXPLICIT_SOURCE`
第三方佐证：ODesign 附录 C.1.2 明写同一组值并归因 PXDesign。`EXPLICIT_SOURCE`

> 换档带来的差异远大于换论文：同一批设计 `af2_easy` 21.19% vs `af2_opt` 9.70%，个别靶点差 300 倍（H1 12.08→0.04、IL7RA 29.80→0.26）。`CODE_VERIFIED`

### 9.1b 七个方法在这十个靶点上的实测（af2_opt 档）

`CODE_VERIFIED`，从 `data/generative_benchmark/10_targets/` 算出：

| 方法 | 设计条数 | af2_easy | af2_opt | ptx_mini |
|---|---:|---:|---:|---:|
| PXDesign | 22,720 | **19.88%** | 8.48% | 4.18% |
| BindCraft | 30,992 | `UNRESOLVED`（该列为空） | **12.24%** | 3.60% |
| BoltzGen | 22,720 | 12.27% | 4.49% | **4.43%** |
| RFdiffusion-3 | 22,720 | 8.61% | 3.07% | 1.14% |
| BoltzDesign-1 | 22,720 | 13.08% | 2.06% | 0.16% |
| ODesign | 22,720 | 8.64% | 1.37% | 0.38% |
| Protpardelle-1c | 22,720 | 1.41% | 0.13% | 0.01% |

**排名随档位变化**：af2_easy 下 PXDesign 第 1、BoltzDesign-1 第 2；af2_opt 下 BindCraft 第 1、BoltzDesign-1 掉到第 5；ptx_mini 下 BoltzGen 第 1、BoltzDesign-1 第 6。BoltzDesign-1 通过率跨档相差 80 倍。

**四个靶点对所有方法都接近 0**（H1、IL17A、IL7RA、TNFa，最好的不到 1.1%），十靶点均值几乎由 BHRF1/PDL1/IR/TrkA 四个决定。`CODE_VERIFIED`

---

## 9.2 BindCraft

| 字段 | 内容 | 标记 |
|---|---|---|
| **task** | 幻觉式 binder 设计（AF2 反向优化，非扩散） | `EXPLICIT_SOURCE` |
| **test cases** | Nature 版 12 个靶点：PD-1、PD-L1、IFNAR2 `2LAG`、CD45 `5FMV`、CLDN1（用可溶类似物 sCLDN1）、BBF-14 `9HAG`、CrSAS-6、Der f7 `3UV1`、Der f21 `5YNY`、Bet v1、SpCas9 REC1 `4ZT0`、CbAgo `6QZK`。**v1 预印本只有 10 个**（无 CLDN1、CbAgo）。HER2 `1N8Z` 只在 AAV 重定向那节，不属于基准 | `EXPLICIT_SOURCE`；PD-1/PD-L1/CLDN1/CrSAS-6/Bet v1 的 PDB 编号 `UNRESOLVED` |
| **case selection** | 覆盖细胞表面受体、常见过敏原、de novo 设计蛋白、多结构域核酸酶 | `EXPLICIT_SOURCE` |
| **input conditioning** | 靶点结构 + hotspot（`settings_target/*.json` 的 `target_hotspot_residues`），长度给区间（PDL1 示例 `[65, 150]`） | `CODE_VERIFIED` |
| **sample count / length** | `number_of_final_designs: 100`；README 建议"先出至少 100 条全过滤的设计，再挑前 5–20 条做实验" | `CODE_VERIFIED` |
| **sequence design route** | ProteinMPNN **soluble** 权重 `v_48_020`，温度 0.1，`num_seqs: 20`，`max_mpnn_sequences: 2` | `CODE_VERIFIED` |
| **evaluator** | AF2 **multimer** 做设计，AF2 **单体**模型重预测（3 轮循环、2 个模板模型、单序列模式）+ Rosetta FastRelax(200) + InterfaceAnalyzer | `EXPLICIT_SOURCE` |
| **threshold** | `settings_filters/default_filters.json`，18 类 54 条。核心：pLDDT>0.8（归一化）、pTM>0.55、i_pTM>0.5、**i_pAE<0.35（归一化）**、形状互补>0.6、表面疏水性<0.35、界面氢键>3、未成键埋藏氢键<4、界面残基>7、dSASA>1、dG<0、Binder RMSD<3.5 Å、Hotspot RMSD<6 Å、Loop 占比<90%、界面 K≤3/M≤3。肽预设另有一套（`peptide_filters.json`：i_pTM 0.4、i_pAE 0.3、Binder RMSD 2.5） | `CODE_VERIFIED` |
| **wet-lab labels** | 每靶点 6–53 条设计做实验，命中数在论文 Fig. 1b | `EXPLICIT_SOURCE` |
| **code / data** | `github.com/martinpacesa/BindCraft`（MIT），设计模型 Zenodo `10.5281/zenodo.14249738` | `CODE_VERIFIED` |
| **compute** | CUDA GPU，README 建议 32 GB+ 显存；AF2 权重约 5.3 GB；**PyRosetta（非学术需商业授权）**、DSSP、DAlphaBall | `EXPLICIT_SOURCE` |
| **Proteo-AA** | `NEEDS_ADAPTER` —— 靶点+hotspot 的条件化方式与我们一致，但它的过滤器叠了一整套 Rosetta 界面能量项，我们要么补这些计算、要么只报它的 AF2 子集并说明 | — |

> ⚠️ **v1 预印本正文写 `i_pAE > 0.35`，符号写反了**；Nature 版和仓库 JSON（`higher: false`）都是 `<`。`CODE_VERIFIED`

## 9.3 BoltzGen

| 字段 | 内容 | 标记 |
|---|---|---|
| **task** | 通用 binder 生成（蛋白、纳米抗体、肽、小分子靶点） | `EXPLICIT_SOURCE` |
| **test cases** | **10 个低同源新靶点**（只给 PDB 编号）：`1G13 1JQD 1NB0 2A1X 2PNY 3APU 3CH4 3QKG 6M1U 7AAH`，与仓库 `example/hard_targets/` 一一对应；另 5 个简单靶点 TNFα、PD-L1、PDGFR `3MJG`、IL-7Rα、InsulinR `4ZXB` | `EXPLICIT_SOURCE` + `CODE_VERIFIED` |
| **case selection** | PDB 单体生物组装 + 仅单体的 30% mmseqs 簇 + Sino Biological 目录可购。与任何非单体链的最大序列同一性 0.0–27.9% | `EXPLICIT_SOURCE` |
| **sample count** | 每个新靶点 6 万条纳米抗体 + 6 万条蛋白，长度 80–140（简单靶点 80–120）；**每靶点各取前 15 条**送实验 | `EXPLICIT_SOURCE` |
| **evaluator** | **Boltz-2**，权重 `boltz2_conf_final.ckpt`，`recycling_steps:3`、`sampling_steps:200`、`diffusion_samples:5`、给靶点模板、**不用 MSA** | `CODE_VERIFIED` |
| **threshold** | **只有 RMSD 和组成两条硬过滤**：refolding RMSD ≤ **2.5 Å**（肽协议 2.0）、**设计单独重折** RMSD ≤ 2.5 Å、`CYS_fraction ≤ 0`、单一氨基酸占比 ≤ 0.3。**无任何绝对置信度阈值**——ipTM/pAE/氢键/ΔSASA 只做排名聚合 | `CODE_VERIFIED`（`src/boltzgen/task/filter/filter.py`） |
| **aggregation** | 加权排名取最差排名，再做质量-多样性贪心筛选（`alpha` 默认 0.001） | `CODE_VERIFIED` |
| **wet-lab labels** | 8 场 campaign，26 个靶点；排名权重在 **Cao 2022 的 11,000 条设计、11 个靶点**上标定 | `EXPLICIT_SOURCE` |
| **code / data** | `github.com/HannesStark/boltzgen`，`pip install boltzgen`，权重约 6 GB 自动下载 | `CODE_VERIFIED` |
| **Proteo-AA** | `NEEDS_ADAPTER` —— 它的"设计不带靶点单独重折"这一项我们没有对应实现；靶点条件化本身兼容 | — |

> 值得借鉴的一项：**设计单独重折**——"它离开靶点还折得起来吗"。AF2-IG 那套没有这个检查。

## 9.4 RFdiffusion

| 字段 | 内容 | 标记 |
|---|---|---|
| **task** | 扩散式 binder 骨架生成 | `EXPLICIT_SOURCE` |
| **test cases** | 5 个：流感 A H1 血凝素、IL-7Rα、PD-L1、胰岛素受体、TrkA（后四个出自 Cao 2022） | `EXPLICIT_SOURCE`（二手，未核原文） |
| **sample count** | 每靶点约 1 万条骨架 × 2 条序列 ≈ 2 万条设计；每靶点选 95 条做实验。README 补充"有些靶点约 1000 条骨架可能就够" | `EXPLICIT_SOURCE` |
| **sequence design route** | ProteinMPNN-FastRelax，每骨架 2 条序列 | `EXPLICIT_SOURCE` |
| **evaluator** | **AF2 带 initial guess 和靶点模板**，脚本 `github.com/nrbennet/dl_binder_design` | `CODE_VERIFIED`（README） |
| **threshold** | README：`pae_interaction < 10`（"过不了的设计不值得下单"）。论文 Extended Data Fig. 8F：单体 pLDDT > 80、interaction pAE < 10、单体 RMSD < 1 Å | README `CODE_VERIFIED`；论文数值 `EXPLICIT_SOURCE`（二手） |
| **wet-lab labels** | 总体实验成功率 19% | `EXPLICIT_SOURCE` |
| **code / data** | `github.com/RosettaCommons/RFdiffusion`；设计结构、AF2 模型、实测数据 figshare | `CODE_VERIFIED` |
| **Proteo-AA** | `NEEDS_ADAPTER` —— 它是骨架生成 + 独立序列设计的两段式，我们是 co-design；要比的话得决定是否也走 ProteinMPNN 重设计那一路 | — |

## 9.5 Latent-X

| 字段 | 内容 | 标记 |
|---|---|---|
| **test cases** | 7 个湿实验靶点，规格完整（表 S3）：MDM2 `4hfz` A26-108、MCL-1 `2pqk`、PD-L1 `5o45` A17-132、BHRF1 `2wh6` A2-158、IL-7Rα `3di3` B17-209、SC2RBD `6m0j` E333-526、TrkA `1www` X282-382。另有 200 结构的 in-silico 集 | `EXPLICIT_SOURCE` |
| **case selection**（in-silico 集） | 2023-11-24 之后发布的 PDB（晚于相关模型训练截止）；排除 NMR；分辨率<3 Å；无核酸；所有链解析>90%；排除 C2/D2 对称；40% 同源簇取一代表；**每靶点自动选 3 个不重叠表位** → 200×3×100 = 6 万条 | `EXPLICIT_SOURCE` |
| **evaluator** | **Chai-1**（主），Boltz-2 复现。不用 AF3 的原因论文写明"由于商业限制" | `EXPLICIT_SOURCE` |
| **threshold** | 按验证器分别网格搜索（在 Cao 数据上调的）：AF3 `min_ipae<1.5 / ptm_binder>0.8 / complex_rmsd<2.5`；**Chai-1 `<1 / >0.9 / <2`**；Boltz-2 `<1 / >0.95 / <2.5`。大环肽**整个丢掉 `ptm_binder`** | `EXPLICIT_SOURCE` |
| **code / data** | **无**——商业模型，代码权重都不公开 | `EXPLICIT_SOURCE` |
| **Proteo-AA** | `OUT_OF_SCOPE`（不可复现）；但它的**靶点规格独立验证了我们的配置**——五个重叠靶点的链、裁剪范围、每个 hotspot 与我们逐字符一致 | `CODE_VERIFIED` |

## 9.6 ODesign

| 字段 | 内容 | 标记 |
|---|---|---|
| **test cases** | 从 Cao 2022 取的靶点。**论文自相矛盾**：正文 §3.1 和附录 C.1.2 写 "eleven"，图 2a 说明写 "ten"。只点名 EGFR 和 CD3d | `UNRESOLVED` |
| **sample count** | ODesign 与 RFdiffusion 每靶点 100 条骨架 × 8 条 ProteinMPNN 序列 = 800 候选；BoltzDesign1/BindCraft 因成本只跑 10 条 | `EXPLICIT_SOURCE` |
| **evaluator** | **AlphaFold3，只给靶点的 MSA**（binder 的 MSA 和模板都排除）；缺 MSA 的配体案例用 Chai-1 + ESM 嵌入 | `EXPLICIT_SOURCE` |
| **threshold** | `ipAE < 10.85`、**`pLDDT > 80`**、`ipTM > 0.5`、复合物 RMSD < 2.5 Å。论文特意解释"pLDDT 阈值设成 80 而不是 0.8，因为 AlphaFold3 的 pLDDT 不做归一化" | `EXPLICIT_SOURCE` |
| **aggregation** | **单张 H100 上 24 小时内通过过滤的骨架数**——同时是个算力基准 | `EXPLICIT_SOURCE` |
| **code / data** | `UNRESOLVED`（论文正文未见仓库地址，只有网页服务） | — |
| **Proteo-AA** | `NEEDS_ADAPTER` —— 它把 BindCraft 的 binder bound/unbound RMSD<3.5 换成**复合物 RMSD<2.5**，是不同的量，接的时候别互换 | — |

## 9.7 Cao et al. 2022 —— 领域的共享底座

| 字段 | 内容 | 标记 |
|---|---|---|
| **test cases** | **12 个天然蛋白 / 13 个靶点位点**（EGFR 取两个位点）：TrkA、FGFR2、EGFR(域 I/III)、PDGFR、胰岛素受体、IGF1R、TIE2、IL-7Rα、CD3δ、TGFβ、流感 A H3 血凝素、VirB8 | `EXPLICIT_SOURCE`（二手） |
| **sample count** | 每位点 1.5 万–10 万条设计，binder 长度 50–65 残基 | `EXPLICIT_SOURCE`（二手） |
| **evaluator** | 无深度学习验证器——**RIFDock + Rosetta**；筛选靠**酵母表面展示 + 多轮 FACS** | `EXPLICIT_SOURCE`（二手） |
| **metric** | 实验指标 **SC₅₀**（50% 表达细胞被收集时的靶点浓度）；设计阶段排序用 Rosetta ddG 和 contact molecular surface | `EXPLICIT_SOURCE`（二手） |
| **threshold** | **硬门槛**（`cao_2021_protocol/final_filtering.ipynb`，全部须通过）：`ddg ≤ −30`、`contact_molecular_surface ≥ 450`、`score_per_res ≤ −2.4`、`mismatch_probability ≤ 0.1`、`sap_score ≤ 35`、`binder_delta_sap ≥ 12`（`ss_sc ≥ 0.77` 在脚本里被注释掉，标为 optional）。<br>过完硬门槛再做**多指标百分位排序**（`top_x_by_multiple`：二分搜索一个百分位，使各项 top-X 的交集恰好等于要订的条数），排序项 `ddg`（越低越好）、`contact_patch`、`target_delta_sap`、`contact_molec_sq5_apap_target`（越高越好）。<br>更早的 motif 选择阶段另有 `motif_extraction.py -ddg_threshold` 默认 **−20** | `CODE_VERIFIED`（2026-09-12 下载 `scripts_and_main_pdbs.tar.gz` 核对） |
| **wet-lab labels** | 这是**领域里最大的公开 binder 标签集**。ProtDBench 的 Cao 打分表含 **236,246 条设计 × 8 打分器，1,485 条阳性**（12 个靶点） | `CODE_VERIFIED` |
| **code / data** | 六个 tar.gz：`files.ipd.uw.edu/pub/robust_de_novo_design_minibinders_2021/supplemental_files/`。`scripts_and_main_pdbs.tar.gz` 62 MB，含 `cao_2021_protocol/`（RosettaScripts XML、motif 抽取脚本、`final_filtering.ipynb`）。**XML 里的 `Ddg` 和 `ContactMolecularSurface` 都带 `confidence="0"`，在 RosettaScripts 里意为只报告不过滤**——真正的筛选在 notebook 里 | `CODE_VERIFIED` |
| **Proteo-AA** | `OUT_OF_SCOPE` 作为生成基准（设计已存在），但作为**打分器校准数据**是 `DIRECT_NOW`——§6.5.2 那组 AUC 就是拿它算的，不需要任何湿实验能力 | — |

> 为什么它重要：BoltzGen 在它上面标定排名权重、Latent-X 在它上面网格搜索三个验证器的阈值、ODesign 从它取基准靶点、Adaptyv 拿它当序列排除库。

## 9.8 Adaptyv EGFR 竞赛（两轮）

| 字段 | 内容 | 标记 |
|---|---|---|
| **task** | 众包 binder 设计竞赛，**前瞻性**（先提交后做实验） | `EXPLICIT_SOURCE` |
| **test cases** | 单靶点 **EGFR**。R1 201 条设计 / 461 条重复测量；R2 **402 条设计** / 953 条重复，另放出全部 400 条的 AF2 结构 | `CODE_VERIFIED`（仓库 CSV 行数） |
| **evaluator** | R1：ColabFold `1.5.5-cuda12.2.2`，**2 模型、5 轮循环、无 initial guess、无模板**<br>R2：**5 模型、3 轮循环、3 种子、带模板**，仍无 initial guess | `EXPLICIT_SOURCE` |
| **metric** | `pae_interaction` 定义指向 `nrbennet/dl_binder_design/af2_initial_guess/predict.py#L197`——跨靶点-binder 残基对的 **PAE 均值，原始 Å**；`plddt` 只对 binder 链取平均（0–100） | `EXPLICIT_SOURCE` |
| **threshold** | **不发布通过线**，全部作排序。硬门槛两条：表达量 **< 0.02 µg/mL 排除**；与已发表序列距离 **< 10 个氨基酸排除**（库含 SwissProt、THPdb、USPTO、Cao 2022，R2 加 R1 序列） | `EXPLICIT_SOURCE` |
| **wet-lab labels** | **BLI 测的 K_D**。R2 另加中和系数 | `EXPLICIT_SOURCE` |
| **code / data** | `github.com/adaptyvbio/egfr_competition_1` 和 `_2`，代码 Apache-2.0、**数据 ODbL** | `CODE_VERIFIED` |
| **Proteo-AA** | `NEEDS_ADAPTER` 作为生成任务（单靶点、需自己下单做实验才有新标签）；作为**已有标签数据**是 `DIRECT_NOW` | — |

> ⚠️ 两轮的 BLI 缓冲液和浓度范围不同（R1 PBS-T + 0.02% BSA、316.2→10 nM；R2 HBS-T + 0.5% BSA、1000→10 nM），**跨轮比较 K_D 不干净**。`EXPLICIT_SOURCE`

## 9.9 BoltzDesign1

| 字段 | 内容 | 标记 |
|---|---|---|
| **task** | 反演 Boltz-1 做通用 binder 设计（蛋白、小分子、核酸、金属） | `EXPLICIT_SOURCE` |
| **test cases** | `UNRESOLVED` —— 仓库只有单个示例（`7v11`/OQO、`5zmc` DNA），论文靶点表未取得（bioRxiv 反复限流）。第三方信息：ODesign 在同样 11 个 Cao 靶点上跑了它，每靶点 10 条 | — |
| **evaluator** | 设计环用 **Boltz-1**（默认只用 distogram，`--recycling_steps 0`）；验证用 **AlphaFold3**（需自行安装） | `CODE_VERIFIED` |
| **threshold** | **只有两条**（`boltzdesign.py:488`）：`i_ptm > 0.5` 且 `complex_plddt > 0.7`（**归一化 0–1，且是复合物 pLDDT 不是 binder pLDDT**）。**无 ipAE、无 RMSD、无 Rosetta 项** | `CODE_VERIFIED` |
| **Proteo-AA** | `NEEDS_ADAPTER` | — |

## 9.10 peptide 那一支

第一轮广度调研结果，**尚未做源码级核实**，标记从保守。

| 名称 | 案例数 | 验证器 / 判据 | 标记 | Proteo-AA |
|---|---:|---|---|---|
| **PepBench / PepGLAD** | 训练 4,157 / 验证 114 / **测试 93**（LNR） | **无结构预测验证器**——对着晶体肽打分：Cα RMSD 分档 ≤2/≤5/≤10 Å、AAR、**PyRosetta ΔG<0** | `EXPLICIT_SOURCE`（Zenodo README） | `NEEDS_ADAPTER` |
| **BOND-PEP** | 193 对 | AF-Multimer 经 ColabFold；**成功 = 超过该靶点天然肽自己的 ipTM**（相对阈值，逐靶点浮动） | `EXPLICIT_SOURCE` | `NEEDS_ADAPTER` |
| **DiffPepBuilder** | 30 个（PDB 编号已记录在 §9.10a） | 无结构预测验证器；L-RMSD、Rosetta ddG、**有效率 = ddG<0 比例** | `EXPLICIT_SOURCE` | `NEEDS_ADAPTER` |
| **RFpeptides** | 4 个靶点（MCL1、MDM2、GABARAP、RbtA） | **AfCycDesign**（带环状位置编码的 AF2）；iPAE<0.3 归一化 + Cα RMSD<1.5，GABARAP 档收紧到 0.13 | `EXPLICIT_SOURCE`；除数 31 是 `INFERRED` | `FUTURE`（大环肽需环状位置编码） |

### 9.10a DiffPepBuilder 的 30 个 PDB

`1BJR 1J7Z 1RJK 1SJH 2A4R 2AQ9 2BBA 2FTS 2IZX 3EQS 3H0A 3ZQI 4ERZ 4GQ6 4K0U 4P6X 4QJR 4R1E 4RRV 5IZU 5LY1 5LY3 5N8B 5UL6 5V1Y 5WUK 6H7B 6JJZ 6MA3 6S07` `EXPLICIT_SOURCE`

---

## 9.11 binder 线未解决项（本线已暂停）

**binder / ProtDBench / A-CODE 的核实到此告一段落**（2026-09-12）。下面是明确留着的口子，都不影响现有结论的可用性。

### 本轮解决掉的

| 项 | 结论 |
|---|---|
| ProtDBench 簇级成功率的聚类参数 | **TMalign**（不是 Foldseek），TM 阈值 `[0.6, 0.8, 1.0]`，只聚通过档的设计，分母是 backbone 数 `CODE_VERIFIED` |
| `ptx` / `ptx_mini` 是否同一次运行 | **不是**，两次独立 `protenix_predict()`；全量 Protenix 默认 `False`。五档需要三次模型运行 `CODE_VERIFIED` |
| Cao 2022 的 ddG / CMS 阈值 | 已下载核对，见 §9.7 的 threshold 行 `CODE_VERIFIED` |

### 仍未解决（优先级从高到低）

| 项 | 状态 | 为什么先放着 |
|---|---|---|
| A-CODE 逐靶点长度排布与样本分配 | `UNRESOLVED` | 原文只给区间和汇总方式。328–728 与逐靶点长度数×8 的算术吻合是 `INFERRED`，且与"80–130"这一表述相冲突 |
| ODesign 10 还是 11 个靶点、未点名的那九个 | `UNRESOLVED` | 论文正文与图注自相矛盾，无法从外部裁决 |
| ODesign 的代码地址 | `UNRESOLVED` | 正文只给网页服务 |
| BoltzDesign1 的靶点清单 | `UNRESOLVED` | bioRxiv 反复限流；仓库只有单例 |
| BindCraft 五个靶点的 PDB 编号（PD-1、PD-L1、CLDN1、CrSAS-6、Bet v1） | `UNRESOLVED` | Nature Data Availability 未列全 |
| RFdiffusion 论文 Extended Data Fig. 8F 的阈值 | 二手 | Nature 付费墙。README 的 `pae_interaction < 10` 已 `CODE_VERIFIED`，够用 |
| BoltzGen 除前两场外的 26 靶点完整名单 | `UNRESOLVED` | 论文未以单一清单给出；非当前 P0 |
| RFpeptides 的 iPAE 除数是否为 31 | `INFERRED` | 按 ColabDesign 族推定，该文未自陈 |

# 十、配方明细：抗体 / 纳米抗体

## 10.0 范式框架的核实结果

**"抗体基准基本是 redesign 而非 de novo"这个判断成立**，但有四处例外要标出来：

1. **Germinal 和 EasyNano 是真正的 de novo**——没有原生可恢复，靠 AF3/ESMFold2 置信度 + Rosetta 理化 + 湿实验评判，而且 **Germinal 有硬性通过阈值**（以 YAML 形式随仓库发布）
2. **ProteinBench 把自洽引进了 redesign 基准**：它的 `scRMSD` 用 IgFold 折叠**设计出的序列**，再和**模型生成的骨架**比 RMSD——就是把 de novo 那套自洽思想套到 CDR-H3 上。论文明说光看 AAR/RMSD "会严重误导"
3. **AbBiBench、FLAb、NbBench 根本不打恢复率**——它们算的是**模型似然与实验测定值的 Spearman 相关**，属于性质预测不是设计
4. **全表只有四处存在真正的阈值**：Germinal 的 YAML 级联、IgGM 的 `DockQ > 0.23`、DiffAb 的 IMP%（ΔΔG<0）、EasyNano 的相对 Δ ipTM>0.1。**其余全部是只排名不设线**——写配方册时要讲明，免得有人自己发明一条通过线

## 10.1 RAbD —— 领域复用最多的抗体案例集

`PLOS Comput Biol 2018`

**原始是 60 个不是 55 个**（46 κ / 14 λ），取自 **2017 年 8 月版 PyIgClassify**。筛选：分辨率 ≤ 2.5 Å、埋藏面积 > 700 Å²、CDR1 和 CDR2 在簇质心 40° 内、轻重链可变域**都**与 CDR 有接触、非冗余。

**ML 圈用的 55 从哪来**（ProteinBench 附录 B.2.3 原文）：`2ghw` 和 `3uzq` 缺轻链、`3h3b` 缺重链、`5d96` 链 ID 标注错误、`4etq` 因 HERN 报错被排除 → 60 − 5 = **55**。

**案例清单是机器可读的，两处，都恰好 60 个且完全一致**：
- `THUNLP-MT/dyMEAN` 的 `configs.py` → `RAbD_PDB`（60 个小写 PDB 编号）
- `THUNLP-MT/MEAN` 的 `summaries/rabd_summary.jsonl`（60 行，带链标注）

完整 60 个：`1a14 1a2y 1fe8 1ic7 1iqd 1n8z 1ncb 1osp 1uj3 1w72 2adf 2b2x 2cmr 2dd8 2ghw 2vxt 2xqy 2xwt 2ypv 3bn9 3cx5 3ffd 3h3b 3hi6 3k2u 3l95 3mxw 3nid 3o2d 3rkd 3s35 3uzq 3w9e 4cmh 4dtg 4dvr 4etq 4ffv 4fqj 4g6j 4g6m 4h8w 4ki5 4lvn 4ot1 4qci 4xnq 4ydk 5b8c 5bv7 5d93 5d96 5en2 5f9o 5ggs 5hi4 5j13 5l6y 5mes 5nuz`

> ⚠️ **三个条目在 SAbDab 里标注是错的**（`2ghw`、`3uzq`、`3h3b`）。dyMEAN 用手工修过的 PDB 补上并硬编码链覆盖；ProteinBench 直接丢掉。自己写 loader 得到 57 而不是 60，原因就在这。
> dyMEAN 的下载脚本对纳米抗体直接返回 None（注释：`we do not handle nanobodies`）。

**阈值：无。** 只排名。

## 10.2 dyMEAN 三任务协议

`arXiv:2302.00203`，`github.com/THUNLP-MT/dyMEAN`

一条命令建全部三个任务：`bash scripts/data_preprocess.sh all_structures/imgt all_data`（约 1 小时，约 5 GB 输出）。需要 SAbDab IMGT 结构 + 仓库自带的 `sabdab_summary.tsv`（快照日期 **2022-11-12**）。

- **任务一 CDR-H3 设计**：训练 SAbDab（**3,256 条抗体 / 1,644 簇**，验证 365/182），MMseqs2 按 **40% CDR-H3 同一性**聚类，9:1 划分，与测试集共簇的丢掉。测试 = **RAbD 60**。指标：AAR、CAAR、Cα RMSD、TM-score、lDDT、DockQ
- **任务二 复合物结构预测**：测试 = IgFold 测试集。**仓库 `IGFOLD_TEST_PDB` 有 70 个 PDB，论文说 51 个复合物**——差异来自 `--filter 111` 和解析失败。两个数都真实，引用时要说清是哪一阶段
- **任务三 亲和力优化**：测试 = SKEMPI V2.0 抗体子集。**仓库 `SKEMPI_PDB` = 53 个 PDB**；论文没写过滤后的数（我们此前记的"53"来自这里，不是 SKEMPI 本身）。指标 ΔΔG 和 ΔL，每条抗体生成 100 个候选取 top-1

**工具**：TMscore 需自行编译（`g++ -static -O3 -ffast-math -lm -o evaluation/TMscore evaluation/TMscore.cpp`）；DockQ 用 `bjornwallner/DockQ`；ΔΔG 预测器随仓库自带（`evaluation/ddg/data/model.pt`）。
**三个案例清单都在 `configs.py` 里，机器可读。阈值：无。**

## 10.3 DiffAb —— 那 19 个复合物具体是哪些

`github.com/luost26/diffab`

**仓库不带清单，是运行时按抗原名筛出来的**：
```python
TEST_ANTIGENS = ['sars-cov-2 receptor binding domain', 'hiv-1 envelope glycoprotein gp160',
                 'mers s', 'influenza a virus', 'cd27 antigen']
```
在仓库自带的 `sabdab_summary_all.tsv`（13,072 行）上跑这个过滤，**得到正好 19 条、涉及 11 个 PDB**：

| 条目 | 抗原 | 分辨率 |
|---|---|---|
| 5tl5_H_L_A | cd27 | 1.8 |
| 5tlj_B_A_X, 5tlj_D_C_X | cd27 | 3.5 |
| 5tlk_B_A_X, 5tlk_D_C_X, 5tlk_F_E_Y, 5tlk_H_G_Y | cd27 | 2.7 |
| 5w9h_B_C_A, 5w9h_E_F_D, 5w9h_H_I_G | mers s | 4.0 |
| 5xku_C_B_A | influenza a | 1.78 |
| 7bwj_H_L_E | SARS-CoV-2 RBD | 2.85 |
| 7chb_H_L_R | SARS-CoV-2 RBD | 2.4 |
| 7che_A_B_R, 7che_H_L_R | SARS-CoV-2 RBD | 3.416 |
| 7chf_A_B_R, 7chf_H_L_R | SARS-CoV-2 RBD | 2.674 |
| 7d6i_B_C_A | SARS-CoV-2 RBD | 3.41 |
| 8ds5_C_B_A | cd27 | 1.926 |

（cd27 8 个、SARS-CoV-2 RBD 7 个、mers 3 个、流感 1 个。注意 `_load_structures` 还会丢掉 Biopython 解析失败的，实跑可能少于 19。）

**有一个真阈值**：**IMP% = 设计 CDR 的结合能低于原生 CDR 的比例**，即 **ΔΔG < 0**，由 Rosetta `InterfaceAnalyzerMover` 算（`set_pack_separated(True)`，读 `dG_separated`）。
评估需 PyRosetta 和 Ray。**建议把上面 19 个 ID 固化成自己的 CSV。**

## 10.4 ProteinBench 抗体分支

`arXiv:2409.06744`，`proteinbench.github.io`

**55 个案例**（RAbD 减去上面五个），任务只有 **CDR-H3 重设计**。
训练/验证：SAbDab IMGT，要求重链+轻链+蛋白抗原，去重后 MMseqs2 **40%** CDR-H3 聚类，含 RAbD 复合物的簇移除，9:1 → **1786 训练 / 193 验证**。能采样的方法每个抗原生成 **64 条 CDR-H3**。

**13 个指标分四组**（结果表机器可读：`huggingface.co/spaces/proteinbench/ProteinBench` 的 `data/antibody_design.csv`）：
- 准确性：AAR、RMSD（CDR-H3 的 Cα，**不做对齐**）、TM-score（只算 CDR-H3，用 TMalign）
- 功能性：Binding Energy（Rosetta 侧链 pack → 只最小化 H3 侧链 100 步 → InterfaceAnalyzer）
- 特异性：SeqSim-outer/inner、PHR（H3 疏水残基占比）
- 合理性：CN-score（生成肽键长的 KDE 密度 vs 天然，天然均值 **1.3310 Å**）、Clashes（非成键 Cα–Cα < **3.6574 Å**，阈值由 RAbD 的 H3 统计导出）、SeqNat（**AntiBERTy** pLL）、Total Energy（Rosetta **REF15**）

**scRMSD 的确切做法**：用 **IgFold** 预测设计序列的结构（**同时输入 H 和 L 两条链**，并把真实的非 H3 抗体结构作为模板）→ 把非 H3 重链区 Kabsch 对齐到真实结构并把变换应用到预测的 H3 → **在抗原存在下用 Rosetta relax 预测的 H3，5 次独立运行 × 200 步，取能量最低的** → scRMSD = 该结构与**模型生成的** H3 骨架的 Cα RMSD。
**噪声下限已给出**：IgFold 对天然结构的 Cα-RMSD 是 relax 前 **1.95 Å**、relax 后 **1.77 Å**——1.77 就是"RAbD（天然）"那一行，实际可达的下限。

**阈值：无**，但表里有一行 `RAbD (natural)` 作为金标参照。55 个的清单不是机器可读的（自己从 60 减去那五个）。

## 10.5 CHIMERA-Bench

`arXiv:2603.13431`，`github.com/mansoor181/chimera-bench`

**2,922 个复合物 / 2,721 个 PDB**（2,485 蛋白抗原 / 437 肽抗原，中位分辨率 2.72 Å）。
管线：`20,509 SAbDab → 9,171（分辨率≤4.0 Å、配对 VH/VL、蛋白或肽抗原）→ 2,981（MMseqs2 95% CDR-H3 同一性、80% 覆盖）→ 2,922（ANARCI 可编号、保守残基、CDR 完整、骨架质检）`。59 个排除项记在 `metadata/excluded_complexes.csv`。

**三种划分，都是簇不相交**：

| 划分 | 训练 | 验证 | 测试 | 考察什么 |
|---|---:|---:|---:|---|
| `epitope_group`（主） | 2,338 | 292 | 292 | 未见过的表位模式 |
| `antigen_fold` | 2,338 | 292 | 292 | 未见过的抗原折叠 |
| `temporal` | 2,337 | 292 | 293 | 前瞻，按沉积日期（2023 后） |

表位聚类规则原文：*"两个复合物属于同一簇当且仅当它们排序后的表位残基标识（链、位置）完全相同，这天然地处理了不连续表位。"*

**12 个指标分五组**：AAR、CAAR（限定在 paratope）、PPL；Kabsch Cα-RMSD、TM-score；Fnat、iRMSD、DockQ；表位 precision/recall/**F1**；`n_liabilities`（**NG、DG、DS、DD、NS、NT、M** 基序计数）。另有两个复合指标 CHIMERA-S 和 CHIMERA-B。
> ⚠️ **它故意用了两套接触定义**：数据集标注、paratope mask、CAAR 用 **4.5 Å 重原子**；Fnat/iRMSD/DockQ/表位 F1 用 **8.0 Å Cα–Cα**（在天然和预测上对称计算）。

**案例清单是全表最规范的**：`splits/{epitope_group,antigen_fold,temporal}.json` 直接给 `{"train":[...],"val":[...],"test":[...]}`；`metadata/final_summary.csv` 2,922 行 × 32 列。数据在 HF `mansoorbaloch/chimera-bench` 和 Zenodo DOI 20598827，`sample_data/`（12 个复合物）已提交，不下载也能跑测试。
纯 Python 打分（numpy/scipy/torch），TM-score 是 Kabsch 近似，README 建议正式发表时改用官方 TMscore 程序。**阈值：无。** 11 个基线方法用同一套设置重训过。

## 10.6 IgGM —— 60 抗体 + 27 纳米抗体，清单已找到

`ICLR 2025`，`github.com/TencentAI4S/IgGM`

**仓库里没有清单，但 Zenodo 上有**：`https://zenodo.org/records/13790269` 的 `IgGM_Test_set.tar.gz`（5.1 MB）里含两个 `prot_ids.txt`。

**SAb-23-H2-Nano（27 个纳米抗体）**——这是目前**最大的、带结构的公开纳米抗体案例清单**：
`8q94_C_NA_A 8fcz_C_NA_A 7z1x_C_NA_A 8q7s_L_NA_J 8ee2_G_NA_E 8h91_D_NA_B 8q7s_E_NA_D 8h5t_B_NA_A 8aix_D_NA_K 8sk5_B_NA_A 7y0o_B_NA_A 8hr2_B_NA_A 8elq_B_NA_A 7zau_D_NA_C 8hbj_E_NA_C 7ymh_D_NA_A 8q95_B_NA_A 7z1x_B_NA_A 8hr2_C_NA_A 8eln_G_NA_E 8b18_B_NA_A 8g0w_D_NA_B 8h5u_D_NA_C 8h64_D_NA_C 8c9x_H_NA_C 8c5h_N_NA_S 8q93_C_NA_A`

ID 格式 `{pdb}_{H}_{L}_{Ag}`，**纳米抗体的轻链槽位填 `NA`**。压缩包还带天然 FASTA、天然 PDB，以及预先挖空的设计 FASTA（抗体 7 种变体、纳米抗体 4 种，挖空位置用 `X` 标出）。
SAb-23-H2-Ab 那 60 个也在同一压缩包里。

**构造**：训练用 2022-12-31 之前的全部抗体结构（6,448 配对链 + 1,907 单链复合物）；测试用 **2023-06-30 到 2023-12-30** 之间发布的，且与训练相似的序列已剔除。
**有一个真阈值**：结构预测的**成功率定义为 `DockQ > 0.23`**。CDR-H3 设计报骨架 RMSD + AAR（36% 恢复率，比 dyMEAN 高 22.4%）。
**注意这是 redesign 不是 de novo**——框架序列始终是给定的。

## 10.7 nanoFOLD

`bioRxiv 2025.04.29.651236`

**43 个晶体结构测试 + 1064 个模型测试**，但**具体是哪些不知道**——论文没给清单，也没有公开仓库。
来源池：435 个非冗余（90% 同一性）PDB 纳米抗体结构、21,276 个 NanoBodyBuilder2 模型（取自 INDI 的一千万里）。
**训练/测试是否分离论文自己都说不清**：原文承认"很难排除某些晶体结构和模型是否在 ESM-IF 乃至 AntiFold 的训练集里"。

任务是**逆折叠**，指标是恢复率（整个 VHH 可变区 + 只看 CDR-H3）。可引的数字只有一条：nanoFOLD-model 在 1064 模型测试上 **可变区 75% / CDR-H3 37%**。
> ⚠️ 完整的方法对比表两次独立抽取给出了互相矛盾的数字，**不要二手引用**，要用得自己读 PDF。

**权重仅供非商业组织非商业使用，无公开代码，清单不可机读。**

## 10.8 NbBench

`arXiv:2505.02022`，`github.com/ZHymLumine/NbBench`

**是性质预测/表征探针，零设计成分。** 11 个预训练语言模型全部**冻结**，只训练任务头，3 个种子取均值±标准差。
**表 1 列了 11 行**（Thermo 和 Affinity 各拆两种），但正文某处说"十二个任务"——这个不一致要标出来。

| 任务 | 训练/验证/测试 | 类型 | 主指标 |
|---|---|---|---|
| VRCls | 13,703/3,426/2,847 | 多分类（token） | Acc, P, R, F1 |
| CDRInf | 13,703/3,426/2,847 | 多分类（token） | **精确匹配、BLOSUM62 相似度** |
| SARS-CoV-2 | 27,606/6,706/38,958 | 二分类 | AUROC, AUPRC |
| hIL6 | 13,020/11,637/449,234 | 二分类 | AUROC, AUPRC |
| Paratope | 851/139/240 | 二分类（token） | AUROC, AUPRC |
| Thermo-seq / Thermo-tm | 522/92/147、396/57/114 | 回归 | Spearman, R², RMSE |
| PolyRx | 101,854/14,613/25,007 | 二分类 | AUROC, AUPRC |
| NbType | 12,647/1,960/3,557 | 五分类（VHH/VNAR/VH/Vλ/Vκ） | Acc, P, R, F1 |
| Affinity-seq / -score | 8,888/1,302/2,547、8,915/1,274/2,548 | 回归 | Spearman, R² |

划分：MMseqs2 **70% 同一性**，从低频簇采 20% 作测试，其余 70/10。**例外**：AgNbBind 用生物学划分（SARS-CoV-2 训练用野生型/温和变体，测试用 Alpha/Beta/Delta/Omicron）。
数据在 HF collection `ZYMScott/nbbench-...`。**阈值：无。**

## 10.9 逆折叠 CDR 基准：203 Fab + 61 VHH

`PLOS ONE 2025` `10.1371/journal.pone.0324566`，`github.com/biomap-research/InverseFoldingEvaluation`

**清单机器可读，行数正好对上**：`Analysis/data/resources/df_fab_info.csv` **203 行**、`df_vhh_info.csv` **61 行**，都带 CDR 边界和序列，ID 形如 `7t86_G-DC`。日期范围 **2023-01 → 2024-05**（都是时间留出集）。

**最有价值的是它的消融设置**：仓库把 Fab 的三种条件分成三个独立目录——带抗原、**把抗原链从模板 PDB 里删掉**、坐标先 relax。这是它测"抗原依赖性"的方式。

**关键结论**（`Analysis/data/processed/method_summary.csv`）：

| 方法 | Fab CDR BLOSUM62 | VHH CDR BLOSUM62 | 突变效应相关 |
|---|---:|---:|---:|
| AntiFold | **0.703** | **0.368** | 0.199 |
| LM-Design | 0.597 | 0.513 | 0.364 |
| ESM-IF | 0.423 | 0.376 | 0.454 |
| ProteinMPNN | 0.349 | 0.362 | 0.399 |

> **AntiFold 在 Fab CDR 上遥遥领先（0.703），到 VHH CDR 上掉回 ProteinMPNN 水平（0.368）**——一个抗体专用模型迁移不到纳米抗体。这条对纳米抗体那块空白是很有力的旁证。

**阈值：无。**

## 10.10 Germinal —— 唯一有阈值又有湿实验的 de novo 纳米抗体协议

`bioRxiv 2025.09.19.677421`，`github.com/SantiagoMille/germinal`

**真正的 de novo**，三阶段：ColabDesign 幻觉 → 选择性 AbMPNN 重设计 → AF3/Chai-1/Protenix 共折叠。VHH 和 scFv 都支持。
**4 个靶点**：PD-L1、IL-3、IL-20、BHRF1。**仓库只有 PD-L1 和 IL-3（外加 insulin）的靶点文件，IL-20 和 BHRF1 缺**。

**实验数字**：下单 101(PD-L1) / 46(IL3) / 43(IL20) / 52(BHRF1) 条。分裂荧光素酶命中 25/101、11/46、11/43、20/52。BLI 确认 7/25、2/11、4/11、11/20，**BLI 成功率 4–22%**，最好的 K_D 分别 170/560/190/140 nM。
> scFv 设计过滤通过率相当，但**"没有一条 scFv 移植体显示出显著结合信号"**。纳米抗体才是被验证的模式。

**阈值——这是它最大的价值**（`configs/filter/final/vhh.yaml`）：
`clashes < 1`、`sc_rmsd < 6.0 Å`、`binder_near_hotspot == true`、`cdr3_hotspot_contacts > 0`、`percent_interface_cdr >= 0.5`、`interface_shape_comp >= 0.6`、`interface_hbonds >= 3`、`surface_hydrophobicity <= 0.4`、`interface_hydrophobicity >= 45`、`pdockq2 > 0.23`、`external_plddt > 0.87`、`external_iptm > 0.74`、`external_ptm > 0.84`、`external_pae < 7.5`。
scFv 那套不同（`sc_rmsd < 7.0`、`percent_interface_cdr > 0.7`、`interface_hbonds >= 6`…）。论文实跑用的 VHH 过滤更松（`external_plddt >= 0.80`、`external_iptm >= 0.75`、`external_pae < 8`），PD-L1 那档还加了 `ipsae >= 0.6`。
幻觉环内的接受阈值：`plddt 0.82`、`i_ptm 0.68`、`i_pae 0.27`、序列熵 0.10。

> ⚠️ **原文警告："这些过滤器只针对 AF3 的置信度指标校准过"**——Chai-1 和 Protenix 后端是实验性的、未校准。

**成本**：H100 80GB 上每条纳米抗体轨迹 2–8 分钟，约每 GPU 小时出 1 条通过的设计；**200–400 H100 小时 → 约 200 条成功设计**，再从中挑 40–50 条下单。

## 10.11 EasyNano

`arXiv:2606.12772`（2026-06）

**6 个靶点-框架对**：Ty1/SARS-CoV-2 spike(**6ZXN**)、KN035/PD-L1(**5JDS**)、VHH72/SARS-CoV-2 RBD(**6WAQ**)、anti-TNF VHH/TNFα(**5M2J**)、VHH3/TNFα 三聚体(**5M2M**)，外加一个手工对接的 **AQP4** de novo 案例（无晶体结构）。

**验证器**：ESMFold2-Fast(721M) 的 distogram 做可微内环，完整 ESMFold2(1.3B) 出最终 ipTM/pTM。表位定义为 **8 Å Cα–Cα**，CDR 用 **Chothia 编号经 `abnumber`**。
**没有绝对通过线**，成功判据是相对的：Δ > 0.1 且相对**每靶点 n=30 的随机 CDR 基线**有统计显著性（用 σ 表示）。结果差异很大：Ty1 0.143→0.702（+0.559，5.7σ）、VHH72 0.776→0.742（**−0.035，变差**）。
成功的决定因素：**CDR 长度 ≥ 22 个位置**，且初始姿态落在正确的势阱里。

> ⚠️ **代码尚未发布**——可用性声明里的组织名占位符 `https://github.com/[organization]/EasyNano` 字面上就没填。**今天跑不了。**

## 10.12 AbBiBench / FLAb（性质预测，不是设计）

**AbBiBench**（`github.com/MSBMI-SAFE/AbBiBench`）：论文说 14 抗体 × 9 抗原、>184,500 条测量；**HF 上的当前版本更大，17 个 CSV 共 215,698 行**。指标是**模型对完整抗原-抗体复合物的对数似然与实验亲和力的 Spearman 相关**，无阈值。当前前三：ProteinMPNN 0.30、ESM-IF1 0.28、AntiFold 0.21——**逆折叠模型赢，纯序列语言模型接近零甚至为负**。`metadata.json` 机器可读。**不含任何纳米抗体**（17 个全是配对 H/L）。

**FLAb**（`github.com/Graylab/FLAb`，现为 FLAb 2）：**241 个数据集、7 类治疗性质、>300 万条实测**。纳米抗体相关子集：**AVIDa-hIL6 VHH 573,892 条结合**、**SARS-CoV-2 VHH 77,004 条结合**、**NbThermo 673 条热稳定性**（与 NbBench 的来源重叠）。论文自己的结论就在标题里：*"蛋白 AI 模型尚不能稳定预测可开发性"*。

## 10.13 纳米抗体那块空白：已尽力证伪，结论不变

**不存在纳米抗体的社区级 de novo 设计基准**——没有公开可下载、固定靶点清单、共识指标加排行榜的东西。查过的地方：

- **RFantibody**（`RosettaCommons/RFantibody`）有完整的 de novo 纳米抗体流程，但只带示例输入（`flu_HA.pdb`、`rsv_site3.pdb`、框架 `h-NbBCII10.pdb`），**没有案例清单**
- **NbBench** 是纳米抗体专用但 100% 性质预测
- **CHIMERA-Bench** 只有配对 VH/VL，且只做 redesign
- **IgGM SAb-23H2-Nano（27）** 是最大的公开纳米抗体清单，但**是给定框架做 CDR 重设计**，不是 de novo
- **PXDesignBench / ProtDBench**：grep 过整个仓库树，"nanobody/VHH" 的命中全部落在 `.a3m` MSA 文件里的偶然匹配。**没有纳米抗体分支**
- **AIntibody** 是盲测竞赛（29 家机构 511 条设计），任务是亲和力成熟和 HCDR3 排序，**抗体不是纳米抗体，也没有可复用的下载集**

**现有最接近的纳米抗体案例清单，按可用性排序**：

| 来源 | N | 是什么 | 可机读？ |
|---|---:|---|---|
| IgGM `SAb-23-H2-Nano/prot_ids.txt`（Zenodo 13790269） | **27** | 纳米抗体-抗原复合物，2023 下半年，带结构和预挖空 FASTA | **是**（txt + PDB + FASTA） |
| PLOS ONE `df_vhh_info.csv` | **61** | VHH-抗原复合物，带 CDR 边界和序列 | **是**（CSV） |
| Germinal 靶点 | **4** | 唯一同时有已发布阈值**和**湿实验命中率的 de novo 纳米抗体靶点 | 部分（4 个里仓库只有 2 个） |
| EasyNano | 5+1 | 表位定向 de novo CDR 设计对 | **否**（正文散列，代码未发布） |
| nanoFOLD | 43 + 1064 | 逆折叠恢复率 | **否**（无 ID、无仓库） |

> **如果现在就要做纳米抗体评估，诚实的做法是：拿 Germinal 的 4 个靶点 + `filter/final/vhh*.yaml` 的阈值当事实标准**——它是唯一既有配套 in-silico 阈值、又有可对照的 BLI 命中率的。但要在文档里写明**那是一套协议，不是社区基准**。IL-20 和 BHRF1 的靶点 PDB 需要从预印本重建。

## 10.15 纳米抗体：按任务类型汇总

纳米抗体（= 单域抗体 = VHH）的条目散在上面各处，这里按任务汇总。**没有哪两家用同一批靶点，也没有共识验证器。**

### 从头设计

| | 靶点 | 工具 | 指标/阈值 | 能拿到吗 | 详见 |
|---|---|---|---|---|---|
| **BoltzGen** | **10 个低同源新靶点**（1G13 1JQD 1NB0 2A1X 2PNY 3APU 3CH4 3QKG 6M1U 7AAH）+ 5 个简单靶点 | Boltz-2 重折 | **无绝对阈值**：只有 RMSD ≤ 2.5 Å 和氨基酸组成两条硬过滤，其余排名聚合 | ✅ 全开 | §9.2 |
| **Germinal** | 4 个（PD-L1、IL-3、IL-20、BHRF1） | ColabDesign → AbMPNN → AF3 | **唯一有完整阈值级联**（见 §10.10） | ⚠️ 仓库只有 2 个靶点文件 | §10.10 |
| **EasyNano** | 6 个靶点-框架对（6ZXN 5JDS 6WAQ 5M2J 5M2M + AQP4） | ESMFold2 | **相对阈值**：Δ ipTM > 0.1 且相对随机 CDR 基线显著 | ❌ 代码未发布 | §10.11 |
| **RFantibody** | 无案例清单 | RFdiffusion 系 | — | ⚠️ 只有示例输入 | §10.13 |

规模上 **BoltzGen 最大**：每个新靶点生成 6 万条纳米抗体，每靶点取前 15 条送实验，10 个靶点里 6 个拿到筛选命中。
**Germinal 是唯一同时有阈值和湿实验命中率的**：BLI 确认成功率 4–22%，最好 K_D 140–560 nM；但原文警告阈值**只针对 AF3 校准过**。

### CDR 重设计（框架给定）

| | 案例数 | 指标 | 能拿到吗 | 详见 |
|---|---:|---|---|---|
| **IgGM SAb-23H2-Nano** | **27** | 骨架 RMSD + AAR；结构预测的成功率 = **DockQ > 0.23** | ✅ Zenodo 13790269，带结构 + 预挖空 FASTA | §10.6 |

**目前最大的、带结构的公开纳米抗体案例清单。** ID 格式 `{pdb}_{H}_{L}_{Ag}`，轻链槽位填 `NA`。

### 逆折叠

| | 案例数 | 指标 | 能拿到吗 | 详见 |
|---|---:|---|---|---|
| **PLOS ONE 逆折叠基准** | **61 个 VHH**（另 203 个 Fab 作对照） | AAR、BLOSUM62 相似度、**Boltz-1 重折 RMSD**、对 ΔΔG 的 Spearman | ✅ `df_vhh_info.csv` 带 CDR 边界 | §10.9 |
| **nanoFOLD** | 43 晶体 + 1064 模型 | 恢复率（整个 VHH 区 + CDR-H3） | ❌ 无清单无仓库 | §10.7 |

> **一条对纳米抗体很关键的结论**：AntiFold（抗体专用）在 Fab CDR 上 **0.703**，到 VHH CDR 掉到 **0.368**——回到 ProteinMPNN（通用，0.362）的水平。**抗体专用模型迁移不到纳米抗体。**

### 结构预测 / 对接

| | 纳米抗体案例 | 工具 | 指标 |
|---|---:|---|---|
| **Hitawala & Gray** | **60 个 Nb-抗原**（独立分支） | AF3、AF2.3-M、Boltz-1、Chai-1 横比 | DockQ 高精度成功率 |
| **ABAG-docking** | 14 个单域抗体（112 核心案例内） | ZDOCK、ClusPro、HDOCK、AF-Multimer | I-RMSD、fnon-nat、DockQ |
| **abag-benchmark-set** | **0——明确排除纳米抗体** | — | — |

### 性质预测（不是设计）

- **NbBench**（§10.8）：唯一专为纳米抗体做的套件，8 个任务但 **7 个是预测**，只有 CDR 填空沾边生成
- **FLAb** 纳米抗体子集：AVIDa-hIL6 VHH **573,892** 条结合、SARS-CoV-2 VHH **77,004** 条、NbThermo **673** 条热稳定性
- **ANDD**（Sci Data 2026）：统一的抗体+纳米抗体资源，48,683 条序列

### 归纳

- **靶点**：从头设计有清单的是 BoltzGen 10 个、Germinal 4 个、EasyNano 6 个；重设计 IgGM 27 个；逆折叠 61 个 VHH。**互不重叠。**
- **工具**：BoltzGen 用 Boltz-2、Germinal 用 AF3、EasyNano 用 ESMFold2、逆折叠用 Boltz-1、对接横比四个。**无共识验证器。**
- **指标**：只有 Germinal 有完整通过阈值，IgGM 有一条 DockQ > 0.23，EasyNano 是相对阈值，BoltzGen 无绝对阈值。**其余只报数不设线。**

**所以"没有社区级纳米抗体基准"的具体含义是**：靶点各用各的、验证器各用各的、大部分连通过线都没有，**没有任何两家的数字能横着比**。

## 10.14 抗体线的横向注意事项

- **RAbD 60 是领域复用最多的抗体案例集**，两处机器可读。别人的"55""56"都是它的**不同下游过滤**——一定要写明自己排除了哪些
- **接触距离阈值各家不同且不可互换**：dyMEAN 用 6.6 Å 任意原子对（算 CAAR）；CHIMERA-Bench 标注用 4.5 Å 重原子但 DockQ 系列用 8.0 Å Cα–Cα；EasyNano 表位用 8 Å Cα–Cα；Germinal 接触 6.0 Å、hotspot 5.3 Å。**任何"基于接触的 AAR"数字只在同一套约定内可比**
- **全表只有四处有真阈值**：Germinal 的 YAML 级联、IgGM 的 DockQ>0.23、DiffAb 的 IMP%（ΔΔG<0）、EasyNano 的相对 Δ。其余都是只排名不设线

---

# 十一、配方明细：酶 / 配体条件设计

## 11.0 可复现性速查表

**"案例清单在不在仓库里"决定了能不能真复现**：

| 基准 | 清单随仓库发布？ | 路径 |
|---|---|---|
| **AME（RFdiffusion2）** | ✅ **是**，JSON + 41 个输入 PDB | `rf_diffusion/benchmark/mcsa_41.json`、`benchmark/input/mcsa_41/*.pdb` |
| **Studio-179（DISCO）** | ✅ SDF 有 / ⚠️ **任务 JSON 是坏的** | `studio-179/priority_{0,1,2,3}/*.sdf`（170 个） |
| **LigandMPNN 测试集** | ✅ **是** | `training/test_{small_molecule,nucleotide,metal}.json` |
| La-Proteina 26 motif | ✅ 是（**纯蛋白原子，无配体**） | `configs/generation/motif_dict.yaml` |
| MotifBench 30 | ✅ 是（**纯蛋白原子，无配体**） | `test_cases.csv` + `motif_pdbs/` |
| RFdiffusionAA 四配体 | ❌ 无清单，只有 2 个示例 PDB | `input/7v11.pdb`(OQO)、`input/1haz.pdb`(CYC) |
| EnzyGen / EnzyBench | ❌ 只能从 Google Drive 下 | — |
| EnzyControl / EnzyBind | ⚠️ 仓库只有 190 行 demo | Zenodo 15462173（1.34 GB） |
| PocketGen | ❌ 靠脚本重新生成（seed 2021） | `data_preparation/split_pl_dataset.py` |
| **PocketFlow** | ❌ **仓库是空的** | 只有一个 251 字节的 README |
| COMPSS | ⚠️ notebook + Zenodo，**README 里的链接全坏了** | `notebooks_for_figures/` |
| PoseBusters | ✅ 检查项定义在仓库里 | `posebusters/config/{redock,dock,gen,mol}.yml` |

## 11.1 AME（Atomic Motif Enzyme）—— 酶这条线唯一的标准

`bioRxiv 2025.04.09.648075` / *Nat Methods* `s41592-025-02975-x`，`github.com/RosettaCommons/RFdiffusion2`（MIT）

**41 个案例，清单在仓库里且机器可读**：`mcsa_41.json`（41 个键，每个是完整的 Hydra 命令行，含 `inference.ligand`、`contigmap.contig_atoms`）+ 41 个 PDB（命名 `M-CSA编号_PDB编号`，含催化残基 ATOM 记录和配体/金属 HETATM 记录）。

全部 41 个（M-CSA 编号_PDB → 配体 CCD 码）：
```
M0024_1nzy BCA        M0040_13pk ADP,MG,3PG   M0050_1dbt U5P      M0054_1qfe DHS
M0058_1cju DAD,MG     M0078_1al6 OAA,HAX      M0092_1dli UDX,NAD  M0093_1dqa NAP,COA
M0096_1chm CMS        M0097_1ctt DHZ,ZN       M0110_1c0p DAL,PER,FAD
M0129_1os7 AKG,FE2,TAU              M0151_1q0n APC,PH2,MG        M0157_1qh5 GSH,ZN
M0179_1q3s ADP,MG     M0188_1xel UPG,NAD      M0209_1lij RPP,ACP,MG
M0255_1mg5 NAI,ACT    M0315_1ey3 DAK          M0349_1e3v DXC      M0365_1pfk ADP,FBP,MG
M0375_4ts9 PO4,FMC    M0500_1e3i CXF,NAI,ZN   M0552_1fgh ATH      M0555_1f8r CIT,FAD
M0584_1ldm NAD,OXM    M0630_1j79 ORO,ZN       M0636_1uaq DUC,ZN   M0663_1rk2 ADP,RIB
M0664_2dhn PH2        M0674_1uf7 CDV          M0710_1ra0 FE,FPY   M0711_2esd G3H,NAP
M0717_1x7d ORN,NAD    M0731_1mt5 MAY          M0732_1xs1 DUT      M0738_1o98 2PG,MN
M0739_1knp SIN,FAD    M0870_1oh9 ADP,MG,NLG   M0904_1qgx MG,AMP   M0907_1rbl MG,FMT,CAP
```

**怎么选出来的**（原文）：把 M-CSA 里 958 个手工整理的催化活性位点与 **PARITY** 数据集交叉比对，挑出所有反应物和辅因子都存在于 PDB 晶体结构里的反应，整理后得到 **41 个活性位点，覆盖 EC 1–5 类**（这五类占 M-CSA 的 96%）。各 EC 类的具体分布只在图 3a 里，**正文没给数字**。

**流程**（`configs/enzyme_bench_n41.yaml`）：RFdiffusion2 出 **100 条骨架/案例** → **LigandMPNN 每条骨架 8 条序列**（`use_ligand: True`，`omit_AA: XC`）→ **Chai-1 每条序列 5 个扩散样本**。全跑一遍是 41×100×8 = 32,800 次 Chai 折叠，README 自己都警告"在单机上跑完要花不可接受的时间"。

**为什么用 Chai-1 不用 AF2**（原文）：*"我们发现 Chai-1 的侧链相互作用距离更接近天然侧链的参考分布。"*

**成功判据（原文，带单位）**：
> 一个设计算 in-silico 成功，当且仅当：(1) 在 **至少一条 LigandMPNN 序列**的 Chai-1 预测里，按催化残基的骨架 **N、Cα、C** 对齐后，**所有催化残基重原子的 RMSD < 1.5 Å**；且 (2) 设计**与配体无撞车，撞车定义为两原子距离 < 1.5 Å**。

Nat Methods 方法部分补充：至少一条序列的**五个 Chai-1 扩散样本之一**满足。

> ⚠️ **两处代码与论文不一致**（读 `per_sequence_metrics.py` 发现）：
> 1. 代码里的对齐原子集是 **`['N','CA','C','O']`**（含 O），论文写的是 N、Cα、C
> 2. `has_valid_ccd()` **硬编码只有 41 个里的 30 个**"CCD 码能被 chai 正确识别"，另外 11 个的配体位姿指标直接返回空。所以 **"41/41"是催化残基 RMSD 那条判据上的成绩，配体位姿指标只覆盖 30/41**。这 11 个是：`M0024_1nzy, M0054_1qfe, M0058_1cju, M0110_1c0p, M0209_1lij, M0255_1mg5, M0630_1j79, M0674_1uf7, M0717_1x7d, M0731_1mt5, M0870_1oh9`

> ⚠️ 同一个文件里还躺着**旧的 AF2 判据**（`criterion_1..6`），那是老的 RFdiffusion motif 基准不是 AME。**别把那里的 2.0 Å 撞车阈值和 AME 的 1.5 Å 搞混。**

**结果**：RFdiffusion2 解出 **41/41**，RFdiffusion 只有 16/41。难度与"残基孤岛"数量相关（最多 7 个）。

**湿实验：有。** 三四个催化位点做到台面上，*"每种情况下测试不到 96 条序列就找到了有活性的催化剂"*。逆羟醛缩合活性用半定量体外转录翻译测定；水解酶报了米氏动力学，kcat/KM = **248±34、77±10、16,000±2,000、53,000±5,000 M⁻¹s⁻¹**。

**跑法**：`apptainer exec --nv ... pipeline.py --config-name=enzyme_bench_n41_fixedligand in_proc=True`。需要 Apptainer/Singularity、GPU、`python setup.py` 拉权重和 `.sif`（下载 >30 分钟）。Chai-1 "并非在所有 GPU 架构上都能跑"。

## 11.2 Studio-179（DISCO）

`arXiv:2604.05181`，`github.com/DISCO-design/DISCO`

**179 = 170 个单配体 + 9 个多配体组合。SDF 确实在仓库里**：`priority_0/`(1) + `priority_1/`(67) + `priority_2/`(36) + `priority_3/`(66) = **170 个 `.sdf`**。内容跨度很大：辅因子（血红素 b、NAD、FAD、PLP、SAM、辅酶 A、F420、B12、叶绿素、西罗血红素、钼蝶呤）、金属和金属簇（`Zn/Cu/Fe/Ni/Mn/Co/Hg/Pb_ideal`、`Fe2-S2`、`Fe3-S4`、`Fe4-S4`、`fe-akg`）、污染物（PFOA、PFOS、DDT、TCDD、灭蚁灵、DEHP）、药物（阿哌沙班、地瑞那韦、奈玛特韦、甲氨蝶呤）、荧光团（BODIPY、DFHBI、伊红 Y、玫瑰红）、有机金属（Ru(bpy)、铱、二茂铁、顺铂）、卡宾前体，以及 **`priority_0/carbene_v1_..._TS1_heme_...sdf`——一个显式的反应中间体/过渡态结构**。

> ⚠️ **可复现性有个坑，要报给团队**：四个任务文件 `input_jsons/all_priorities_ligands_split_{0,1,2,3}.json` 是坏的。
> - `split_1.json` **JSON 格式非法**——一次失败的路径替换留下了 `"ligand": "FILE_studio-179studio-179"ligand": {`，根本解析不了
> - 四个 split 里引用的 **240 个 `FILE_*.sdf` 路径只有 1 个能解析**。JSON 指向 `priority_N/xtb/` 和 `priority_N/ideal/` 这些不存在的子目录（文件被扁平化了），另有 70 个是作者集群上的绝对路径
> - **但 239 个坏路径里有 236 个能靠 basename 找回**，一个约十行的修复脚本（去掉 `/xtb/`、`/ideal/` 和绝对前缀，按 basename 重映射）就能让整套跑起来
> - README 里单配体的例子（`input_jsons/heme_b.json`）用的是正确的扁平路径，那个能跑

**流程**：RDKit ETKDG 构象 → **GFN2-xTB** 优化 → RDKit 键信息 + 优化几何作为输入。**每个配体在 150/200/250 三个目标长度各生成 50 个复合物 = 150 条设计**。**没有独立的逆折叠步骤**——DISCO 序列结构共设计。验证器 **Chai-1**。

**共设计成功判据（原文 Eq. S52–S53）**：
> RMSD_backbone(X_design, X_Chai) **< 2.0 Å**，且 ‖r̄ℓ_design − r̄ℓ_Chai‖ **< 2.0 Å** 对**每一个**配体 ℓ 成立（r̄ℓ 是配体**质心**）。多配体体系要求每个配体独立满足。

**撞车定义是另一回事**（不参与成功判据）：`d_ij < r_vdW_i + r_vdW_j − 0.5 Å`。

> **PoseBusters 在这里不是过滤器**。它出现在图 3K 和图 S18，是对生成配体构象的**报告性诊断**（图 S18 标题明写"未做共设计过滤的条件生成结果的 PoseBusters 有效性分解"）。共设计判据只有 Chai-1 的 RMSD。

**基线**：RFdiffusion3、BoltzGen、RFdiffusion All-Atom（对 RFAA 每条骨架只出一条 LigandMPNN 序列）。结果 DISCO 在 **178/179** 上最好。

**湿实验：有，而且很扎实。** 血红素酶**仅以反应中间体为条件**、不预先指定催化残基。整细胞大肠杆菌 96 孔筛选，2.2–66% 的变体超过对照。最好结果：对甲氧基苯乙烯环丙烷化 **72% 产率、TTN 4,050、99:1 非对映选择性**；B–H 插入 **98% 产率、5,170 TTN**；C(sp³)–H 烷基化 **42% 产率、2,360 TTN**。随机突变再给了 **4 倍**活性提升。
筛选管线在 SI A.10 里全给了：Chai-1 + AF3 双重共设计性、中间体 4.0 Å 内至少 5 个蛋白重原子、SASA 埋藏比例 > 0.5、最坏情况立体角包围度 > 0.5、金属配位键长在理想值 ±0.5 Å 内、表面疏水比例 < 50%、净电荷 ±15 内、硬阈值 **ipTM ≥ 0.7、pTM ≥ 0.7、链对 ipTM ≥ 0.5**。

## 11.3 RFdiffusionAA —— SAM/OQO/FAD/IAI 那个惯例

`Science` 384, eadl2528（2024），`github.com/baker-laboratory/rf_diffusion_all_atom`

**四个配体，仓库里没有清单**，只有两个示例输入。选这四个的理由：FAD 和 SAM 在训练集里常见；**IAI 和 OQO 与训练集里任何分子都很不同（Tanimoto ≤ 0.5：IAI 0.50、OQO 0.46）**。
（顺带：DISCO 的 `studio-179/priority_3/` 里正好有 `IAI_final_0.sdf` 和 `OQO_final_0.sdf`，这两个可以从那儿拿。）

**每个配体 400 条设计**，**LigandMPNN 每条骨架 8 条序列**，验证器是 **AF2 单序列**（不是 AF3/Chai-1）。
判据：*"AF2 骨架 RMSD **< 2 Å**（八条序列取最优）"*，小分子结合设计里至少 45% 达标。Rosetta ΔG 为负的比例：52%（无辅助势）/ 72%（有接触势），而纯蛋白 RFdiffusion 是 **0%**。

**湿实验：有，三个campaign。** 地高辛配基（4,416 条设计 → 酵母展示 → 最紧的 **Kd = 10 nM**，95°C 稳定）；血红素（168 条设计 → 135 条表达 → 96 条 UV/Vis 符合 → 45 条纯化 → 38 条单体且过 SEC 后保留血红素）；胆色素。

## 11.4 LigandMPNN 测试集

`Nature Methods (2025) s41592-024-02591-1`，`github.com/dauparas/LigandMPNN`（MIT）

**317 / 74 / 83，PDB 编号清单确实在仓库里**：`training/test_small_molecule.json`（317）、`test_nucleotide.json`（74）、`test_metal.json`（83），另有 `train.json`（149,488）和 `valid.json`（7,530）。

**划分规则（原文）**：PDB 截至 **2022-12-16**，X 射线或冷冻电镜、优于 **3.5 Å**、少于 6,000 残基；解析除 `["HOH","NA","CL","K","BR"]` 外的所有残基；序列用 **mmseqs2 按 30% 同一性聚类**；留出**互不重叠**的三个子集。训练时对骨架/环境原子加 **0.1 Å 标准差的高斯噪声**防止记忆。

> ⚠️ **对清单做集合运算发现两处完整性问题**：
> 1. **两个核酸测试 ID 同时出现在 `train.json` 里：`2zio`、`3olt`**
> 2. **五个 ID 同时在金属和核酸两个测试表里：`1qum`、`1u3e`、`2nq9`、`6wdz`、`7kii`**。所以三个测试集的并集是 **469 个唯一 PDB，不是 474**

**指标**：**天然序列恢复率**，只算侧链原子在任何非蛋白原子 **5.0 Å** 内的残基，每个蛋白 10 条设计取中位数。**没有通过阈值**——这是恢复率基准不是成功率基准。

| 环境 | Rosetta | ProteinMPNN | LigandMPNN |
|---|---|---|---|
| 小分子 | 50.4% | 50.4% | **63.3%** |
| 核酸 | 35.2% | 34.0% | **50.5%** |
| 金属 | 36.0% | 40.6% | **77.5%** |

注意仓库里的 `training/*.json` **只有 ID 列表**，结构要自己去 PDB 取；**仓库没有训练脚本**（只有推理）。

## 11.5 La-Proteina 26 个 motif 任务 —— **确认是纯蛋白原子，不含配体**

`arXiv:2507.09466`（ICLR 2026），仓库是 **`github.com/NVIDIA-BioNeMo/la-proteina`**（不是 NVIDIA-Digital-Bio，那个放的是 Proteina 系列）

**26 个基础任务 × 2 种原子选择模式 = 52 个配置项**，清单在 `configs/generation/motif_dict.yaml`。26 个是：`1PRW, 1BCF, 5TPN, 5IUS, 3IXT, 5YUI, 5AOU, 5AOU_QUAD, 7K4V, 1YCR, 4JHW, 5WN9, 4ZYP, 6VW1, 1QJG, 1QJG_NATIVE, 2KL8, 7MRX_60/85/128, 5TRV_SHORT/MED/LONG, 6E6R_SHORT/MED/LONG`。

> **确认无配体**：对 `motif_dict.yaml` grep `ligand|hetatm|sdf` → **0 次命中**。motif 是蛋白残基的骨架+侧链或末端官能团。论文局限性部分自陈"本文的 La-Proteina 实例未针对蛋白复合物训练"，并明确把酶设计列为范围之外。
> **它的 tip-atom 模式看起来很像酶活性位点（图 10–12 就是），但条件里没有任何配体、辅因子或金属——不要归到酶设计那一类。**

**判据（原文 §F.3）**：motif 序列 **100% 恢复**；motif **Cα** 全原子 RMSD **< 1 Å**；motif 全原子 RMSD **< 2 Å**；且整体全原子共设计（**全原子 scRMSD < 2 Å**，用 **ESMFold** 折自己产出的序列）。每个任务 **200 个样本**。La-Proteina 解出 21–25/26，Protpardelle 只有 4/26。

**对比案例 MotifBench**（`blt2114/MotifBench`）同样确认**零配体记录**（"ligand"这个词在白皮书里只出现一次，还是在一篇参考文献标题里）。30 题清单在 `test_cases.csv`。判据：8 条 ProteinMPNN 序列 → ESMFold → motif 骨架（**N、Cα、C**）RMSD ≤ **1.0 Å** 且 scRMSD（**只算 Cα**）≤ **2.0 Å**。每题 100 条骨架，约 1 GPU-天。

## 11.6 EnzyGen / EnzyBench

`arXiv:2405.08205`（ICML 2024），`github.com/LeiLiLab/EnzyGen`

**1,500 条测试**：101,974 条 PDB → 3,157 个四级 EC 类 → 归并成 256 个三级类 → **验证和测试各留 30 个三级类，每类随机取 100 条，50 验证 50 测试** = 30×50。覆盖 323 个四级家族。防泄漏：**按 50% 序列同一性聚类**，确保验证/测试不与训练同簇。
**清单不在仓库里**，要从 Google Drive 下。

**指标和阈值**：**ESP score**（0–1），论文明写 **"ESP ≥ 0.6 表示底物结合为阳性"**，EnzyGen 平均 **0.65**（PROTSEED 0.59、RFDiffusion+ProteinMPNN 0.53）。另有 Gnina 对接的结合亲和力（EnzyGen −9.44 kcal/mol）和 AF2 pLDDT（87.45），**这两个没有阈值**。
**湿实验：无**，论文明说这些是"湿实验之前"的评估。

## 11.7 EnzyControl / EnzyBind

`arXiv:2510.25132`，`github.com/Vecteur-libre/EnzyControl`

**11,100 对酶-底物（源自 PDBbind）。测试划分大小未知**——摘要、§3、§5.1、附录 B/D.2/F.1/F.2、表 1 说明、仓库配置全找过，**没有任何地方写出来**。仓库里 `configs/datasets.yaml` 声明了 `trainset.csv` 和 `testset.csv` 但**两个文件都不在**，另有 `splite_rate: 0.05`（若按 11,100 算约 555 条，但这是推断不是论文所述）。
仓库只带 `metadata/demo.csv`（**190 行**）+ 190 个 PDB 目录。完整数据在 **Zenodo 15462173**（1.34 GB）。

**划分规则**：用 **CD-HIT** 聚类酶序列，确保训练和测试**不相交**。预处理：丢掉 RDKit 解析不了的、EquiBind 式标准化、`reduce` 加氢、**只保留酶原子 10 Å 内的底物原子**、只留单链。
EC 分布：EC1 4%、EC2 42%、EC3 41%、EC4 7%、EC5 3%、EC6 3%。

**流程**：生成骨架 → **ProteinMPNN** 逆折叠 → **ESMFold** 全原子预测 → 所有指标在 ESMFold 结构上算。每案例 20 个样本。
**指标**：设计性 = **scRMSD < 2 Å 的比例**；另报 **scTM > 0.5** 的比例（沿用 Chroma 的定义）；EC 匹配率（用 **CLEAN** 预测）；kcat 预测；GNINA 对接亲和力；ESP。**湿实验：无**，全部功能性结论来自预测器。

## 11.8 PocketGen / PocketFlow

**PocketGen**（`Nature Machine Intelligence 2024`，`github.com/zaixizhang/PocketGen`）：CrossDocked 100 + Binding MOAD 100 测试。**清单不随仓库发布，靠脚本重新生成**（`split_pl_dataset.py`，seed 2021）。
> ⚠️ 坑：默认带 `--fixed_split ./data/split_by_name.pt` 时代码直接整份采用 TargetDiff 的划分，**100 个口袋的子集分支被跳过**——要走那条路径得传 `--fixed_split ""`。

口袋定义 = 配体任意原子 **3.5 Å** 内的所有残基（平均约 8 个）。**这是口袋重设计不是从头搭建**。每个复合物生成 100 条序列和结构，三次独立运行。
指标：AAR；Vina（kcal/mol）；**成功率 = 结合亲和力优于参考口袋的比例**（PocketGen top-1 达 0.97）；RMSD；ESMFold pLDDT；scTM（8 条 ProteinMPNN 序列 → ESMFold → TM-score）。
> ⚠️ 预印本表 2 和 README 表格对同一批实验报的列不一样（前者 AAR/RMSD/Vina，后者 AAR/Designability/Vina），而 PocketFlow 论文又报成 AAR/scRMSD/Vina 且数值不同。**引用时要说清是哪张表。** README 里那个 "Designability" 的确切定义在预印本和 README 里都没给。

**PocketFlow**（NeurIPS 2024，`github.com/zaixizhang/PocketFlow`）：**仓库是空的**——`main` 分支上只有一个 251 字节的 README 写着"我们正在为期刊准备这些项目，代码和数据会一起发布"。**今天不可复现。**
论文里的划分与 PocketGen 相同，另有两个泛化集：**PPDBench**（133 个非冗余蛋白-肽复合物）和 **PDBBind RNA**（56 对）。

## 11.9 COMPSS —— 酶版的"哪个指标真的预测活性"

`Nature Biotechnology`（PMC11919684），`github.com/seanrjohnson/protein_scoring`

不是固定案例集，是**指标校准研究**。开发用两个酶家族（苹果酸脱氢酶 MDH、铜超氧化物歧化酶 CuSOD），外部验证再用六个家族。

**20 个指标对实测活性的 AUC-ROC**（表 1，两家族平均）：

| 组 | 指标 | AUC |
|---|---|---|
| 结构/能量函数 | **Rosetta-relax** | **0.76** |
| 结构/逆折叠 | **ProteinMPNN** | **0.75** |
| | MIF-ST | 0.72 |
| | ESM-IF | 0.70 |
| 单序列/语言模型 | **ESM-1v** | **0.68** |
| | CARP-640M | 0.68 |
| | ESM-MSA | 0.61 |
| 结构/预测置信度 | AlphaFold2 pLDDT | 0.66 |
| 序列比对 | BLOSUM62 / PFASUM15 | 0.62 |
| | 同一性 | 0.61 |
| 单序列/残基计数 | 净电荷 | 0.39 |
| 结构/表面积 | SASA | 0.39 |
| | 非极性 SASA | 0.37 |

**最终选了 ESM-1v + ProteinMPNN**（原文）：*"这个组合有吸引力，因为 ESM-1v 基于序列、ProteinMPNN 考虑结构信息，且两个指标都不与序列同一性强相关…… 大多数逆折叠或能量函数类指标表现相近，但 **ProteinMPNN 计算效率最高。Rosetta-relax 在 MDH 上最好但成本高得多**。"* 两个指标只是中度相关（Spearman ρ = 0.60）。
**序列同一性不能预测活性。**

**过滤器的确切阈值**：序列须以甲硫氨酸开头、无长重复、无跨膜结构域；**ESM-1v 阈值 = 同家族天然序列的第 10 百分位**；与天然序列同一性落在 **50–80%**；AF2 预测结构后**按 ProteinMPNN 分数从前 40 名里随机取 18 条**。
**结果**：144 条序列表达纯化测活，合并后 **74% 有活性**，比未通过过滤的对照高 **77%**（双尾 Fisher P = 0.00018）。

> ⚠️ **仓库 README 是坏的**——所有超链接（源数据、依赖仓库、论文 DOI、甚至 conda 环境）都被改写成了同一个 zip 路径。直接用 notebook，别信 README 的链接。

## 11.10 PoseBusters —— PB-valid 的确切检查项

`Chemical Science` 15, 3130–3139（2024），`github.com/maabuu/posebusters`

**PB-valid = 通过 PoseBusters 里的全部检查**。三组，阈值（论文表 4，已与仓库 `config/redock.yml` 交叉核对，两者一致）：

**化学有效性**：RDKit 可加载、通过 sanitisation、可转 InChI、原子全连通、无自由基、分子式/键/四面体手性/双键立体化学与真值一致。

**分子内**：
- 键长在距离几何上下界的 **0.75–1.25** 倍之间
- 键角同样 **0.75–1.25**
- 内部撞车：非共价键原子对的距离 **> 下界的 0.7 倍**
- 芳香环（5/6 元）所有原子距最近共面 **< 0.25 Å**；脂肪族碳碳双键的两个碳及四个邻居同样 **< 0.25 Å**
- 能量比：不超过 **50 个构象**系综平均能量的 **100 倍**（UFF，ETKDGv3 生成后 UFF 弛豫 200 步）

**分子间**：
- 蛋白-配体最小距离 **> 0.75 × vdW 半径和**
- 到有机辅因子同样 **0.75 × vdW**；到**无机**辅因子是 **0.75 × 共价半径**
- 与蛋白的体积重叠 **< 7.5%**（vdW 半径 **× 0.8**）；与无机辅因子重叠 **< 7.5%**（vdW **× 0.5**）

**模式很重要**：`redock.yml` 需要真值配体；**设计管线要用的是 `dock.yml`**（只要 `mol_pred` + `mol_cond`，不需要参考配体）。装起来只要 `pip install posebusters`。

## 11.11 RFdiffusion3 的 AME 成绩

`bioRxiv 2025.09.18.676967`，代码 `github.com/RosettaCommons/foundry`

**37/41（90%）**，原文：*"RFD3 在 41 个案例中的 37 个上优于 RFD2（90%）。"*
判据（图 3d 说明）：*"通过的骨架定义为：**8 条 LigandMPNN 序列中至少有一条**的 **Chai-1** 预测里，motif 骨架对齐后的 motif 全原子 RMSD **< 1.5 Å**。"*

> ⚠️ **这个复述省掉了 RFD2 自己判据里的配体撞车那一条**。所以拿 41/41 和 37/41 直接比，两篇论文引的其实是略有差别的过滤器。

残基孤岛超过 4 个的案例上 RFD3 通过率 15% vs RFD2 的 4%（n=12）。RFD3 比 RFD2 快约 **10 倍**。
**AME 清单在 RFD3 仓库里只是引用**（配置指向外部数据目录），**要用就用 RFdiffusion2 仓库里那份**。

**湿实验：有。** DNA binder 5 条设计里 1 条结合（EC50 = 5.89 ± 2.15 μM）；半胱氨酸水解酶/酯酶——**筛 190 条设计得到 35 条多轮转化的，最活跃的 kcat/Km = 3557**，超过此前针对同一反应的设计。

---

## 11.12 酶线的横向注意事项

**一、配体条件 vs 纯蛋白原子，别混。**
真正的配体/辅因子/金属条件：**AME、Studio-179、RFdiffusionAA 四配体、RFD3 四配体、LigandMPNN 测试集、EnzyGen、EnzyControl、PocketGen、PocketFlow**。
**纯蛋白原子（确认无配体）：La-Proteina 的 26 个任务、MotifBench 的 30 题。**

**二、两套互不兼容的验证器惯例。**
- AME / RFD3 / DISCO 用 **Chai-1 或 AF3** 重折，评的是**催化侧链/配体位姿**精度，阈值 **1.5–2.0 Å**
- La-Proteina / MotifBench / PocketGen / PocketFlow / EnzyControl 用 **ESMFold** 重折，评的是**骨架**自洽，阈值 **2.0 Å**

**这两族的数字不可比。**

**三、撞车阈值有四套不同定义。**
AME 发表的定义是**两原子 < 1.5 Å**；同一仓库代码里的 `criterion_4/5` 用 **2.0 Å**（那属于旧的 AF2 motif 基准）；DISCO 用 **vdW 半径和 − 0.5 Å**；PoseBusters 用 **0.75 × vdW 半径和**。

**四、拿了就能跑的**：AME（文档最全，一条命令）、MotifBench、La-Proteina、LigandMPNN 测试集、PoseBusters。
**需要修复脚本**：Studio-179。
**需要外部下载**：EnzyGen（Drive）、EnzyControl（Zenodo 1.34 GB）、PocketGen（Drive + 重新生成）。
**跑不了**：PocketFlow（仓库空的）。

**五、AME 是这批里唯一同时做到"配体条件 + 湿实验锚定 + 随仓库发布可机读清单"的。** 如果酶这条线只标准化一个基准，这个组合是它独有的。

---

# 十二、实操：怎么"多跑几套"

前面十一节是"市面上有什么"。这一节换个角度——**如果要一次生成、多套打分，具体怎么安排**。

## 12.1 最省事的做法：一批设计，多把尺子

**生成设计是最贵的一步，打分相对便宜。** 所以合理的做法不是选一个基准跑一遍，而是**生成一次，用不同的判定各算一遍**——这样能看出改动是普遍变好，还是只在某一把尺子下变好（后者往往是过拟合到那把尺子）。

**ProtDBench 已经把这件事做成产品了**：同一批设计同时按五档打分。

| 档 | 等价于谁 | 条件 | 需要的模型 |
|---|---|---|---|
| `af2_easy` | BindCraft | pLDDT>0.8、i_pTM>0.5、i_pAE<0.35（归一化）、bound_unbound_RMSD<3.5 | AF2 |
| `af2_opt` | A-CODE / PXDesign / 我们 | pLDDT>0.9、**unscaled_i_pAE**<7.0、binder RMSD<1.5 | AF2 |
| `ptx` | Protenix 全量 | iptm_binder>0.85、ptm_binder>0.88、RMSD<2.5 | Protenix |
| `ptx_mini` | Protenix-Mini | 同上 | Protenix-Mini |
| `ptx_basic` | Protenix 放宽 | iptm_binder>0.8、ptm_binder>0.8、RMSD<2.5 | Protenix |

**但不是"跑一次 AF2 + 一次 Protenix 就拿到五个数"** —— 要跑**三次模型**：`CODE_VERIFIED`

| 模型运行 | 产出哪几档 | 说明 |
|---|---|---|
| AF2（复合物 + 单体两次预测） | `af2_easy`、`af2_opt` | 两档共用同一批 AF2 输出，只是阈值和 RMSD 定义不同 |
| Protenix-**Mini** | `ptx_mini` | 用 `ptx_mini_*` 列 |
| Protenix **全量** | `ptx`、`ptx_basic` | 共用 `ptx_*` 列，`ptx_basic` 是放宽档。**`eval_protenix` 默认 `False`** |

依据：`protd_configs/eval.py` 里 `eval_protenix_mini: True` / `eval_protenix: False` 是两个独立开关，`tasks/binder.py:145,148` 是两个独立分支，各自调一次 `protenix_predict()`。
旁证：放出的 PXDesign 数据里 `ptx_success` 和 `ptx_basic_success` **整列为空**，与全量 Protenix 默认不跑一致。`CODE_VERIFIED`

这五档覆盖了领域里两套主要惯例（BindCraft 系和 AF2-IG 系）。

**对我们有个额外优势**：`Protenix` 本来就在 `PXDesign-train/Protenix/` 里，是我们的依赖。所以 `ptx*` 那三档对我们是**边际成本最低**的。

还可以在同一批设计上再加的判定（需要额外装模型）：
- **ODesign 那套**：`ipAE<10.85、pLDDT>80、ipTM>0.5、复合物 RMSD<2.5` —— 需要 AF3
- **Latent-X 那套**：`min_ipae<1、ptm_binder>0.9、complex_rmsd<2` —— 需要 Chai-1，注意是 **min 不是 mean**
- **BoltzGen 那套**：refolding RMSD ≤2.5 + 设计单独重折 ≤2.5 —— 需要 Boltz-2

## 12.2 工具依赖去重

把全表的依赖合并，实际只需要这几样：

| 工具 | 谁要用 | 代价 |
|---|---|---|
| **AF2**（权重约 5.3 GB） | AF2-IG 惯例、BindCraft、RFdiffusion、ProtDBench 的 af2 两档、Adaptyv（经 ColabFold） | 中 |
| **Protenix-Mini** | `ptx_mini` 档 | **我们已有**（`PXDesign-train/Protenix/`） |
| **Protenix 全量** | `ptx` + `ptx_basic` 两档，**默认不跑** | 我们已有，但要显式打开 |
| **ESMFold** | MotifBench、La-Proteina、PocketGen、EnzyControl、无条件生成的约定俗成协议 | 中 |
| **Chai-1** | AME、DISCO、Latent-X、RFdiffusion3 | 中（"并非所有 GPU 架构都能跑"） |
| **AF3** | ODesign、Germinal、BoltzDesign1 | 高（需自行申请安装 + 数据库 + HMMER） |
| **Boltz-1 / Boltz-2** | BoltzGen、Latent-X 的验证 | 中 |
| **ProteinMPNN / LigandMPNN** | 几乎所有方法的序列设计步 | 低 |
| **PyRosetta** | BindCraft、DiffAb、PepGLAD、DiffPepBuilder、RFpeptides | 低，但**非学术用途要商业授权** |
| **Foldseek** | 多样性和新颖性，到处都在用 | 低 |
| **PoseBusters** | 配体构象合法性 | **`pip install` 就完事** |

## 12.3 拿了就能跑 / 要修 / 跑不了

| 状态 | 有哪些 |
|---|---|
| **拿了就能跑**（清单随仓库发布） | **AME**（41 个案例，一条命令，文档最全）、**MotifBench**（30 题）、**La-Proteina**（26 个 motif）、**LigandMPNN 测试集**（317/74/83）、**PoseBusters**、**ProtDBench**（含 164 MB 现成数据）、**CHIMERA-Bench**（splits JSON 最规范）、**IgGM 测试集**（Zenodo，含 27 个纳米抗体）、**PLOS ONE 逆折叠集**（203 Fab + 61 VHH，CSV） |
| **要写个修复脚本** | **Studio-179**：JSON 格式非法 + 240 个路径只有 1 个能解析，但 236/239 能靠文件名找回，约十行代码 |
| **要外部下载** | EnzyGen（Google Drive）、EnzyControl（Zenodo 1.34 GB）、PocketGen（Drive + 重新生成脚本）、Cao 2022（IPD 六个 tar.gz） |
| **清单要自己重建** | **DiffAb 19 个**（按抗原名过滤 TSV 得到，文档 §10.3 已列出全部 19 条）、**ProteinBench 55 个**（RAbD 60 减去五个具名条目） |
| **跑不了** | **PocketFlow**（仓库是空的）、**EasyNano**（代码未发布，可用性声明里组织名占位符都没填）、**Latent-X**（商业模型，代码权重都不公开）、**nanoFOLD**（无清单无仓库） |

## 12.4 单套的代价量级

同样是"跑一个基准"，代价差几个数量级，排期时要有数：

| 量级 | 例子 |
|---|---|
| **一杯咖啡** | PoseBusters（纯 CPU 检查）、Foldseek 聚类 |
| **1 GPU-天** | MotifBench（30 题 × 100 骨架 × 8 序列 → ESMFold） |
| **几 GPU-天** | AlphaProteo 十靶点单长度 + AF2 打分 |
| **几十 GPU-天** | Studio-179（179 个配体 × 150 条设计 → Chai-1）、La-Proteina（26 × 200） |
| **劝退级** | **AME 全量 = 41 × 100 × 8 = 32,800 次 Chai 折叠**，README 自己写"在单机上跑完要花不可接受的时间"；**逐靶点长度扫描**（BHRF1 41 个长度、IL-17A 91 个长度，比单长度贵几十倍） |

## 12.5 in-silico 基准 vs 带湿实验标签的数据

这两类混在同一份清单里，用法完全不同：

| | **in-silico 基准** | **带湿实验标签的数据集** |
|---|---|---|
| 你做什么 | 跑模型出设计 → 用结构预测器打分 | **不能跑模型**（设计已存在），只能拿来检验打分方式 |
| 回答什么问题 | 我的模型比别人强吗 | **我的打分方式能预测现实吗** |
| 例子 | 上面绝大多数 | 见下表 |

**可下载、带真实实验结果的**：

| 数据 | 规模 | 测的什么 |
|---|---:|---|
| **ProtDBench 的 Cao 打分表** | **236,246 条设计 × 8 打分器 + 结合标签**（1,485 个真结合） | 结合与否 |
| **Cao et al. 2022** | 12 蛋白 / 13 位点，每位点 1.5 万–10 万条设计 | 酵母展示 SC₅₀ |
| **Overath 元分析** | 3,766 条实测 binder / 15 靶点 | 结合与否 |
| **Adaptyv EGFR 两轮** | 601 条表征过的设计 | BLI 的 K_D |
| **Rocklin 组** | 614 个实测单体 / 11 项研究 | 折叠成功与否 |
| **COMPSS** | 144 条序列表达纯化 | 酶活性 |
| **AbBiBench / FLAb** | 21 万行 / 300 万条 | 亲和力、可开发性 |
| **AIntibody** | 511 条 AI 设计抗体，29 家机构 | 盲测亲和力 |

**这类数据不需要任何湿实验能力就能用**——§6.5.2 那组 AUC（ColabFold 0.801 / AF2-IG 0.727 / ESMFold 0.607）就是这么算出来的。

> ⚠️ 另有一类湿实验**不是可复用数据**：AME 的 kcat/KM 53,000、DISCO 的 TTN 4,050、Germinal 的 BLI 4–22%、RFdiffusionAA 的 Kd 10 nM——这些是各家验证自己方法的一次性实验，**别人没法在上面比，也不能拿来跟我们的 in-silico 数对标**。

## 12.6 如果要动手，成本最低的顺序

不是推荐"用哪个"，是按"投入产出"排的：

1. **ProtDBench 的档位**——靶点已复现，数据现成，Protenix 我们本来就有。注意要拿满五档需要三次模型运行（AF2、Protenix-Mini、Protenix 全量），全量那次默认是关的；只加 AF2 权重先拿到 `af2_easy` + `af2_opt` 两档已经够对标 A-CODE Table 4
2. **在同一批设计上加 Chai-1 或 AF3 的判定**——看结论跨验证器族是否还成立（这是最能暴露过拟合的一步）
3. **MotifBench 30 题**——独立维护、有排行榜、约 1 GPU-天，是单体/scaffolding 那条线最省事的入口
4. **AME 41 个**——酶那条线唯一会被审稿人认的数字，但要先算清 32,800 次 Chai 折叠的账，可能得先跑子集

---

# 十三、配方明细：单体 / 无条件生成 / motif scaffolding

第二轮核实（2026-09-13）。只做调研与协议核准，**不建实现、不写 runner**。

## 13.0 先分清三个词——这一类到底在测什么

这一类基准里反复出现"designability"，但**指的不是同一件事**，而这个差别正好卡在我们的 claim 上。

| 词 | 序列从哪来 | 比什么 | 测到我们的全原子 claim 了吗 |
|---|---|---|---|
| **designability** | **ProteinMPNN 重新生成**（丢掉模型自己的序列） | 骨架 RMSD（通常只 Cα） | **没有**。它问的是"这个骨架存不存在某条序列能折成它"，与共设计能力无关 |
| **co-designability** | **模型自己出的序列** | 骨架 RMSD | 测到了共设计，但**仍不测侧链** |
| **all-atom co-designability** | **模型自己出的序列** | **全原子 RMSD** | **测到了** |

`EXPLICIT_SOURCE`（La-Proteina 原文）：*"ESMFold to fold the corresponding sequence. If this **all-atom RMSD is less than 2Å**, the sample is deemed all-atom co-designable"*；而 *"**Designability** ... uses **ProteinMPNN** to produce sequences for generated structures"*。同一篇论文里两个指标并列，口径不同。

> **对我们的直接含义**：领域里最常见的那套无条件生成协议（长度 {100,150,200,250,300}、每骨架 8 条 ProteinMPNN 序列、ESMFold、取最小 Cα scRMSD）**会把我们共设计出来的序列丢掉**，只考骨架。要考到我们真正主张的东西，必须用 all-atom co-designability 那一档。

## 13.1 MotifBench

| 字段 | 内容 | 标记 |
|---|---|---|
| **task** | motif scaffolding：给定一段（或几段）功能 motif 的骨架坐标，生成把它包住的支架 | `EXPLICIT_SOURCE` |
| **test cases** | **30 题**，`test_cases.csv` 随仓库发布，列为 `pdb_id, motif_residues, redesign_idcs, length, group`；另有 30 个对应的 `motif_pdbs/*.pdb` | `CODE_VERIFIED` |
| **case selection** | 按 motif 的**连续片段数**分三组，各 10 题：group 1 单段、group 2 两段、group 3 三段及以上。30 题中 **20 题是多段** | `CODE_VERIFIED` |
| **input conditioning** | motif 每段单独作为一条链（A/B/C…）。**可重设计的残基**剥成 N/CA/C/O 且残基类型设为 `UNK`；**不可重设计的残基保留侧链重原子**。30 题中 16 题有 `redesign_idcs` | `CODE_VERIFIED`（readme §Motif pdb files；30 个 PDB 里 28 个确实含侧链原子） |
| **expected output** | 指定长度的支架骨架，motif 嵌在其中。motif 在支架里的位置**可以自己选**（甚至动态选） | `EXPLICIT_SOURCE` |
| **sample count / 长度** | 每题 **100 条支架**（`sample_num` 0–99）。**长度是固定值**，逐题给定，范围 75–225 | `CODE_VERIFIED` |
| **sequence design route** | **ProteinMPNN 出 8 条序列**（骨架不全 N/Cα/C/O 时用 Cα-only ProteinMPNN） | `EXPLICIT_SOURCE` |
| **evaluator** | **ESMFold** | `EXPLICIT_SOURCE` |
| **metric / units** | `motifRMSD`：输入 motif 与预测结构对应原子的 RMSD，**只用 N、Cα、C**；`scRMSD`：生成结构与预测结构的 RMSD，**只用 Cα** | `EXPLICIT_SOURCE`（白皮书正文 + Algorithm 2 两处一致） |
| **threshold** | 八条序列中**至少一条**同时满足 `motifRMSD ≤ 1.0 Å` 且 `scRMSD ≤ 2.0 Å` | `EXPLICIT_SOURCE` |
| **aggregation** | 成功率 = 成功数/100；唯一解 = 成功者的 Foldseek 聚类簇数；**MotifBench score** = `(1/30)·Σ (100+α)·nᵢ/(α+nᵢ)`，`α = 5`，范围 0–100（1/5/50 个解分别得 17.5/52.5/95.5） | `EXPLICIT_SOURCE` |
| **diversity / novelty** | Foldseek 聚类算唯一解；新颖性 = 各簇 `1 − max PDB TM-score` 的均值 | `EXPLICIT_SOURCE` |
| **wet-lab status** | 无 | `EXPLICIT_SOURCE` |
| **code / data** | `github.com/blt2114/MotifBench`，`test_cases.csv` + `motif_pdbs/` 随仓库；内嵌 Scaffold-Lab。**有活跃排行榜**（readme 里） | `CODE_VERIFIED` |
| **compute** | 约 **1 GPU-天**；需 ProteinMPNN、ESMFold、Foldseek + 一个 PDB 数据库 | `EXPLICIT_SOURCE` |
| **Proteo-AA** | `NEEDS_ADAPTER`，见 §13.3 | — |

> **要点：MotifBench 的评分是纯骨架的。** 输入虽然对功能残基保留了侧链，但两条判据一条用 N/Cα/C、一条只用 Cα，**侧链摆放完全不进分数**。而且它规定用 ProteinMPNN 出序列——对共设计模型而言，等于**我们自己出的序列被丢掉**。

## 13.2 La-Proteina 的 motif scaffolding

| 字段 | 内容 | 标记 |
|---|---|---|
| **task** | **全原子** motif scaffolding，两种给法：`all_atom`（给 motif 残基的全部原子）和 **`tip_atoms`（只给功能关键的侧链末端原子）** | `EXPLICIT_SOURCE` |
| **test cases** | **26 个基础案例 × 2 种原子给法 = 52 个条目**，全在 `configs/generation/motif_dict.yaml`；对应 19 个 PDB 文件在 `motif_benchmark_pdb_files/`（5TRV/6E6R/7MRX 各有 SHORT/MED/LONG 变体） | `CODE_VERIFIED` |
| **input conditioning** | 每条目给 `contig_string`（如 `5-20/A1-20/10-25/B1-20/5-20`，**linker 是可变长区间**）、`motif_min_length`/`motif_max_length`、`segment_order`、`atom_selection_mode` | `CODE_VERIFIED` |
| **expected output** | 全原子结构 + 序列（模型自己共设计出来的） | `EXPLICIT_SOURCE` |
| **sample count / 长度** | 每任务 **200 条样本**；长度是**区间**不是固定值 | `EXPLICIT_SOURCE` + `CODE_VERIFIED` |
| **sequence design route** | **模型自己出序列**，无 ProteinMPNN 环节（这是与 MotifBench 最本质的差别之一） | `EXPLICIT_SOURCE` |
| **evaluator** | **ESMFold** | `EXPLICIT_SOURCE` |
| **threshold** | 四条**同时**满足：① motif 序列 **100% 恢复**；② motif Cα RMSD **< 1 Å**；③ **motif 全原子 RMSD < 2 Å**；④ **全原子 scRMSD < 2 Å**（= all-atom co-designable）。**不设 pLDDT 门槛**，pLDDT 另行单独报 | `EXPLICIT_SOURCE` |
| **aggregation** | 每任务成功数；唯一解 = Foldseek 聚类（`--alignment-type 1 --cov-mode 0 --min-seq-id 0 --tmscore-threshold 0.5 --single-step-clustering`） | `EXPLICIT_SOURCE` |
| **两个难度轴** | `indexed`（给 motif 残基的序号）vs **`unindexed`**（模型自己推断放哪）；× `all_atom` vs `tip_atoms` = 四种设置 | `EXPLICIT_SOURCE` |
| **wet-lab status** | 无 | `EXPLICIT_SOURCE` |
| **code / data** | `github.com/NVIDIA-BioNeMo/la-proteina`（**不是** NVIDIA-Digital-Bio，那是 Proteina 系列）；案例清单和 PDB 都随仓库。**无独立排行榜**，是论文自带基准 | `CODE_VERIFIED` |
| **compute** | 26 任务 × 200 样本 + ESMFold；需 ProteinMPNN（只给 designability 那条用）、ESMFold、Foldseek | `EXPLICIT_SOURCE` |
| **Proteo-AA** | `NEEDS_ADAPTER`，见 §13.3 | — |

## 13.3 两者到底差在哪（问题一与问题三的回答）

**它们不是同一个基准的两个版本，案例集只重叠 6 个 PDB。** `CODE_VERIFIED`

| | MotifBench | La-Proteina |
|---|---|---|
| 案例数 | **30 题** | **26 基础 × 2 模式 = 52** |
| PDB 重叠 | 共有 `1BCF 4JHW 5IUS 5WN9 5YUI 6E6R` 六个 | 同左 |
| motif 给什么 | 非可重设计残基带侧链，可重设计的剥成骨架 + `UNK` | `all_atom` 全给 / **`tip_atoms` 只给侧链末端** |
| 支架长度 | **固定值**（逐题给定，75–225） | **区间**（contig 里 linker 可变长） |
| motif 位置 | 自己选 | `indexed` 给序号 / **`unindexed` 自己推断** |
| 序列从哪来 | **ProteinMPNN 8 条** | **模型自己共设计** |
| 判据 | **纯骨架**：motifRMSD 用 N/Cα/C ≤1 Å；scRMSD 只用 Cα ≤2 Å | **全原子**：序列 100% 恢复 + motif Cα <1 Å + **motif 全原子 <2 Å** + **全原子 scRMSD <2 Å** |
| 每题样本 | 100 | 200 |
| 维护方 | **独立维护 + 活跃排行榜** | 论文自带，无独立排行榜 |
| 测到侧链了吗 | **没有** | **测到了**（尤其 tip_atoms 档） |

**一句话**：MotifBench 考的是"**骨架能不能把 motif 的主链摆对**"，La-Proteina 考的是"**全原子能不能把 motif 连侧链一起摆对，而且你自己出的序列要能折回去**"。

**对问题三的回答**：无条件/单体那一支的主流协议**只测骨架 designability**，而且会用 ProteinMPNN 替换掉我们的序列——**覆盖不到我们的全原子/侧链 claim**。能覆盖到的只有 all-atom co-designability 这一档（目前只有 La-Proteina 系统性地报）。

## 13.4 motif 条件化与 Proteo-AA 推理语义是否一致（问题二的回答）

**机制层面：一致，但那份实现不在我们当前这条线上。** `CODE_VERIFIED`

- **我们现在用的采样器 `pxdesign_train/cogenerate.py` 没有任何坐标还原**——去噪每一步之后没有把固定部分压回给定坐标。它做的是"设计链整条从噪声生成，靠 trunk / pair 表示感知上下文链"。
- **但项目里存在这个机制**：`pxdesign_train/stage4.py` 的 `generate()` 有逐步固定原子还原：

```python
atom_design = design[feat["atom_to_token_idx"].long()]        # token 级 design 掩码展开到原子
xyz = torch.where(atom_design[None,:,None], xyz, fixed_xyz)   # 初始化就压
for current, following in zip(schedule[:-1], schedule[1:]):
    denoised = model.diffusion_module(...)
    xyz = xyz + (following-current)*(xyz-denoised)/current
    xyz = torch.where(atom_design[None,:,None], xyz, fixed_xyz)   # 每一步都压
xyz = torch.where(atom_design[None,:,None], denoised, fixed_xyz)  # 收尾再压
```

- 掩码是**原子级**的（由 `design_token_mask` 经 `atom_to_token_idx` 展开），**原则上可以表达残基级固定**——这正是 motif scaffolding 需要的语义。
- **但 `stage4.py` 不在 `main` 上，也不在我们这几条工作分支上**，只在 `feat/official-pxdesign-fampnn`、`feat/sc-rigid-augmentation`、`sjm/stage4-ligandmpnn`、`stage4-fampnn` 四条分支上。`CODE_VERIFIED`

**差距在哪（不在采样器，在特征化和规格解析）**：

| 缺什么 | 说明 |
|---|---|
| 残基级填充 design 掩码 | 现在 `token_is_design` 由 `_token_level_mask(..., binder_atom_mask)` 从**链级** binder 掩码派生。要做 motif，得改成按 motif 残基逐个填 |
| contig 字符串解析 + 可变长 linker 采样 | 两个基准的规格都是 contig 或残基区间，我们没有解析器 |
| unindexed 放置 | La-Proteina 的更难档要求模型自己推断 motif 放在哪，我们没有对应机制 |
| 分段 motif 的多链输入 | MotifBench 把每段 motif 作为独立链给出，20/30 题是多段 |

**结论**：`NEEDS_ADAPTER`。适配点主要在**特征化层和规格解析**，**采样器本身的机制已经存在**（虽然在另一条分支上）。这和 binder 那条"靶点整条链作为上下文"的语义不同，不能直接套用。

> **顺带纠正一处流传的说法**：有材料称 `pxdesign_train/stage4.py` 的固定原子还原可以直接用于当前的 alphaproteo10 评估。核实结果是 `stage4.py` **不在 `main`、也不在 `bench/alphaproteo-10-targets` 上**——引用它时必须说明是哪条分支。`CODE_VERIFIED`

## 13.5 无条件 / 单体那一支：哪些算基准，哪些只是惯例（问题四、五的回答）

| 名称 | 是基准还是惯例 | 案例/协议是否公开 | 有基线吗 | 标记 |
|---|---|---|---|---|
| **MotifBench** | **基准**——独立维护、固定 30 题、**有活跃排行榜**、有明确 score 公式 | 是（`test_cases.csv` + 30 个 PDB） | 是（排行榜：RFdiffusion3 43.52、Protpardelle-1c 33.81、Genie3 31.14、La-Proteina 29.75、RFdiffusion 21.95、RFdiffusionAA 20.99） | `EXPLICIT_SOURCE` |
| **La-Proteina 26 任务** | **论文自带基准**——案例和判据都公开，但**无独立维护方、无排行榜** | 是（`motif_dict.yaml` + 19 个 PDB） | 是（论文内对比 Protpardelle 等） | `CODE_VERIFIED` |
| **RFdiffusion 25 题 motif 集 / Genie2 的 24 题变体** | **前身**，已被 MotifBench 取代（MotifBench 明说建立它就是因为"各家评估方式差异大导致没法比"） | 在 RFdiffusion SI 里 | 有 | `EXPLICIT_SOURCE` |
| **无条件生成的 "FrameFlow/FoldFlow 协议"** | **只是惯例，不是基准**——没有固定案例集（长度就是 {100,150,200,250,300}），没有维护方，没有排行榜 | 协议公开、无案例集可言 | 各家自报 | `EXPLICIT_SOURCE` |
| **ProteinBench 的 co-design / backbone design 分支** | **基准**，但**是 2024 年的快照**，有 HuggingFace 排行榜 | 部分（结果 CSV 机器可读；motif scaffolding 的案例数 `UNRESOLVED`） | 是（9 个结构设计模型、4 个逆折叠、4 个序列设计） | `EXPLICIT_SOURCE` |
| **Protein-SE(3)** | **统一训练框架**而非评估基准——它的价值是让六个模型在同一数据同一指标下训练 | 代码公开，案例数 `UNRESOLVED` | 6 个模型 | `EXPLICIT_SOURCE` |

**关键区分**：**无条件生成那套"标准协议"没有测试集**。它规定的是长度、序列条数、折叠器和阈值——是一套**评估配方**，不是一组题。所以"在无条件生成上比别人强"这句话，各家跑的是各自生成的样本，**没有共同的题面**。这与 MotifBench 那种固定 30 题有本质区别。

## 13.6 候选短名单（问题六的回答）

**只做分级和理由，不代表现在就去实现。**

### `STRONG_CANDIDATE`

**MotifBench（30 题）**
- **唯一一个满足"固定案例集 + 独立维护 + 活跃排行榜 + 明确 score 公式"四条的**，报出来审稿人认
- 案例清单和 motif PDB 全部随仓库发布，`CODE_VERIFIED`
- 约 1 GPU-天，是这一类里最便宜的入口
- **但要说清楚它测不到我们的侧链 claim**：判据纯骨架，而且规定用 ProteinMPNN 出序列——我们共设计的序列会被丢掉。**它验证的是我们的骨架生成和 motif 保持能力，不是全原子能力**

**La-Proteina 的 all-atom co-designability（无条件那一档）**
- **唯一一个直接考"模型自己出的序列 + 全原子 RMSD"的指标**，正对我们的 claim
- 定义清楚（ESMFold 折模型自己的序列，全原子 RMSD < 2 Å，不设 pLDDT 门槛）
- 无条件生成不需要 motif 适配，**接入成本在这一类里最低**
- 缺点：它是论文自带指标不是社区基准，横向比较对象有限

### `POSSIBLE`

**La-Proteina 的 26 个 motif 任务（尤其 `tip_atoms` 档）**
- **唯一系统性考全原子 motif 的**，`tip_atoms` 那档（只给侧链末端原子、模型补全其余）几乎是为我们这类模型设计的
- 案例和判据都随仓库发布
- 但适配成本明显更高：contig 解析、可变长 linker、`unindexed` 放置都要新写；而且无独立排行榜，说服力不如 MotifBench

**ProteinBench 的 co-design 分支**
- 有排行榜、有机器可读结果
- 但是 2024 年快照，且它的 motif scaffolding 案例数我们还没核实（`UNRESOLVED`）

### `SURVEY_ONLY`

**无条件生成的 "FrameFlow/FoldFlow 协议"** —— 没有测试集，不构成可复现的基准；可以照它的协议报数，但那是"按惯例报"不是"在基准上报"。

**RFdiffusion 25 题 / Genie2 24 题** —— 已被 MotifBench 取代，没有理由再跑旧集合。

**Protein-SE(3)** —— 是训练框架不是评估基准。

### 如果只能选两个

**MotifBench + La-Proteina 的 all-atom co-designability。**

理由是这两个**恰好互补且不重叠**：
- MotifBench 给**外部可信度**（固定题面、排行榜、审稿人认），代价是测不到侧链
- all-atom co-designability 给**claim 对位**（考的就是共设计序列 + 全原子），代价是社区认可度弱

反过来如果只选一个，任一个单独都有明显缺口：只做 MotifBench 等于我们的全原子卖点没被任何数字支撑；只做 all-atom co-designability 则没有公认题面可以横比。

**都先不实现**——等其他类别调研完统一 review。
