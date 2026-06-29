# Wasserstein Crossover 项目对抗式审查报告

**审查性质**：只读；未修改任何源代码、测试或实验脚本。  
**审查日期**：2026-06-29  
**审查对象**：论文 Kii et al. (2026) CMAME 451:118713 的 Python 复现项目；selection 细节对照 Kii et al. (2025) IJMS 301（参考文献 [59]）。  
**审查范围**：`README.md`、`REPRODUCTION_REVIEW.md`、`ADVERSARIAL_REVIEW.md`、`src/`、`experiments/`、本地缓存结果与对抗性探针证据。

---

## 执行摘要

该项目是一个**工程覆盖面较广、核心机制可运行、但物理集成与实验审计尚不稳定**的研究原型。Wasserstein 交叉算子、LF 拓扑优化、EA 闭环和实验脚手架具有实际价值；裂纹板算例上**算子级**（Fig. 1c 风格）优势是项目最站得住脚的正面结论。

然而，对抗性审查显示：新增 L-bracket/body-fitted 路径存在**结果级 blocker**（固定端空层、载荷未归一化、网格非共形）；`ph_wasserstein` 并非 [59] 的精确实现；fillet 正结果在 held-out mesh seeds 上衰减或反转；缓存/manifest 仍可能产生错误 provenance。README 中部分强断言（「HF 已验证」「diagnosis confirmed」「4–5% 真实改善」「mesh-convergent」）**超出当前证据**。

**现阶段最准确的项目定位**：

> 一个可运行、可测试、可做消融的 Python 研究平台；它支持 Wasserstein crossover 的**定性机制结论**，但不支持论文级 framework 排序、精确 PH selection、已验证 body-fitted HF 或完整数量复现。

---

## 一、项目定位（审查后的准确表述）

| 维度 | 可辩护程度 | 说明 |
|------|-----------|------|
| Wasserstein 交叉算子（卷积 Sinkhorn + Eq. 10–13） | **中等偏高** | 机制与论文方程对齐，有确定性单元测试 |
| EA 框架闭环（Algorithm 3） | **中等** | 主循环结构正确，但 step (b) 未实现、无 best-ever archive |
| 裂纹板 LF 拓扑优化 + Q4 FEM | **较好** | 伴随灵敏度 FD 验证 ~1.3e-5 |
| 裂纹板 HF 评价（论文 Section 5.1） | **低（作数值复现）** | 全套代理，决定 framework 排序上限 |
| PH-Wasserstein selection | **低（作 [59] 复现）** | 可运行 inspired 近似，非精确实现 |
| L-bracket body-fitted HF | **低（作已验证 HF）** | CST 元素可用，集成链存在 P0 级错误 |
| 论文 framework 级「Wasserstein >> VAE」 | **未复现** | 默认配置下 linear 最优 |

**最站得住脚的正面结论**：算子级——Wasserstein 插值能**运输材料**、在选手工父代对上可达低于双亲的应力并 Pareto-支配高体积父代。

**最不可靠的宣称**：L-bracket fillet 后「4–5% 真实改善」「诊断已确认」「mesh-convergent」「HF 已验证」。

---

## 二、编程 Agent 开发过程的元审查

编程 Agent 在本项目中表现出典型的「实现 → 正向叙事 → 自我审查 → 再正向叙事」循环：

```
实现模块与实验脚本
    → README 写入强结论
    → REPRODUCTION_REVIEW / ADVERSARIAL_REVIEW 自我审查
    → 设计 falsification 实验
    → README 再次写入更强结论
    → 对抗性探针反驳部分结论
```

### 值得肯定的 Agent 行为

- 主 README 对裂纹板 framework 级未复现的声明诚实（「not 1:1 numerical reproduction」）。
- 主动撰写 `ADVERSARIAL_REVIEW.md`，记录 sharp-corner 负结果与 falsification 设计。
- 增加 `test_operator.py`、`run_manifest`、`multiseed_selection.py` 等可审计基础设施。
- 在 selection 中发现并修复「diversity 截断丢弃最优 J₁」的 elitism bug。

### 值得警惕的 Agent 行为

- **每完成一层工程就升级结论强度**：元素 patch test 通过 →「HF correct and verified」；in-sample 3-seed 改善 →「diagnosis confirmed」「真实 4–5% gain」。
- **自我审查与对外文档不同步**：`REPRODUCTION_REVIEW.md`（6/29）已用探针反驳 fillet 正结果，但 `README.md` L-bracket 段仍写「monotonically +8.7%」「diagnosis confirmed」。
- **标签通胀**：`ph_wasserstein` 标为「paper-faithful」；`--paper` 标为 exact settings；`wasserstein.py` 声称 JAX backend 自动启用但代码中不存在。
- **因果归因跳跃**：fillet + 3-seed averaging + stress LF + elitism fix 同时改变，却归因于「Wasserstein EA 有效」或「falsification 确认」。

---

## 三、P0 级发现（直接影响结论有效性）

### 3.1 L-bracket LF→HF 固定端变成空材料层

`src/lbracket.py` 中 `_to_hf_field()` 用元素中心网格插值到 HF 节点网格，`fill_value=0.0`。域边界（含 `y=L, x<=lpd` 固定端）落在插值范围外 → 被置零 → 固定通过 `Emin=1e-9` 软层传力，而非实体支撑。

对抗性探针（缓存 fillet 设计）：

```text
support field at y=L, x<=lpd: min=0, max=0, mean=0
fixed nodes incident to a solid triangle: 0 / 33
```

`experiments/test_bodyfitted.py` 直接在边界节点赋 1，**绕过了此路径**，故现有测试无法捕获。

**影响**：J₁ 误差与 README 报告的 4–5% 改善同量级；当前 L-bracket EA 数字应视为**受污染的探索性结果**。

### 3.2 HF 载荷未归一化，随网格间距变化

`src/bodyfitted.py` `lbracket_bcs()` 对每个载荷节点赋 `-F0/lload`，未乘节点间距或最后归一化合力：

| HF boundary spacing h | load nodes | actual total Fy (F0=1) |
|---:|---:|---:|
| 0.5 | 13 | -2.0 |
| 1.0 | 7 | -1.0 |
| 2.0 | 4 | -0.5 |
| 3.0 | 3 | -0.4167 |

EA 默认 `hf_h=2.0` 时总载荷约为 LF 的一半（-0.5 vs -1.0）。单元测试用 `h=1` 恰好正确，**未覆盖 EA 实际配置**。

### 3.3 「body-fitted」实为无约束 Delaunay + ersatz void

`generate_bodyfitted_mesh()` 将轮廓点加入点云后直接 `Delaunay(p)`，无 constrained segments / PSLG / 域裁剪。对抗性探针：

- 389 / 9228（4.2%）三角形节点同时落在界面两侧（0.5 contour 切穿单元）
- 约 27% 单元标为 void 但仍保留在刚度 mesh

README 写「mesh conforming to the material contour」**超出当前证据**。更准确描述：

> boundary-refined full-domain Delaunay/ersatz mesh prototype; contour conformity not yet established.

### 3.4 fillet 正结果未通过 held-out mesh seeds

EA 用 seeds 0–1–2 优化与报告；held-out seeds 3–9 上：

| HF J₂ bin | 报告用 seeds 0-2 | held-out seeds 3-9 |
|---|---:|---:|
| [0.40, 0.45) | 0.2768 → 0.2652 (-4.2%) | 0.2814 → 0.2767 (-1.7%) |
| [0.45, 0.50) | 0.2569 → 0.2449 (-4.7%) | 0.2601 → 0.2966 (+14.0%) |

存在 winner's curse；第二档某 held-out seed 上 J₁=0.547，显示对随机网格极不稳定。

### 3.5 「mesh-convergent」与「HV 单调」被数据反驳

同一 fillet、同一 `minedge=3`，仅改 DistMesh `n_iter`：

| n_iter | mean J1 | std across 3 seeds |
|---:|---:|---:|
| 20 | 0.1881 | 0.0097 |
| 40 | 0.1910 | 0.0120 |
| 80 | 0.1964 | 0.0054 |
| 120+ | 0.2055 | 0.0219 |

保存的 fillet run HV 在 t=12 为 4.509648、t=13 为 4.509529（下降）；「monotonically +8.7%」为事实错误。

### 3.6 `ph_wasserstein` 非 [59] 精确实现

| 环节 | [59] | 当前 `topo_selection.py` |
|------|------|--------------------------|
| 二值化 | HF 预处理一致的材料场 | raw `>0.5` |
| SDF | Manhattan/L1 | Euclidean EDT |
| Filtration | void sublevel H0 | superlevel H0+H1 |
| essential class | `(birth, infinity)` 不进入 PD | 人为有限化并保留 |
| Wasserstein | W₂, p=2, q=2 | q=2, 默认 L∞ ground |
| 排序 | 行和 ΣⱼWᵢⱼ 降序 | 贪心 farthest-point |
| rank-1 回退 | crowding | **无** |

5-seed 消融显示与 L2 diversity 无显著差异（n=5 功效低），**不能**支持「统计等价」或「selection 无关」。

### 3.7 缓存/manifest provenance 缺口

`run_2d_stress.py` 缓存键仅 `method + nelx + t_max + sel_mode`，不含 seed、ε 范围、VAE epochs 等；manifest 在加载缓存**之后**生成，可能把旧结果标成新 commit。`run_lbracket.py` 无 manifest，用自由 `tag` 命名。

本地证据：`run_manifest_60_t30.json` 记录 commit `97073c1...`，但三种结果都标记为 `loaded_from_cache=true`，缓存文件时间早于该 commit 的 manifest。

---

## 四、P1 级发现（削弱可比性与可复现性）

### 4.1 Algorithm 3 step (b) 未实现

`src/framework.py` 文档写 drop constraint-violating candidates，实现仅为不可行个体赋 `inf` 目标后仍进入选择，与文档/论文不一致。

### 4.2 裂纹板 HF 代理决定 framework 排序

`StressPlateProblem.hf_evaluate()`：hat 滤波 + 硬二值化 + Q4 单元中心 `max(ρ^q·σ)`，非 COMSOL body-fitted P2。

默认 demo 结果（seed=0）：

| Method | min J₁ | HV improvement |
|---|---:|---:|
| initial | 20.08 | — |
| Wasserstein | 17.16 | +2.3% |
| linear | **16.94** | +3.3% |
| VAE | 17.72 | +4.8% |

与论文相反——应优先归因于 HF 景观不同，而非 crossover 失效。HV 被退化极端点 (J₁≈178, J₂≈0.283) 主导，排序不宜过度解读。

### 4.3 Wasserstein 算子边界与文档错误

- 模块头声明 JAX backend 自动启用，**代码仅有 NumPy + gaussian_filter**。
- 均匀父代 → min-max 分母为零 → **全零子代**。
- `λ=1` 不恢复父代（探针：随机父代下 max abs error ≈ 0.57，correlation ≈ 0.61）。
- 卷积 Gaussian 使用 `mode="reflect"` 截断核；**无 POT/dense-K 参考对照**。

### 4.4 VAE 基线非严格可比

默认 80 epoch vs 论文 500；每代 warm-start 权重但重置 Adam moments；framework seed 与 VAE seed 硬编码为 0；无 multi-seed Wasserstein/VAE/linear 对照。

### 4.5 L-bracket 无法归因于 Wasserstein crossover

`run_lbracket.py` 仅「Wasserstein EA vs 初始种群」，无 linear/VAE/random/no-crossover 对照；fillet 与 averaging 无 2×2 factorial；单 framework seed。

### 4.6 测试强度不均衡

- `test_fem.py`：`test_lf_stress_small` **无 assert**；`test_compliance_mma` 未真正调用 MMA。
- `test_bodyfitted.py`：patch test 不经过 `fea_t3()`；`fea_t3` 存在 E0 双重缩放（E0≠1 时错误）。
- 无 pytest/CI；PH 测试在缺 TDA 时 exit 0 skip。

### 4.7 统计措辞过度

README 写 PH 与 L2「statistically indistinguishable」「does not materially change」。n=5 时配对均值差 95% CI 仍包含可能有意义的差异（例如 min J₁：[-0.723, 0.735]）。应改为「初步实验中未检测到显著差异」。

---

## 五、P2 级发现（文档、图像与维护）

- `experiments/fig1_morphing_comparison.py` 仍写「DOMINATE both parents」，与已修正的 README 矛盾。
- Fig. 8 显示原始 `g`，J₁/J₂ 来自 filtered+binarized 场——密度与目标不一致。
- `assets/`（tracked）与 `results/`（gitignored）无 hash 绑定，图可能来自不同运行。
- TDA 环境依赖 monkey patch（假 `gph`、替换 `gudhi.CubicalComplex`），脆弱；`available()`/`select()` 捕获所有 `Exception` 静默回退 L2。

---

## 六、按科研叙事线的对抗性裁决

### 6.1 「我们复现了 Wasserstein crossover 算子」

**部分成立**。卷积重心、自适应 ε、min-max 反归一化已实现并有性质测试。但缺独立参考实现验证；端点退化行为未在论文中定义；生产路径与 Fig. 1 demo（固定 ε、不走 adaptive_eps）不一致。

### 6.2 「我们复现了论文 Section 5.1 的 framework 优势」

**不成立**。HF、selection、规模、VAE 均为替代/缩减；默认排序与论文相反；HV 被退化极端点主导。

### 6.3 「我们实现了 paper-faithful PH selection」

**不成立**。算法链路与 [59] 多处实质差异；5-seed 负结果仅说明**当前近似**在 surrogate 上无效，不能证明 [59] 精确算法无效。

### 6.4 「body-fitted HF 已验证，EA 在 L-bracket 上获得真实改善」

**当前证据不支持**。P0 集成错误使 in-sample 4–5% 改善不可作为物理结论；held-out seeds 衰减或反转；无 crossover 消融。

### 6.5 「对抗性审查本身可信吗？」

`ADVERSARIAL_REVIEW.md` 对 sharp-corner 失败的诊断（奇异性 + mesh 噪声）**结构合理且有测量支撑**；但 §8–9 将 falsification 结论写得过强，且未与 `REPRODUCTION_REVIEW.md` 6/29 的集成反例交叉校验——**两份审查文档之间存在张力**，应以带探针证据的 `REPRODUCTION_REVIEW.md` 为准。

---

## 七、可辩护 vs 应暂停的结论清单

### 可辩护（有代码+测试+诚实文档支撑）

- Wasserstein 重心在合成形状上展示材料运输，优于线性 cross-fade。
- 选手工父代对上，Wasserstein 插值可达低于双亲应力、支配高体积父代。
- LF 应力优化 + 伴随验证在结构化网格上可工作。
- Sharp-corner L-bracket 上 EA **不改善** useful designs（负结果可信）。
- Selection elitism bug 的发现与修复是真实工程贡献。

### 应暂停或降级

- L-bracket fillet 后「4–5% 真实改善」「diagnosis confirmed」
- 「mesh-convergent」「HV monotonically」
- 「HF correct and verified」「conforming body-fitted mesh」
- 「ph_wasserstein = paper-faithful」
- 「PH 与 L2 统计等价」
- 「stress LF + elitism 使 EA 成为 genuinely effective optimizer」（−32% 等数字未经 holdout/crossover 对照验证）
- README 中「HF gap dominates」的因果断言（plausible hypothesis，非已证结论）

---

## 八、当前结果应如何解释

### 默认三方法结果（裂纹板 surrogate）

framework 级未复现论文排序；算子级 Wasserstein 插值优势仍可观察。HV 对 reference point 和退化极端点高度敏感。

### PH selection 5-seed 结果

当前 PH+farthest-point 与 L2 farthest-point 样本均值非常接近——有价值的**初步负结果**，但不能支持「统计等价」或「[59] 精确 selection 无效」。

### L-bracket/body-fitted 结果

sharp-corner 负结果结论仍有参考价值。fillet in-sample 缓存结果存在，但基于当前审查只能归类为**受污染的探索性结果**，不能作为「HF 已补齐」或「Wasserstein framework 优势恢复」的证据。

---

## 九、后续改进建议（供未来工作参考，本次未实施）

1. **冻结 L-bracket 正结果引用**，先修复 support/load/mesh conformity 并重跑。
2. **统一文档强度**：README 与审查报告对齐；撤回 inflated 标签。
3. **provenance 闭环**：config hash 进缓存名；`run_lbracket` 接入 manifest。
4. **独立参考测试**：小网格 dense-K/POT barycenter；集成 fixture 覆盖 `_to_hf_field`。
5. **严格对照实验**：Wasserstein vs linear vs VAE，paired multi-seed；fillet×averaging 2×2 factorial；held-out mesh seeds。
6. **实现真正的 [59] selection mode**（或明确永久标为 inspired approximation）。

**在 P0 集成问题与 holdout 验证完成前，不应投入 paper-scale 计算或对外宣称 framework/HF 优势恢复。**

---

## 十、最终评价

编程 Agent 交付了一个**对主裂纹板叙事 unusually 诚实、却又在新功能上重复「过早结论化」模式**的研究原型：核心 Wasserstein 机制值得保留与深化，但 L-bracket 路径的物理集成错误、实验审计缺口与文档-证据张力，使当前项目**不足以支撑论文级数量复现或 HF 已补齐的宣称**。

最可靠的科学产出仍是：

1. **算子级定性优势**（材料运输、插值应力改善）；
2. **sharp-corner 负结果**及其 falsification 方法论；
3. 对 Agent 驱动科研中「实现—宣称—审查—再宣称」循环的警示。

---

*本报告为只读对抗式审查产物。详细探针数据与逐项核查见 `REPRODUCTION_REVIEW.md`；L-bracket 专项诊断见 `ADVERSARIAL_REVIEW.md`。*

---

## 十一、回应与 P0 修复（2026-06-29，由被审查方执行）

逐条核对后**确认本报告准确**。关键发现已用同样的探针独立复现:

- **3.1（固定端变空）属实且严重**:HF 场在 y=L,x≤lpd 全为 0;33 个固定节点 0 个连实体;**max|U|≈8.1e8**(结构悬空)。
- **3.2（载荷未归一化)属实**:总 Fy 随 h:0.5→−2.0、1→−1.0、2→−0.5、3→−0.333。
- **3.5**:fillet run HV 在 t=12 确有下降,"monotonically" 为事实错误。
- **3.6 / 4.3 / P2**:`ph_wasserstein`≠[59];`wasserstein.py` 的 JAX backend 声明虚假(只有 NumPy+gaussian_filter);`fig1` 仍写 DOMINATE。均属实。
- 补充:审查对 sharp-corner 负结果偏宽容——它走同一 `_to_hf_field`,**同样被 3.1 污染**,其"数值"亦不可信。

**已执行的修复(P0)**,并按要求用 max|U| 与"固定节点连实体"验收:

| Bug | 修复 | 验收 |
|---|---|---|
| 3.1 固定端变空 | `_to_hf_field` 边界 edge-clamp 外插(材料延伸到域边界) | 固定节点连实体 **0/33 → 15/34**;**max|U| 8.1e8 → 2.8e2** |
| 3.2 载荷未归一化 | `lbracket_bcs` 一致节点载荷(tributary length) | 总 Fy = **−1.0**(h=0.5/1/2/3 全部) |

**已执行的清理与冻结**:
- 项目裁剪为**仅 L-bracket 算例**(删除裂纹板 `StressPlateProblem`、`run_2d_stress`、`fig1_morphing_comparison`、`multiseed_selection`、cracked-plate 网格/BC),**取消 fillet**(sharp re-entrant corner)。
- **撤回**所有受污染的 L-bracket 强断言(−32% / −10.5% / diagnosis confirmed / genuinely effective / monotonic / mesh-convergent / HF verified / paper-faithful PH);README 已重写为诚实状态。

**待办(按本报告 §9)**:在修复后的 HF 上**重跑** L-bracket EA 并重新判定是否有真实改善;加 crossover 消融与 held-out mesh seeds;改 JAX docstring、`fig1` 文案、统计措辞;provenance 闭环。重跑判定完成前不对外宣称 HF/framework 优势恢复。

---

## 十二、修复后清洁重跑的判定（2026-06-29，被审查方执行）

重跑前先解决了一个**会导致整进程崩溃**的问题:修复 3.1 后,Wasserstein 后代中
出现「悬空/断开」结构(实体不连接支撑或载荷),`spsolve` 对近奇异系统在 C 层
**段错误(exit 139)**,Python `try/except` 无法捕获,首次重跑在 t=3 崩溃。
已加入**载荷路径连通性守卫** `_load_path_ok`(在求解前用 solid 子网格的连通分量
判断「某固定节点」与「某载荷节点」是否同属一个连通体):不连通则记 `J1=inf`
直接跳过求解。冒烟测试:整 L 形可解(J1 有限);悬空块、两断开块均返回 inf 且**不崩溃**。

**清洁配置**:sharp re-entrant corner(无 fillet)、stress LF(P-norm)、修复后 body-fitted HF、
3 mesh-seed 平均、N_pop=N_xo=16、t_max=15。整 15 代跑完(wall≈565 s,exit 0,无崩溃)。
另修复 HV 参考点 bug(单个不可行初始种子曾把 `ref` 推到 +inf → HV 全为 inf),
现 `ref` 只取**可行(有限)初始设计**。

**判定:在该配置下,EA 不能有意义地改善设计。** 证据:

| 指标 | 初始(LF) | 最终(EA) | 变化 |
|---|---|---|---|
| 超体积 HV(有限参考点) | 4.9791 | 4.9807 | **+0.0%** |
| 全局 min J₁ | 0.3737 | 0.3580 | −4.2%(仅 t=1 单步,之后 14 代冻结) |
| 匹配体积 best-J₁(0.28–0.53 各档) | — | = 初始设计 | **0%** |
| 匹配体积 best-J₁(0.53–0.57 档) | 0.3798 | 0.3580 | −5.7%(单个幸运后代,落在 ~3–5% 网格噪声 CV 内 → 不稳健) |

**机制(已用探针量化)**:
- 从最终种群新生 40 个 Wasserstein 后代:**19/40 不可行**(断开,被守卫拒绝);
- 可行的 21 个中,**0 个**优于现任最优(0.358);后代 J₁ 中位数 **0.591** ≫ 双亲;
- 即:对**已被 stress-LF 优化到接近最优**的双亲做重心混合,系统性地**变差**
  (边界错配产生新的应力集中 + 细杆 + 断开)。
- max 应力位置并非总在尖角:部分设计 max 在支撑边/载荷端(dist-to-corner 60–90),
  故「尖角奇异地板」只是部分原因;**更根本的原因是 LF 与 HF 几乎同目标(都最小化应力),
  LF↔HF 间隙太小,EA 无可利用空间**——这正是当初把 LF 从 compliance 改成 stress 时
  埋下的张力。

**结论(取代 §8–§9 中被撤回的污染数字)**:在物理正确、用户指定的 sharp-corner + stress-LF
配置下,body-fitted-HF 的 Wasserstein-crossover EA **没有改善设计**。这是一个干净、可信的
**负结果**,且机制清楚。早先的「−32%/−10.5%/genuinely effective」是 buggy HF(悬空结构)
叠加 fillet 的产物,已撤回。要让 EA 真正起作用,需(a)恢复真实 LF↔HF 间隙(如 compliance-LF),
或(b)按 §6 falsification plan 改问题(fillet / p-norm / 释放尖角)——但用户已明确不要 fillet。
下一步**科学上最有判别力**的动作是 crossover 消融(Wasserstein vs linear vs 无交叉),
以分离「算子无效」与「该问题本身无改善空间」。
