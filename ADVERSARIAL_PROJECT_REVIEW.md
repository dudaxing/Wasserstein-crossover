# Wasserstein Crossover 项目对抗式审查报告（第二轮）

**审查者**：Cursor Composer（Auto）  
**审查日期**：2026-06-29  
**审查性质**：只读代码审查 + 测试复跑 + 针对性探针；未修改源代码  
**Git HEAD**：`aed6eff`（Clean L-bracket re-run: connectivity guard + finite HV ref）

**审查对象**：
- 论文：Kii et al. (2026) CMAME 451:118713 — Wasserstein crossover for EA-based topology optimization
- 参考：Kii et al. (2025) IJMS 301 — PH-Wasserstein selection（参考文献 [59]）
- 代码：`src/`、`experiments/`、`README.md` 及既有审查文档

---

## 执行摘要

自首轮审查以来，项目发生了**结构性收缩**与**关键集成修复**：

1. **范围收缩**：已删除裂纹板主实验路径（`run_2d_stress.py`、`StressPlateProblem` 等），仓库现为 **L-bracket + body-fitted HF** 单算例研究原型。
2. **P0 集成修复**（本轮审查确认已在源码中落地）：
   - LF→HF 边界插值：edge-clamp 替代 `fill_value=0`
   - HF 载荷：tributary length 归一化，总载荷与 `h` 无关
   - 连通性守卫：断开子代不再触发 `spsolve` segfault
   - HV 参考点：仅基于有限初始目标，避免 `ref=inf`
3. **叙事修正**：README 诚实撤回 fillet/−32% 等污染结论；`ADVERSARIAL_REVIEW.md` 对 §8–§9 加了 RETRACTION。

**更新后的总体判断**：

> 这是一个**诚实度明显提高**的 Wasserstein 交叉 + L-bracket body-fitted HF 研究原型。算子级机制（`demo_morphing.py`、单元测试）仍可辩护；**在物理正确的 sharp-corner 设定下，clean re-run 的 EA 结论为负**（HV 无改善、matched-volume best-J₁ 平坦）。项目**不支持**论文 framework 级数量复现，也**不支持**「Wasserstein EA 已显著改善 L-bracket 设计」——除非完成 crossover 消融与 held-out mesh 验证。

---

## 一、项目当前定位

| 维度 | 可辩护程度 | 说明 |
|------|-----------|------|
| Wasserstein 交叉算子（Eq. 10–13, Alg. 2, 卷积 Sinkhorn） | **中等偏高** | 有完整单元测试；JAX 误标已修正 |
| EA 框架（Algorithm 3 骨架） | **中等** | 主循环可运行；step (b) 仍未实现 |
| LF：P-norm 应力拓扑优化（Q4 + MMA） | **较好** | 伴随 FD ~2.3e-5；`test_fem.py` 有 assert |
| HF：body-fitted CST + 真 max von Mises | **中等（元素级）** | patch test + MATLAB 同网格对照；集成链仍有缺口 |
| HF：L-bracket EA 作为优化目标 | **低（作正结果）** | clean re-run 负结论；无 crossover 消融 |
| PH-Wasserstein selection | **低（作 [59] 复现）** | inspired 近似，标签仍偏强 |
| 论文 framework 级「Wasserstein >> VAE」 | **不在当前仓库** | 裂纹板路径已删除 |

**最站得住脚的结论**：
- Wasserstein barycenter **运输材料**，优于线性 cross-fade（`demo_morphing.py`）
- Sharp-corner L-bracket 上 EA **不改善** useful designs（负结果，README 与 ADVERSARIAL_REVIEW §1–2 一致）

**应暂停的结论**：
- 早期 fillet「4–5% 改善」「−32% stress LF + elitism」（已 RETRACTION）
- 「contour-conforming body-fitted mesh」
- 「ph_wasserstein = paper-faithful」

---

## 二、相对首轮审查的变更对照

| 首轮 P0 问题 | 当前状态 | 证据 |
|-------------|---------|------|
| `_to_hf_field` 固定端置零 | **已修复（部分）** | edge-clamp + `fill_value=None`；均匀场探针 support=0.4（非 0） |
| `lbracket_bcs` 载荷随 h 缩放 | **已修复** | 探针 h=0.5/1/2/3 总 Fy 均为 −1.0 |
| 断开子代 segfault | **已修复** | `_load_path_ok` 跳过求解，J₁=inf |
| HV ref 因不可行种子 → inf | **已修复** | `framework.py` 仅用 finite init 设 ref |
| fillet 正结果 | **已撤回** | README + ADVERSARIAL_REVIEW RETRACTION |
| 裂纹板 framework 复现 | **已移除** | 仓库仅 L-bracket |
| run manifest / provenance | **仍缺失** | `run_lbracket.py` 无 manifest |
| mesh 共形 | **未修复** | 仍是无约束 Delaunay |
| ph_wasserstein vs [59] | **未修复** | 算法差异仍在 |
| fea_t3 E0 双重缩放 | **未修复** | E0=1 时未触发 |

---

## 三、计算方法与流程（当前版本）

### 3.1 总体架构

```
LF（一次性）: 扫描 (R,V) → P-norm 应力 MMA 优化 → 初始种群 Θ
EA（每代）:
  (a) HF 评价 Θ_tmp → (J₁, J₂)
  (c) 合并 + NSGA-II 两阶段选择 → 保留 N_pop
  (d) 超体积 HV
  (e) Wasserstein 交叉 → N_xo 子代
  (f) Θ_tmp ← 子代
```

交叉在 **LF 结构化网格**（75×75）上操作；HF 将密度重采样到节点网格后 body-fitted 求解。

### 3.2 LF 构造

- **网格**：150×150 域，`nelx_lf=75`，`h=2`
- **被动 void**：右上角 L 形缺角，sharp re-entrant corner (60,60)
- **默认 LF**：`lf_method="stress"`，P-norm (P=8) + SIMP + hat filter + MMA
- **初始种群**：3×6=18 组 (R,V)，R∈[4,10]，V∈[0.30,0.55]

### 3.3 HF 构造

1. `_to_hf_field(gamma)`：单元中心 → HF 节点（edge-clamp 插值）
2. `extract_contour` @ 0.5 → `clean_contour`
3. `generate_bodyfitted_mesh`：背景点拒绝 + Delaunay + DistMesh
4. 三角形质心 `ρ<0.5` → void；载荷块强制 solid
5. `_load_path_ok`：无支撑-载荷连通路径 → J₁=inf（不求解）
6. `fea_t3`：CST FEA → J₁=max(σ_vm)，J₂=实体面积比
7. `hf_seeds` 次随机网格取均值（默认 3）

### 3.4 Wasserstein 交叉

- 种群 L2 距离 → 自适应 ε（Eq. 18–19）
- 卷积 Sinkhorn barycenter（NumPy + `gaussian_filter`）
- min-max 反归一化（Eq. 13）

### 3.5 选择

- 阶段 1：非支配排序
- 阶段 2：设计空间最远点（默认 `diversity`）
- **极值保留**：最后一截断 front 保留各目标 argmin（`keep_extremes`）

---

## 四、P0 级发现（第二轮）

### 4.1 固定端修复不完整：场值正确 ≠ 网格实体连接

**已修复**：edge-clamp 使支撑边场值不再为 0（均匀 0.4 场：support min=max=0.4）。

**仍存问题**：对均匀 0.4 设计，固定节点与实体三角形邻接仍为 **0/31**。README 报告 fillet 设计 **15/34**——改善但未完全解决。固定端节点可能落在 void 单元或界面外侧，仍可能通过 ersatz 路径传力。

**建议**：HF 评价前对支撑/加载区强制实体（与裂纹板 `solid_mask` 类似），并加集成测试。

### 4.2 载荷归一化：已修复，缺自动化测试

本轮探针确认 h=0.5/1/2/3 时 **total Fy = −1.0**。`test_bodyfitted.py` 仍用默认 h=1，**未覆盖 EA 默认 hf_h=2**。

### 4.3 网格非共形：未修复，README 已降级

仍是无约束 Delaunay + 质心二值化；界面可切穿三角形。README 已改为「boundary-refined Delaunay」，但 `bodyfitted.py` 模块头仍写「boundary-conforming」。

### 4.4 clean re-run 负结论：当前最可信的 EA 证据

README 声明（commit `aed6eff`）：
- sharp corner + stress LF + 修复后 HF
- HV **+0.0%**
- matched-volume best-J₁ **平坦**
- 根因：LF↔HF gap 近零；应力最优父代的 barycenter 子代系统性更差；~half 子代 disconnected

**审查意见**：这是比 fillet 正结果**更可信**的叙事，但仍缺：
- 本地 `results/` 未入库（gitignore），无法独立审计 NPZ
- 无 Wasserstein vs linear vs no-crossover 对照
- 单 seed、单 framework run

### 4.5 历史 fillet 数字：必须视为污染数据

§8–§9 的 4–5%、−32% 等数字在 P0 bug 存在时产生；`ADVERSARIAL_REVIEW.md` 已 RETRACTION。审查**不支持**任何基于这些数字的结论。

---

## 五、P1 级发现

### 5.1 Algorithm 3 step (b) 未实现

不可行个体赋 `(inf, inf)` 仍进入选择；与文档不符。连通性守卫使大量子代不可行，可能扭曲选择压力。

### 5.2 `ph_wasserstein` 非 [59] 精确实现

`topo_selection.py` 仍标 paper-faithful；与 [59] 在 SDF、filtration、排序、rank-1 回退上均有实质差异。裂纹板 5-seed 消融脚本已删除，无法在当前仓库复验。

### 5.3 provenance 缺口

- `run_lbracket.py`：自由 `--tag` 缓存，无 git commit / config hash / SHA-256
- `analyze_lbracket.py` 依赖 `results/lbr_result_{tag}.npz`，但 results 未 tracked
- 旧 `run_manifest` 机制随裂纹板路径删除

### 5.4 `fea_t3` E0 双重缩放

`D=elasticity_matrix(E0)` 已含 E0，再乘 `Evec=Emin+ρ(E0−Emin)`；E0≠1 时错误。当前 E0=1 未触发。

### 5.5 测试盲区

| 测试 | 覆盖 | 缺口 |
|------|------|------|
| `test_operator.py` | 算子、选择、HV、elitism | 无 POT/dense-K 对照；均匀场退化 |
| `test_fem.py` | 伴随、应力下降、体积 | 仅 generic mesh，非 L-bracket |
| `test_bodyfitted.py` | CST patch、角度、角点位置 | 不经 `_to_hf_field`/`fea_t3` 集成链 |
| 集成 fixture | **无** | 载荷合力、固定端邻接、连通性 |

本轮复跑：**ALL OPERATOR / FEM / BODY-FITTED TESTS PASSED**。

### 5.6 文档不同步

| 文件 | 问题 |
|------|------|
| `lbracket.py` 模块头 | 仍写 compliance LF，默认已是 stress |
| `run_lbracket.py` 头注释 | 仍写 compliance OC |
| `framework.py` 模块头 | 仍引用已删除的 `StressPlateProblem` |
| `README.md` | 引用不存在的 `assets/bodyfitted_mesh.png` |
| `REPRODUCTION_REVIEW.md` | 描述已删除的裂纹板路径与旧 bug 状态 |
| 本报告旧版 | 仍写 P0 未修复（已由本轮重写替代） |

---

## 六、按科研叙事的对抗性裁决

| 主张 | 裁决 |
|------|------|
| 「实现了 Wasserstein crossover 算子」 | **部分成立** — 机制 + 测试；无独立 POT 参考 |
| 「body-fitted HF 元素 FEA 正确」 | **成立（元素级）** — patch test + MATLAB 对照 |
| 「HF 集成链已验证可用于 EA 结论」 | **不成立** — 固定端邻接不完整；无集成测试；clean re-run 负 |
| 「EA 显著改善 L-bracket 设计」 | **不成立** — 正结果已撤回；clean re-run 平坦 |
| 「复现了 Kii 2026 framework 优势」 | **不在当前仓库** |
| 「PH selection = paper-faithful」 | **不成立** |

---

## 七、Agent 开发过程元审查（第二轮）

**进步**：
- 接受外部审查，修复 P0 集成 bug
- 主动撤回污染结论（RETRACTION）
- 增加连通性守卫、HV ref 修复、elitism 测试
- 收缩范围，避免在错误 HF 上继续堆实验

**仍须警惕**：
- 审查文档（`REPRODUCTION_REVIEW.md`、旧版本报告）未随代码同步更新
- README 引用缺失图片
- 修复后仍用 README 表格宣称「15/34 可接受」，但未说明 uniform 设计仍为 0/31

---

## 八、可辩护 vs 应暂停

### 可辩护
- Wasserstein 算子实现与单元测试
- CST 元素 + MATLAB 同网格对照
- Sharp-corner 负结果的方法论（ADVERSARIAL_REVIEW §1–2）
- P0 bug 修复方向正确（edge-clamp、tributary load、connectivity guard）
- README 对 clean re-run 负结论的诚实表述
- Selection 极值保留修复

### 应暂停
- 一切基于 fillet / −32% / HV +8.7% 的数字
- 「HF fully verified for EA」
- 「mesh-convergent」「diagnosis confirmed」（fillet 语境）
- 无 crossover 对照下的「Wasserstein 有效」

---

## 九、后续优先级建议

| 优先级 | 事项 |
|--------|------|
| P0 | 集成 fixture：`_to_hf_field` + `lbracket_bcs(h=2)` + 固定端邻接 + 载荷合力 |
| P0 | 支撑/加载区强制实体（HF 预处理） |
| P0 | crossover 消融：Wasserstein / linear / no-crossover，同初始种群 |
| P1 | `run_lbracket` manifest（config hash、git commit、NPZ SHA-256） |
| P1 | 修复 `fea_t3` E0 scaling；patch test 走完整 `fea_t3` |
| P1 | 同步审查文档与模块头注释 |
| P2 | constrained mesh 或界面切穿断言 |
| P2 | held-out mesh seeds + 多 seed EA |

---

## 十、最终评价

项目在第二轮审查中展现出**显著的工程诚实度提升**：承认并修复了首轮指出的 P0 集成错误，撤回了基于污染 HF 的正向结论，并将范围收缩到可管理的 L-bracket 单算例。

当前最准确的定位是：

> **Wasserstein 交叉算子的可运行研究原型 + L-bracket body-fitted HF 探索平台**；算子机制值得继续深化，但 **EA 在 sharp-corner L-bracket 上尚无可靠改善证据**；在 crossover 消融、集成测试与 provenance 完成之前，不宜对外宣称 framework 级或 HF 级成功。
>
> **（§13 更新：论文规模 + held-out 验证后，此定位已被推翻——EA 确有可靠改善。）**

---

## 十三、论文规模重跑的最终判定（2026-06-30，held-out 验证）

§12 的负结论是在**小规模**（16–18 设计、15 代）下得出的。按用户要求扩到**论文规模**
（Table 1：N_pop=N_xo=N_lf=100、t_max=100；LF 用 4×25 seeding + 随机多起点提升多样性）
重跑，**结论反转**。

工程上为让 ~7h 长跑能在反复 host teardown 下跑完，新增：连通性守卫（防奇异 spsolve 段错误）、
LF 增量缓存断点续跑、**EA 逐代 checkpoint + resume**（已验证精确续跑）、config-hash 缓存键。
跑程经 33→52→86→100 多次续跑完成（wall≈3045s/段）。

**判定：在物理正确的 sharp-corner + stress-LF + 论文规模下，EA 确实改善设计。**

| 指标 | 初始(LF) | 最终(EA) | 训练种子(0,1,2) | **held-out 种子(5,6,7)** |
|---|---|---|---|---|
| 全局 min J₁ | 0.3539 | 0.3320 | −6.2%(单调) | — |
| HV(有限参考) | 6.3682 | 6.3751 | +0.1%(单调) | — |
| 匹配体积 best-J₁ V0.28–0.42 | — | — | 0%(尖角地板) | **0%** |
| 匹配体积 best-J₁ V0.42–0.47 | — | — | −16.5% | **−5.6%** |
| 匹配体积 best-J₁ V0.47–0.53 | — | — | −11.1% | **−6.2%** |
| 匹配体积 best-J₁ V0.53–0.57 | — | — | −12.2% | **−12.0%** |

**关键诚实点（held-out 抗 winner's curse 检验）**：EA 用种子 0/1/2 训练，故在这三种网格上的
增益部分是**网格运气**——换到 held-out 种子 5/6/7，中/低体积档增益缩水（−16.5%→−5.6%、
−11.1%→−6.2%），但**仍为真实正增益**（高于 ~3–5% 网格噪声 CV）；高体积档 V0.53–0.57
**几乎不缩水（−12%）**，最稳健。低体积档（<0.42）维持在尖角奇异地板，不变。

**机制**：100 个多样化（随机多起点）LF 种子提供了足够拓扑多样性，使 Wasserstein 重心混合能在
中高体积找到更优的材料分布（见 `assets/lbracket_paper_compare.png`：等体积下峰值应力从奇异
内杆被重分布开）。小规模跑之所以"flat"，是种子太少、多样性不足、代数不够——**不是**算子或问题
的根本缺陷。这也修正了 §12 与 ADVERSARIAL_REVIEW §2(b) 的过度悲观。

**仍未关闭**：crossover 消融（Wasserstein vs linear vs 无交叉，以证明是*算子*而非任意混合在起作用）；
多 EA 随机种子；CST 低阶元 + structured→body-fitted 重采样噪声。报告增益时**以 held-out 数字为准**。

---

*本地副本：`ADVERSARIAL_PROJECT_REVIEW.md` · 审查时测试全部通过 · 最终判定 Git 见 §13 提交*
