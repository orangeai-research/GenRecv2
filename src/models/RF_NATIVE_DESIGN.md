# RF-Native 多模态推荐：简约设计

## 一、原则

- **简约**：只保留与 RF 直接相关的三个模块，不做频谱、不做多尺度全局/局部（SMORE、LGMRec 已做）。
- **优雅**：图条件 + 双流对齐 + 流式去噪，三条线清晰、可组合。
- **高效**：单次前向、单次反向；条件维度可控，ODE 步数可调。

---

## 二、三个核心模块

| 模块 | 全称 | 一句话 |
|------|------|--------|
| **GC-RF** | Graph-Conditioned Rectified Flow | 速度场以图嵌入为条件：`v(x_t, t, c_mm, c_graph)`。 |
| **MSDF-CMA** | Modality-Specific Dual Flow + Cross-Modal Aligner | 图像流、文本流分别生成，再用流做跨模态对齐并融合。 |
| **CFD** | Causal Flow Denoiser | 一条流从噪声/脏嵌入映射到干净嵌入，替代 IPW-GCN。 |

---

## 三、GC-RF：图条件流

- **输入**：`x_t`, `t`, `condition_mm`（多模态 concat）, `graph_embed`（GNN 在 adj 上对 [user;item] 的嵌入）。
- **输出**：`v_pred`。
- **接口**：`velocity_net(x_t, t, concat(condition_mm, graph_embed))`；条件维 = `dim_mm + dim_graph`。
- **用途**：主表示由 GC-RF 生成时，图结构显式参与生成，无需再靠多模态间接带图信息。

---

## 四、MSDF-CMA：双流 + 流对齐

- **双流**：  
  - Flow_image：`z0 → z_image`，condition = explicit_image_embeds。  
  - Flow_text：`z0 → z_text`，condition = explicit_text_embeds。  
- **对齐**：现有 `RectifiedFlowAligner`：学习 image↔text 的传输，对齐后融合，如 `z = (z_image + z_text_align) / 2` 或可学习加权。
- **接口**：  
  - `generate(condition_image, condition_text)` → 双流生成 → aligner 融合 → `z_main`。  
  - 损失：flow_matching_image + flow_matching_text + aligner_loss。
- **用途**：主表示由双流生成、流内对齐，替代“单流 + 多模态 concat”。

---

## 五、CFD：流式去噪

- **输入**：noisy_emb（或 z0）；可选 condition（如 propensity/rating）。
- **输出**：denoised_emb = ODE_solve(noisy_emb, v_denoise; condition)。
- **训练**：flow_matching(noisy_emb, clean_emb)，clean_emb 来自高评分或 IPW 聚合。
- **用途**：  
  - **方案 A**：CFD 输出作为主 RF 的 target（teacher）。  
  - **方案 B**：主表示 = 主 RF；CFD 仅作辅助 loss（对 batch 构造 noisy/clean 对）。

---

## 六、端到端架构（简约版）

```
[Raw] user_emb, item_emb, v_feat, t_feat, adj
         │
         ▼
[编码] 多模态 → explicit_image, explicit_text  （space_trans + conv_ii + R，与现有一致）
         │
         ├──► [图编码] graph_embed = GNN(adj, [user_emb; item_emb])  （1 层即可）
         │
         ▼
[主生成] 二选一：
   (1) GC-RF(z0; condition_mm, graph_embed) → z_main
   (2) MSDF-CMA(cond_image, cond_text) → z_main
         │
         ▼
[可选] CFD：teacher 或辅助 loss
         │
         ▼
[BPR + flow_matching + 可选 aligner/CFD] 一次 backward
```

- **不引入**：频谱编码、多尺度流、全局/局部双路。
- **融合**：若用 MSDF-CMA，融合在双流 + aligner 内完成；若用 GC-RF，条件已含多模态 + 图，单流即可。

---

## 七、损失（统一写法）

- **主**：`L = BPR(z_main) + λ_reg · reg`
- **RF**：`L += λ_rf · flow_matching_loss`（GC-RF 或 MSDF-CMA 的 MSE(v_pred, v_target)）
- **MSDF-CMA**：`L += λ_align · aligner_loss(z_image, z_text)`
- **CFD**：若作 teacher 则主 RF 的 target = CFD(noise)；若作辅助则 `L += λ_cfd · flow_matching_denoise`

单次 `forward` + 单次 `backward`，所有参数由主 optimizer 更新。

---

## 八、实现顺序

1. **GC-RF**：在现有 velocity_net 上增加 `graph_embed` 的 concat 条件；1 层 GNN 提供 graph_embed。
2. **MSDF-CMA**：双 VelocityNet（image/text 各一）+ 复用 RectifiedFlowAligner；generate 与损失对齐文档。
3. **CFD**：小 velocity_net，target = 高评分或 IPW 聚合嵌入；接入 teacher 或辅助 loss。
4. **RFNR 主类**：选 (1) 或 (2) 作为主生成，可选 CFD，串联编码 → 生成 → BPR + 上述损失。

---

## 九、与现有工作区分

| 项目 | RFGUME / E2E-RFGUME | RF-Native Rec (简约版) |
|------|---------------------|-------------------------|
| 主干 | GUME + RF 插件 | GC-RF 或 MSDF-CMA（+ 可选 CFD） |
| 图 | GUME 的 norm_adj + ii_adj | 图进条件 (GC-RF) 或仅编码 (MSDF-CMA) |
| 多模态 | GUME 的 coarse-fine + UM | 双流 + 流对齐 (MSDF-CMA) 或条件 concat (GC-RF) |
| 去噪 | CausalDenoiser (IPW-GCN) | CFD（流式去噪） |
| 频谱/多尺度 | 无 | 无（刻意不做，保持简约） |

设计只保留 **GC-RF、MSDF-CMA、CFD** 三个模块，简约、优雅、高效，便于实现与消融。
