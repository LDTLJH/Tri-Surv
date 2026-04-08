# Tri-Surv 代码审查与改进计划书

> 作者：Claude (AI Assistant)  
> 日期：2026-04-07  
> 项目：Tri-Surv — 三模态生存分析模型

---

## 一、代码现状总览

本项目基于 TMI 2026 论文 **DeReF** 的框架进行改进，构建了一个名为 **Tri-Surv** 的三模态（Pathomics + Genomics + Clinical）生存分析模型。创新方向包括：变分信息瓶颈（VIB）提纯特征、cVAE 缺失模态补全、Cross-Attention 自适应融合。

**核心文件清单：**

| 文件 | 作用 |
|------|------|
| `datasets/dataset_survival.py` | 数据集加载，支持多模态组合 |
| `models/tri_surv.py` | Tri-Surv 主模型架构 |
| `models/sit_loss.py` | SIT-Loss：HSIC、SurvNCE、SlicedWasserstein |
| `main_trisurv.py` | 训练主循环 + TriSurvEngine |
| `utils/util.py` | DataLoader 工具函数 |
| `utils/options.py` | 命令行参数解析 |
| `models/deref/network.py` | 原始 DeReF 架构（参考） |

---

## 二、发现的问题与状态

> 状态说明：✅ 已修复 | 🔄 进行中 | ⏳ 待处理

### 🔴 P0 — 致命 Bug

| 编号 | 问题 | 状态 | 说明 |
|------|------|------|------|
| Bug 1 | `dataset_survival.py` 中 `trimodal` 分支缺失实现 | ✅ **已确认正常** | 该分支（第 253–290 行）已有完整实现，不是 bug |
| Bug 2 | `imputer` 模块调用逻辑只覆盖 3 种缺一个模态的情况，缺两个模态时无法处理 | ✅ **已修复** | 重构为统一的 `cVAE_Imputer` 类，覆盖所有 7 种组合 |
| Bug 3 | 缺失模态的 embedding 用零向量初始化 → 梯度消失 | ✅ **已修复** | 改为 `torch.randn(1, dim) * 0.02` 随机初始化 |

### 🟡 P1 — 重大功能缺陷

| 编号 | 问题 | 状态 | 说明 |
|------|------|------|------|
| Issue 1 | `batch_size=1` 无梯度累积，训练不稳定 | ✅ **已修复** | 新增 `--accumulation_steps=8`，effective batch=8 |
| Issue 2 | 验证时只测全模态，未体现缺失模态鲁棒性 | ✅ **已修复** | `validate()` 现测试 full + 3 种单模态缺失 |
| Issue 3 | 无早停机制，容易过拟合 | ✅ **已修复** | 新增 `EarlyStopping` 类，默认 patience=10 |
| Issue 4 | `args.resume` 定义但未实现加载逻辑 | ✅ **已修复** | 每个 fold 保存 `model_last.pth`，支持断点续训 |
| Issue 5 | 训练过程无可视化记录 | ✅ **已修复** | 集成 TensorBoard `SummaryWriter`，记录所有关键指标 |
| Issue 6 | SIT Loss 数值不稳定（NaN 风险） | ✅ **已修复** | 三个损失函数均加入 `clamp` 钳位，SurvNCE 增加对角线 mask |

### 🟡 P2 — 架构层面的改进

| 编号 | 改进 | 状态 | 说明 |
|------|------|------|------|
| Imp 1 | Cross-Attention 融合替代加权拼接 | ✅ **已实现** | 新增 `CrossModalAttention` 类，三个模态之间做信息交互 |
| Imp 2 | 临床特征编码器升级 | ✅ **已实现** | 保留原有架构，但加入 `nn.LayerNorm` 替代 `nn.BatchNorm` |
| Imp 3 | 损失超参数可配置化 | ✅ **已实现** | 新增 `--loss_kl`、`--loss_hsic`、`--loss_nce` 参数 |
| Imp 4 | 消融实验框架 | ✅ **已实现** | 新增 `--ablation` 参数，支持 9 种消融模式 |

---

## 三、已完成修改详解

### 3.1 `models/tri_surv.py` — 架构全面重构

**主要变更：**

1. **统一的 `_get_z()` 辅助函数**：同时处理 VIB 和非 VIB 模式，代码更简洁

2. **`cVAE_Imputer` 类**：完全重写，覆盖所有 7 种缺失组合：
   - 缺 1 个模态：分别有独立的 decoder
   - 缺 2 个模态：共用一个 decoder 从单一模态生成两个
   - 全缺失：有合理的 fallback

3. **`CrossModalAttention` 类**（新增）：
   ```
   输入: [z_p, z_g, z_c] 拼接 → QKV 变换 → Multi-Head Attention → 残差连接 → 输出各模态融合表示
   ```
   每个模态在生成自己的表示时，已看过其他模态的信息。

4. **`GatedFusion` 类**：保留了原有的门控融合机制，与 Cross-Attention 配合使用。

5. **`TriSurv.forward` 流程**：
   ```
   特征提取 → VIB 压缩 → cVAE 补全 → Cross-Attention → 门控融合 → 分类器
   ```

6. **零向量初始化修复**：`nn.Parameter(torch.randn(1, dim) * 0.02)`，方差小但非零，梯度正常流动。

---

### 3.2 `models/sit_loss.py` — 数值稳定性修复

| 损失函数 | 修复内容 |
|----------|---------|
| `HSIC_Loss` | 加入 `clamp(hsic, 0, 10)` 防止数值爆炸 |
| `SurvNCE_Loss` | 对角线 mask（避免自身配对）+ clamp 钳位 + `sigma_min` 下界 |
| `SlicedWassersteinLoss` | 加入 None 检查 + clamp 钳位 |

---

### 3.3 `utils/options.py` — 命令行参数扩展

新增参数：

```
--accumulation_steps   梯度累积步数（default: 8）
--patience             早停耐心值（default: 10）
--clip_grad            梯度裁剪阈值（default: 1.0）
--loss_kl              KL 损失权重（default: 0.01）
--loss_hsic            HSIC 损失权重（default: 0.05）
--loss_nce             SurvNCE 损失权重（default: 0.1）
--ablation             消融模式（9种）
--log_interval         TensorBoard 记录间隔（default: 10）
--save_dir             结果保存目录（default: results_trisurv）
```

`--ablation` 支持的模式：
- `no_clinical`：去掉临床模态（path + geno 双模态）
- `no_vib`：去掉 VIB 信息瓶颈
- `no_cvae`：缺失模态直接填零
- `no_cross_attn`：用原始加权拼接替代 Cross-Attention
- `no_hsic`：去掉 HSIC 解耦损失
- `no_nce`：去掉 SurvNCE 对比损失
- `path_only` / `omic_only`：单模态基线
- `full`：完整 Tri-Surv

---

### 3.4 `main_trisurv.py` — 训练主循环重构

**新增功能：**

1. **`EarlyStopping` 类**：监控验证集 C-index，patience 轮未提升自动停止

2. **梯度累积**：每个 batch 的 loss 除以 `accumulation_steps` 再 backward，每 `accumulation_steps` 步执行一次优化器 step

3. **TensorBoard 记录**：每 `log_interval` 个有效优化步骤记录一次：
   - `loss_total`、`loss_cox`、`loss_hsic`、`loss_nce`、`loss_kl`
   - 学习率 `lr`

4. **缺失模态鲁棒性验证**：每个 epoch 验证 4 种组合：
   - `full`：全模态
   - `missing_path`：缺失病理图
   - `missing_geno`：缺失基因组
   - `missing_clin`：缺失临床数据

5. **Checkpoint 管理**：
   - `model_best.pth`：验证集最佳模型
   - `model_last.pth`：最新 checkpoint（含 optimizer/scheduler/epoch 状态）

6. **Cosine LR Schedule**：`torch.optim.lr_scheduler.CosineAnnealingLR`

7. **`_build_model()` 函数**：支持 `--ablation` 参数动态构建模型变体

---

## 四、使用方法

### 完整训练（5-fold CV）

```bash
CUDA_VISIBLE_DEVICES=0 python main_trisurv.py \
    --dataset tcga_blca \
    --modal trimodal \
    --num_epoch 30 \
    --batch_size 1 \
    --accumulation_steps 8 \
    --lr 2e-4 \
    --loss_kl 0.01 \
    --loss_hsic 0.05 \
    --loss_nce 0.1 \
    --patience 10 \
    --log_data
```

### 消融实验

```bash
# 去掉 VIB
python main_trisurv.py --dataset tcga_blca --ablation no_vib

# 去掉 cVAE（缺失填零）
python main_trisurv.py --dataset tcga_blca --ablation no_cvae

# 去掉 Cross-Attention
python main_trisurv.py --dataset tcga_blca --ablation no_cross_attn

# 双模态基线（无临床）
python main_trisurv.py --dataset tcga_blca --ablation no_clinical
```

### 断点恢复

```bash
python main_trisurv.py \
    --dataset tcga_blca \
    --resume results_trisurv/tcga_blca_2026-04-07-10-30/fold_0/model_last.pth
```

---

## 五、待改进方向（论文投稿前）

以下为论文投稿前需要补充的内容，不影响代码功能运行：

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 高 | SHAP / Attention 可视化 | 生成临床可解释性分析图 |
| 高 | 更多 SOTA 对比 | 确保在同等数据集上超过 DeReF、PORPOISE 等 |
| 高 | 第四个数据集 | TCGA-BLCA、LUAD、UCEC 之外再增加一个癌种 |
| 中 | 损失权重自动化搜索 | 用 Optuna/Bayesian Optimization 调参 |
| 中 | 多任务学习分支 | 同步预测分期、亚型等辅助任务 |

---

## 六、文件修改清单

| 文件 | 操作 | 状态 |
|------|------|------|
| `models/tri_surv.py` | 完全重写 | ✅ |
| `models/sit_loss.py` | 修复数值稳定性 | ✅ |
| `main_trisurv.py` | 完全重写 | ✅ |
| `utils/options.py` | 扩展参数 | ✅ |
| `datasets/dataset_survival.py` | 无需修改（trimodal 分支已存在） | ✅ |
| `utils/util.py` | 无需修改 | ✅ |
| `IMPROVEMENT_PLAN.md` | 本文档 | ✅ |

