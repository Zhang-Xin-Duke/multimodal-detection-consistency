# 完整实验配置指南

**作者**: 张昕 (zhang.xin@duke.edu)  
**机构**: Duke University  
**日期**: 2025-01-05

## 概述

本文档描述了多模态检测一致性实验的完整配置矩阵，涵盖4个数据集和4种攻击方法的所有组合，总共16个实验配置。

## 实验矩阵

### 数据集

| 数据集 | 配置文件前缀 | 描述 | 样本数量 |
|--------|-------------|------|----------|
| MS COCO | `coco_` | 图像-文本匹配数据集 | 1000 |
| Flickr30K | `flickr_` | 图像描述数据集 | 1000 |
| CC3M | `cc3m_` | 概念标题数据集 | 2000 |
| Visual Genome | `vg_` | 场景图数据集 | 1500 |

### 攻击方法

| 攻击方法 | 配置文件后缀 | 描述 | 主要参数 |
|----------|-------------|------|----------|
| PGD | `_pgd_full.yaml` | 投影梯度下降攻击 | ε=8/255, steps=500 |
| Hubness | `_hubness_full.yaml` | 中心性攻击 | hubs=20-30, iter=100 |
| FSTA | `_fsta_full.yaml` | 特征空间目标攻击 | iter=20, feature_layer=penultimate |
| SMA | `_sma_full.yaml` | 语义错位攻击 | iter=15, semantic_weight=2.0 |

## 完整实验配置列表

### MS COCO 数据集

1. **coco_pgd_full.yaml** - COCO + PGD攻击 + 完整防御
2. **coco_hubness_full.yaml** - COCO + Hubness攻击 + 完整防御
3. **coco_fsta_full.yaml** - COCO + FSTA攻击 + 完整防御
4. **coco_sma_full.yaml** - COCO + SMA攻击 + 完整防御

### Flickr30K 数据集

5. **flickr_pgd_full.yaml** - Flickr30K + PGD攻击 + 完整防御
6. **flickr_hubness_full.yaml** - Flickr30K + Hubness攻击 + 完整防御
7. **flickr_fsta_full.yaml** - Flickr30K + FSTA攻击 + 完整防御
8. **flickr_sma_full.yaml** - Flickr30K + SMA攻击 + 完整防御

### CC3M 数据集

9. **cc3m_pgd_full.yaml** - CC3M + PGD攻击 + 完整防御
10. **cc3m_hubness_full.yaml** - CC3M + Hubness攻击 + 完整防御
11. **cc3m_fsta_full.yaml** - CC3M + FSTA攻击 + 完整防御
12. **cc3m_sma_full.yaml** - CC3M + SMA攻击 + 完整防御

### Visual Genome 数据集

13. **vg_pgd_full.yaml** - Visual Genome + PGD攻击 + 完整防御
14. **vg_hubness_full.yaml** - Visual Genome + Hubness攻击 + 完整防御
15. **vg_fsta_full.yaml** - Visual Genome + FSTA攻击 + 完整防御
16. **vg_sma_full.yaml** - Visual Genome + SMA攻击 + 完整防御

## 防御配置

所有实验配置都包含完整的防御机制：

### 文本变体一致性 (TVC)
- **启用**: 是
- **变体数量**: 5
- **策略**: 同义词替换、释义、重排序
- **相似度阈值**: 0.85

### 生成参考一致性 (GenRef)
- **启用**: 是
- **参考图像数量**: 3
- **质量阈值**: 0.7
- **多样性阈值**: 0.3

### 检测器权重
- **直接一致性**: 0.25
- **文本变体一致性**: 0.35
- **生成参考一致性**: 0.40

## 评估指标

每个实验配置都会计算以下指标：

1. **攻击成功率 (ASR)**
   - 无防御ASR (%)
   - 有防御ASR (%)

2. **防御效果**
   - 防御成功率 (%) ↑
   - 检测率 (%)
   - 假阳性率 (%)

3. **检索性能**
   - Top-k 检索精度 (%)

4. **计算开销**
   - 计算时间开销 (%)
   - GPU内存使用

## 使用方法

### 1. 运行单个实验

```bash
# 运行COCO + PGD实验
python run_experiments.py --config configs/experiments/coco_pgd_full.yaml

# 运行Flickr30K + Hubness实验
python run_experiments.py --config configs/experiments/flickr_hubness_full.yaml
```

### 2. 运行所有实验

```bash
# 运行完整实验矩阵
python run_complete_experiments.py
```

这将：
- 按顺序运行所有16个实验配置
- 自动收集和解析结果
- 生成完整的结果表格
- 保存为JSON、CSV和Markdown格式

### 3. 批量运行特定数据集

```bash
# 只运行COCO数据集的所有攻击
for attack in pgd hubness fsta sma; do
    python run_experiments.py --config configs/experiments/coco_${attack}_full.yaml
done
```

### 4. 批量运行特定攻击方法

```bash
# 只运行PGD攻击在所有数据集上
for dataset in coco flickr cc3m vg; do
    python run_experiments.py --config configs/experiments/${dataset}_pgd_full.yaml
done
```

## 结果输出

### 目录结构

```
results/
├── coco_pgd_full/          # COCO + PGD结果
├── coco_hubness_full/      # COCO + Hubness结果
├── flickr_pgd_full/        # Flickr30K + PGD结果
├── ...
├── complete_experiments_final_YYYYMMDD_HHMMSS.json
├── complete_experiments_table_YYYYMMDD_HHMMSS.csv
└── complete_experiments_table_YYYYMMDD_HHMMSS.md
```

### 结果表格格式

| 数据集 | 攻击方法 | 无防御ASR (%) | 有防御ASR (%) | 防御成功率(%) ↑ | Top-k 检索精度(%) |
|--------|----------|---------------|---------------|----------------|-------------------|
| MS COCO | PGD | 98.5 | 23.1 | 76.5 | 87.2 |
| MS COCO | Hubness | 100.0 | 27.0 | 73.0 | 88.7 |
| ... | ... | ... | ... | ... | ... |

## 实验时间估算

### 单个实验
- **小数据集** (COCO, Flickr30K): 30-60分钟
- **大数据集** (CC3M, Visual Genome): 60-120分钟

### 完整实验矩阵
- **总时间**: 约16-32小时
- **建议**: 使用多GPU并行或分批运行

## 硬件要求

### 最低配置
- **GPU**: 8GB显存 (GTX 1080, RTX 2070等)
- **内存**: 16GB RAM
- **存储**: 100GB可用空间

### 推荐配置
- **GPU**: 24GB显存 (RTX 3090, RTX 4090等)
- **内存**: 32GB RAM
- **存储**: 500GB可用空间

## 故障排除

### 常见问题

1. **内存不足**
   - 减少batch_size
   - 减少样本数量
   - 使用梯度累积

2. **实验超时**
   - 增加timeout设置
   - 减少迭代次数
   - 使用早停机制

3. **配置文件错误**
   - 检查YAML语法
   - 验证文件路径
   - 确认参数范围

### 日志文件

- **主日志**: `complete_experiments.log`
- **单个实验日志**: `results/{experiment_name}/experiment.log`
- **错误日志**: `results/{experiment_name}/error.log`

## 扩展实验

### 添加新数据集

1. 在 `configs/datasets/` 中创建数据集配置
2. 为每种攻击方法创建实验配置
3. 更新 `run_complete_experiments.py` 中的数据集列表

### 添加新攻击方法

1. 在 `configs/attacks/` 中创建攻击配置
2. 为每个数据集创建实验配置
3. 更新 `run_complete_experiments.py` 中的攻击方法列表

### 自定义防御配置

1. 在 `configs/defenses/` 中创建新的防御配置
2. 在实验配置中引用新的防御配置
3. 调整检测器权重和阈值

## 结论

通过这16个完整的实验配置，我们可以全面评估多模态防御系统在不同数据集和攻击方法下的性能，为研究提供完整的实验基础。

实验配置的模块化设计使得添加新的数据集、攻击方法或防御策略变得简单，支持灵活的实验扩展和定制。