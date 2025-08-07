# 多模态对抗检测一致性实验框架

**作者**: 张昕 (zhang.xin@duke.edu)  
**机构**: Duke University  
**版本**: 1.0.0

## 概述

本框架提供了一个完整的多模态对抗检测实验平台，支持多种攻击方法和防御策略的评估。框架采用模块化设计，易于扩展和定制。

## 主要特性

- **多种攻击方法**: PGD, Hubness, FSTA, SMA
- **多模态防御**: 文本变体生成、检索参考、生成参考、一致性检测
- **多数据集支持**: COCO, Flickr30K, CC3M, Visual Genome
- **消融实验**: 自动化组件贡献分析
- **可视化分析**: 丰富的图表和统计分析
- **配置驱动**: 灵活的YAML配置系统

## 实验目标

通过全面的实验验证文本变体一致性检测和生成参考图像防御机制在面对跨模态检索对抗攻击时的有效性和鲁棒性。

### 具体目标
- **有效性验证**：显著降低对抗攻击成功率 (ASR↓)
- **泛化性验证**：在不同数据集（MS COCO、Flickr30K、Conceptual Captions、Visual Genome）上的表现
- **基线对比**：对比现有常见防御方法，证明性能优势
- **消融分析**：验证各防御模块的具体贡献
- **效率分析**：确保防御方法满足实际部署需求（速度、GPU内存占用）

## 实验内容设计

### 基本实验设计
- **数据集**：MS COCO、Flickr30K、Conceptual Captions、Visual Genome
- **攻击方法**：PGD、Hubness、FSTA、SMA
- **评测指标**：攻击成功率 (ASR↓)、Top-k检索精度、检测准确率、误报率

### 防御效果实验
对比未使用防御方法（Baseline）和使用防御方法时，攻击成功率的下降情况。

### 基线模型 (Baselines)
1. **无防御 (No Defense)**：原始CLIP模型，无任何防御机制
2. **单模态异常检测 (Unimodal Anomaly Detection)**：使用图像或文本的单模态异常检测方法
3. **随机文本变体防御 (Random Text Variants)**：随机生成文本变体
4. **检索参考防御 (Retrieval Reference Only)**：仅使用检索到的真实图像作为参考
5. **生成参考防御 (Generative Reference Only)**：仅使用生成的Stable Diffusion图像作为参考

### 消融实验设计
通过启用或禁用不同模块，分析各个模块对防御成功率和检测性能的贡献。

## 目录结构

```
experiments/
├── configs/                 # 配置文件
│   ├── base_experiment.yaml # 基础实验配置
│   └── demo_experiment.yaml # 演示配置
├── defenses/               # 防御模块
│   ├── detector.py         # 主检测器
│   ├── consistency_checker.py
│   ├── text_variants.py
│   ├── generative_ref.py
│   └── retrieval_ref.py
├── datasets/               # 数据集加载器
│   ├── coco_loader.py
│   ├── flickr_loader.py
│   ├── cc_loader.py
│   └── vg_loader.py
├── runners/                # 实验运行器
│   ├── run_attack.py       # 攻击实验
│   ├── run_detection.py    # 检测实验
│   └── run_ablation.py     # 消融实验
├── utils/                  # 工具模块
│   ├── config_loader.py
│   ├── metrics.py
│   ├── visualization.py
│   └── logger.py
└── run_experiments.py      # 主入口脚本
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision transformers diffusers
pip install clip-by-openai faiss-cpu pillow
pip install matplotlib seaborn scikit-learn
pip install pyyaml tqdm numpy pandas

# 准备数据
mkdir -p data/coco data/flickr30k
# 下载并解压数据集到对应目录
```

### 2. 运行演示实验

```bash
# 完整实验（攻击+检测）
python run_experiments.py --config configs/demo_experiment.yaml --mode full

# 仅运行攻击生成
python run_experiments.py --config configs/demo_experiment.yaml --mode attack

# 仅运行检测评估
python run_experiments.py --config configs/demo_experiment.yaml --mode detection
```

### 3. 自定义实验

```bash
# 使用不同数据集
python run_experiments.py --config configs/base_experiment.yaml --dataset flickr30k

# 调整样本数量
python run_experiments.py --config configs/demo_experiment.yaml --max-samples 1000

# 启用调试模式
python run_experiments.py --config configs/demo_experiment.yaml --debug --max-samples 50
```

## 配置说明

### 基本配置结构

```yaml
# 实验基本信息
experiment_name: "my_experiment"
description: "实验描述"

# 数据集配置
dataset:
  name: "coco"              # 数据集名称
  max_samples: 1000         # 最大样本数
  batch_size: 32            # 批处理大小

# 攻击配置
attack:
  method: "hubness"         # 攻击方法
  hubness:
    epsilon: 0.0627         # 扰动强度
    num_iterations: 100     # 迭代次数

# 防御配置
defense:
  text_variants:
    enabled: true           # 启用文本变体
    num_variants: 3         # 变体数量
  
  retrieval_reference:
    enabled: true           # 启用检索参考
    num_references: 5       # 参考数量
  
  generative_reference:
    enabled: true           # 启用生成参考
    num_references: 3       # 生成数量

# 硬件配置
hardware:
  device: "auto"            # 自动选择设备
  gpu_ids: [0, 1]          # GPU列表
```

## 实验类型

### 1. 攻击实验

生成对抗样本，评估攻击成功率：

```bash
python run_experiments.py --config configs/attack_config.yaml --mode attack
```

输出：
- `adversarial_samples.pkl`: 对抗样本
- `attack_summary.json`: 攻击统计
- `attack_log.json`: 详细日志

### 2. 检测实验

评估防御系统性能：

```bash
python run_experiments.py --config configs/detection_config.yaml --mode detection
```

输出：
- `detection_results.json`: 检测结果
- `metrics_summary.json`: 性能指标
- `figures/`: 可视化图表

### 3. 消融实验

分析各组件贡献：

```bash
python runners/run_ablation.py --config configs/ablation_config.yaml --output-dir results/ablation
```

输出：
- `ablation_results.json`: 完整结果
- `ablation_summary.json`: 结果摘要
- `figures/ablation_*.png`: 可视化分析

## 扩展指南

### 添加新的攻击方法

1. 在 `src/attacks/` 中实现新攻击类
2. 继承 `BaseAttack` 并实现 `attack()` 方法
3. 在配置文件中添加相应参数

```python
class MyAttack(BaseAttack):
    def attack(self, image, text):
        # 实现攻击逻辑
        return {
            'success': True,
            'adversarial_image': adv_image,
            'perturbation': perturbation
        }
```

### 添加新的防御组件

1. 在 `experiments/defenses/` 中实现新组件
2. 实现标准接口方法
3. 在 `MultiModalDefenseDetector` 中集成

```python
class MyDefenseComponent:
    def __init__(self, config):
        self.config = config
    
    def process(self, image, text):
        # 实现防御逻辑
        return processed_data
```

### 添加新的数据集

1. 在 `experiments/datasets/` 中实现加载器
2. 继承 `BaseDatasetLoader`
3. 在 `__init__.py` 中注册

```python
class MyDatasetLoader(BaseDatasetLoader):
    def load_annotations(self):
        # 加载标注数据
        pass
    
    def get_image_path(self, image_id):
        # 获取图像路径
        pass
```

## 性能优化

### GPU优化

```yaml
hardware:
  mixed_precision: true     # 混合精度训练
  compile_model: true       # 模型编译
  gpu_ids: [0, 1, 2, 3]    # 多GPU并行
  
memory_optimization:
  gradient_checkpointing: true
  cpu_offload: true
```

### 批处理优化

```yaml
dataset:
  batch_size: 64            # 增大批处理
  num_workers: 8            # 多进程加载
  pin_memory: true          # 内存固定
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小 `batch_size`
   - 启用 `cpu_offload`
   - 使用 `gradient_checkpointing`

2. **模型加载失败**
   - 检查 `cache_dir` 权限
   - 验证模型名称正确性
   - 确保网络连接正常

3. **数据集路径错误**
   - 检查 `data_dir` 配置
   - 验证文件结构正确性
   - 确保数据完整下载

### 调试模式

```bash
# 启用详细日志
python run_experiments.py --config configs/demo.yaml --debug --log-level DEBUG

# 快速测试
python run_experiments.py --config configs/demo.yaml --max-samples 10 --dry-run
```

## 结果分析

### 指标解释

- **Accuracy**: 整体准确率
- **Precision**: 检测精确率
- **Recall**: 检测召回率
- **F1-Score**: F1分数
- **ROC-AUC**: ROC曲线下面积
- **PR-AUC**: PR曲线下面积

### 可视化图表

- `roc_curve.png`: ROC曲线
- `pr_curve.png`: PR曲线
- `confusion_matrix.png`: 混淆矩阵
- `score_distribution.png`: 分数分布
- `ablation_results.png`: 消融实验结果

## 引用

如果您使用了本框架，请引用：

```bibtex
@misc{zhang2024multimodal,
  title={Multi-Modal Adversarial Detection Framework},
  author={Zhang, Xin},
  institution={Duke University},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

- **作者**: 张昕
- **邮箱**: zhang.xin@duke.edu
- **机构**: Duke University

## 硬件要求

- **GPU**: 6×RTX4090 或同等性能
- **内存**: 至少32GB
- **存储**: 至少100GB可用空间（用于数据集和结果存储）

## 更新日志

### v1.0.0 (2024-12-19)
- 初始版本发布
- 支持多种攻击和防御方法
- 完整的实验框架
- 消融实验功能
- 可视化分析工具