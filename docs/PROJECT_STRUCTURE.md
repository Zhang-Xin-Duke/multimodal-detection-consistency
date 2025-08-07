# 项目结构说明

> **多模态检索对抗防御系统 - 完整项目结构文档**

## 📁 项目根目录

```
多模态检测一致性实验代码/
├── 📄 README.md                # 项目主要说明文档
├── 📄 QUICK_START.md           # 快速启动指南
├── 📄 LICENSE                  # 开源许可证
├── 📄 CITATION.cff             # 学术引用信息
├── 📄 .gitignore               # Git忽略文件配置
├── 📄 requirements.txt          # Python依赖包列表
├── 📄 setup.py                 # 项目安装配置
└── 📁 [核心目录]/              # 详见下方说明
```

## 🔧 配置管理 (`configs/`)

统一的配置文件管理，支持模块化和继承机制：

```
configs/
├── 📄 default.yaml             # 默认基础配置
├── 📄 reproducibility.yaml     # 可复现性配置 (种子、版本等)
├── 📄 efficiency_analysis.yaml # 效率分析专用配置
├── 📄 ablation_no_text_variants.yaml # 消融实验配置
├── 📁 attacks/                 # 攻击方法配置
│   ├── 📄 pgd.yaml            # PGD攻击参数
│   ├── 📄 hubness.yaml        # Hubness攻击参数
│   ├── 📄 fsta.yaml           # FSTA攻击参数
│   └── 📄 sma.yaml            # SMA攻击参数
├── 📁 baselines/               # 基线方法配置
│   ├── 📄 no_defense.yaml     # 无防御基线
│   ├── 📄 unimodal_anomaly.yaml # 单模态异常检测
│   ├── 📄 random_variants.yaml # 随机文本变体防御
│   ├── 📄 retrieval_only.yaml # 仅检索参考防御
│   └── 📄 generative_only.yaml # 仅生成参考防御
├── 📁 datasets/                # 数据集配置
│   ├── 📄 coco.yaml           # MS COCO数据集
│   ├── 📄 flickr30k.yaml      # Flickr30K数据集
│   ├── 📄 cc3m.yaml           # Conceptual Captions
│   └── 📄 visual_genome.yaml  # Visual Genome数据集
├── 📁 defenses/               # 防御方法配置
│   ├── 📄 base.yaml           # 基础防御配置
│   ├── 📄 tvc.yaml            # 文本变体一致性配置
│   └── 📄 genref.yaml         # 生成参考配置
├── 📁 experiments/            # 具体实验配置
│   ├── 📄 coco_pgd_full.yaml  # COCO+PGD完整实验
│   ├── 📄 flickr_hubness_full.yaml # Flickr+Hubness实验
│   ├── 📄 efficiency_profile.yaml # 效率分析实验
│   └── ... (其他数据集×攻击组合)
└── 📁 dynamic/                # 动态配置
    └── 📄 unified_config.yaml # 统一动态配置模板
```

## 💻 核心源代码 (`src/`)

项目的核心实现代码，模块化设计：

```
src/
├── 📄 __init__.py              # 包初始化
├── 📄 pipeline.py              # 主防御流水线
├── 📄 text_augment.py          # 文本增强模块
├── 📄 retrieval.py             # 多模态检索模块
├── 📄 sd_ref.py                # Stable Diffusion参考生成
├── 📄 detector.py              # 对抗检测器
├── 📄 ref_bank.py              # 参考向量库管理
├── 📄 config.py                # 配置管理
├── 📁 attacks/                 # 攻击方法实现
│   ├── 📄 __init__.py
│   ├── 📄 pgd_attack.py        # PGD攻击实现
│   ├── 📄 hubness_attack.py    # Hubness攻击实现
│   ├── 📄 fgsm_attack.py       # FGSM攻击实现
│   ├── 📄 cw_attack.py         # C&W攻击实现
│   └── 📄 text_attack.py       # 文本攻击实现
├── 📁 evaluation/              # 评估模块
│   ├── 📄 __init__.py
│   ├── 📄 experiment_evaluator.py # 实验评估器
│   └── 📄 data_validator.py    # 数据验证器
└── 📁 utils/                   # 工具函数库
    ├── 📄 __init__.py
    ├── 📄 config_manager.py     # 配置管理器
    ├── 📄 hardware_detector.py  # 硬件检测
    ├── 📄 cuda_utils.py         # CUDA工具
    ├── 📄 multi_gpu_processor.py # 多GPU处理
    ├── 📄 dynamic_config.py     # 动态配置
    ├── 📄 data_loader.py        # 数据加载器
    ├── 📄 metrics.py            # 评估指标
    ├── 📄 seed.py               # 随机种子管理
    └── 📄 visualization.py      # 结果可视化
```

## 🧪 实验框架 (`experiments/`)

独立的实验运行框架，支持各种实验类型：

```
experiments/
├── 📄 README.md                # 实验框架说明
├── 📄 run_experiments.py       # 实验运行主脚本
├── 📁 configs/                 # 实验专用配置
│   ├── 📄 base_experiment.yaml # 基础实验配置
│   └── 📄 demo_experiment.yaml # 演示实验配置
├── 📁 datasets/                # 数据集加载器
│   ├── 📄 __init__.py
│   ├── 📄 base_loader.py       # 基础加载器
│   ├── 📄 coco_loader.py       # COCO数据集加载器
│   ├── 📄 flickr_loader.py     # Flickr数据集加载器
│   ├── 📄 cc_loader.py         # CC数据集加载器
│   └── 📄 vg_loader.py         # Visual Genome加载器
├── 📁 defenses/                # 防御方法实现
│   ├── 📄 __init__.py
│   ├── 📄 detector.py          # 检测器实现
│   ├── 📄 text_variants.py     # 文本变体生成
│   ├── 📄 retrieval_ref.py     # 检索参考实现
│   ├── 📄 generative_ref.py    # 生成参考实现
│   └── 📄 consistency_checker.py # 一致性检查器
├── 📁 runners/                 # 实验运行器
│   ├── 📄 __init__.py
│   ├── 📄 run_attack.py        # 攻击实验运行器
│   ├── 📄 run_detection.py     # 检测实验运行器
│   └── 📄 run_ablation.py      # 消融实验运行器
└── 📁 utils/                   # 实验专用工具
    ├── 📄 __init__.py
    ├── 📄 config_loader.py      # 配置加载器
    ├── 📄 logger.py             # 日志管理
    ├── 📄 metrics.py            # 指标计算
    ├── 📄 seed.py               # 种子管理
    └── 📄 visualization.py      # 可视化工具
```

## 📊 结果分析 (`analysis/`)

论文级别的结果分析和可视化：

```
analysis/
├── 📄 __init__.py              # 分析包初始化
├── 📄 run_analysis.py          # 统一分析运行器
├── 📄 generate_comprehensive_report.py # 综合报告生成
├── 📄 generate_charts.py       # 图表生成器
└── 📄 generate_latex_tables.py # LaTeX表格生成器
```

**功能特性**:
- 自动生成HTML/PDF综合报告
- 生成论文级别的图表 (PNG/SVG)
- 生成LaTeX格式的实验结果表格
- 支持多种可视化样式和主题
- 统计分析和趋势分析

## 🛠️ 脚本工具 (`scripts/`)

各种实用脚本和工具：

```
scripts/
├── 📄 deploy.py                # 统一部署工具
├── 📄 run_complete_experiments.py # 完整实验运行脚本
├── 📄 run_analysis.sh          # Shell分析脚本 (推荐)
├── 📄 project_summary.py       # 项目总结生成
├── 📄 validate_experiment_configs.py # 配置验证工具
├── 📄 build_faiss_indices.py   # FAISS索引构建
├── 📄 download_real_datasets.py # 数据集下载工具
└── 📄 validate_datasets.py     # 数据集验证工具
```

## 🧪 测试套件 (`tests/`)

完整的测试覆盖，确保代码质量：

```
tests/
├── 📄 __init__.py              # 测试包初始化
├── 📄 test_analysis.py         # 分析模块测试
├── 📄 benchmark_analysis.py    # 基准测试脚本
├── 📄 test_basic_functionality.py # 基础功能测试
├── 📄 test_config.py           # 配置管理测试
├── 📄 test_detector.py         # 检测器测试
├── 📄 test_pipeline.py         # 流水线测试
├── 📄 test_retrieval.py        # 检索模块测试
├── 📄 test_sd_ref.py           # SD参考生成测试
├── 📄 test_text_augment.py     # 文本增强测试
└── 📁 configs/                 # 测试配置文件
```

**测试覆盖率目标**: ≥80%

## 📚 文档和示例

### 文档 (`docs/`)
```
docs/
├── 📄 COMPLETE_EXPERIMENTS_GUIDE.md # 完整实验指南
└── 📄 PROJECT_STRUCTURE.md     # 项目结构说明 (本文档)
```

### 示例 (`examples/`)
```
examples/
└── 📄 analysis_demo.py         # 分析功能演示
```

### 笔记本 (`notebooks/`)
```
notebooks/
└── 📄 demo.ipynb              # Jupyter演示笔记本
```

## 📖 参考资料 (`references/`)

相关论文和攻击方法的参考实现：

```
references/
├── 📄 README.md                # 参考资料说明
├── 📁 FSTA_Feature_Space_Targeted_Attack/ # FSTA攻击参考
│   ├── 📄 README.md
│   ├── 📄 feature_space_attack_paper.pdf
│   └── 📁 code/
└── 📁 SMA_Semantic_Misalignment_Attack/ # SMA攻击参考
    ├── 📄 README.md
    ├── 📄 adversarial_illusions_paper.pdf
    └── 📁 code/
```

## 📈 结果存储 (`results/`)

实验结果的统一存储目录：

```
results/                        # 自动创建
├── 📁 experiment_name/         # 具体实验结果
│   ├── 📁 logs/               # 实验日志
│   ├── 📁 metrics/            # 评估指标
│   ├── 📁 checkpoints/        # 模型检查点
│   └── 📁 visualizations/     # 可视化结果
└── 📁 analysis_output/         # 分析结果输出
    ├── 📁 reports/            # HTML/PDF报告
    ├── 📁 charts/             # 图表文件
    └── 📁 tables/             # LaTeX表格
```

## 🔄 工作流程

### 1. 实验执行流程
```
配置验证 → 环境检查 → 数据准备 → 实验执行 → 结果保存
```

### 2. 分析流程
```
结果加载 → 数据处理 → 统计分析 → 可视化 → 报告生成
```

### 3. 开发流程
```
需求分析 → 配置设计 → 代码实现 → 测试验证 → 文档更新
```

## 📋 配置管理策略

### 配置继承关系
```
default.yaml (基础配置)
├── dataset-specific.yaml (数据集特定配置)
├── attack-specific.yaml (攻击特定配置)
└── experiment-specific.yaml (实验特定配置)
```

### 配置验证
- 格式验证 (YAML语法)
- 内容验证 (参数范围、依赖关系)
- 兼容性验证 (硬件要求、软件版本)

## 🎯 质量保证

### 代码质量
- **模块化设计**: 清晰的接口和职责分离
- **错误处理**: 完善的异常处理机制
- **日志记录**: 详细的操作日志
- **文档完整**: 每个模块都有详细说明

### 实验质量
- **可复现性**: 固定随机种子和版本
- **可验证性**: 多次运行结果一致
- **可比较性**: 标准化的评估指标
- **可扩展性**: 易于添加新的攻击和防御方法

## 📞 维护和支持

**项目维护者**: 张昕 (ZHANG XIN)  
**联系方式**: zhang.xin@duke.edu  
**机构**: Duke University  

---

*本文档随项目更新而更新，最后更新时间: 2025-01-05*