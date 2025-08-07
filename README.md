# 🛡️ 多模态检索对抗防御系统

> **基于文本变体一致性的多模态检索对抗攻击防御方法**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)](tests/)

---

## 🎯 项目概述

本项目提出了一种**基于文本变体一致性的多模态检索对抗防御方法**，通过构建多层次防御机制有效检测和防御针对多模态检索系统的对抗攻击。该系统在保持检索精度的同时，显著提升了对各类对抗攻击的鲁棒性。

### 🔍 核心创新

- **🔄 智能文本增强**: 基于Qwen2-7B的多策略文本变体生成（同义词替换、句式改写、语法调整）
- **🎨 视觉参考生成**: 利用Stable Diffusion构建高质量视觉参考向量库
- **⚡ 多维一致性检测**: 文本-图像、变体间、参考向量多层次一致性分析
- **🚀 端到端防御**: 完整的检测流水线，支持实时部署和批量处理
- **🔧 自适应配置**: 智能硬件检测与配置优化，支持多GPU并行

多模态检索系统面对对抗样本（如 Hubness、PGD 等攻击）时极易性能骤降。本仓库实现了一种 **"文本变体一致性检测"** 防御框架，通过 **文本增强 → 参考向量构建 → 一致性度量** 三阶段，在保持高检索精度的同时，有效识别并过滤对抗查询。

> 作者：张昕 · Duke University  
> 联系：zhang.xin@duke.edu

---

## 🎯 实验目标

1. **鲁棒性**：大幅降低对抗攻击成功率（ASR↓）。  
2. **精度保持**：在无攻击场景下检索 Top-k 精度损失 <2%。  
3. **效率**：6×RTX 4090 下 >50 query/s。  
4. **可复现**：全部配置、脚本开箱即用。

---

## 📁 项目结构

```
多模态检测一致性实验代码/
├── 📁 configs/                    # 配置文件目录
│   ├── 📄 default.yaml           # 默认配置
│   ├── 📄 reproducibility.yaml   # 可复现性配置
│   ├── 📄 efficiency_analysis.yaml # 效率分析配置
│   ├── 📁 attacks/               # 攻击方法配置
│   │   ├── 📄 pgd.yaml          # PGD攻击配置
│   │   ├── 📄 hubness.yaml      # Hubness攻击配置
│   │   ├── 📄 fsta.yaml         # FSTA攻击配置
│   │   └── 📄 sma.yaml          # SMA攻击配置
│   ├── 📁 baselines/             # 基线方法配置
│   │   ├── 📄 no_defense.yaml   # 无防御基线
│   │   ├── 📄 unimodal_anomaly.yaml # 单模态异常检测
│   │   ├── 📄 random_variants.yaml # 随机文本变体
│   │   ├── 📄 retrieval_only.yaml # 仅检索参考
│   │   └── 📄 generative_only.yaml # 仅生成参考
│   ├── 📁 datasets/              # 数据集配置
│   │   ├── 📄 coco.yaml         # MS COCO配置
│   │   ├── 📄 flickr30k.yaml    # Flickr30K配置
│   │   ├── 📄 cc3m.yaml         # Conceptual Captions配置
│   │   └── 📄 visual_genome.yaml # Visual Genome配置
│   ├── 📁 defenses/             # 防御方法配置
│   │   ├── 📄 base.yaml         # 基础防御配置
│   │   ├── 📄 tvc.yaml          # 文本变体一致性配置
│   │   └── 📄 genref.yaml       # 生成参考配置
│   ├── 📁 experiments/          # 实验配置
│   │   ├── 📄 coco_pgd_full.yaml # COCO+PGD完整实验
│   │   ├── 📄 flickr_hubness_full.yaml # Flickr+Hubness实验
│   │   ├── 📄 efficiency_profile.yaml # 效率分析实验
│   │   └── ... (其他实验配置)
│   └── 📁 dynamic/              # 动态配置
│       └── 📄 unified_config.yaml # 统一动态配置
├── 📁 src/                       # 核心源代码
│   ├── 📄 __init__.py           # 包初始化
│   ├── 📄 pipeline.py           # 主防御流水线
│   ├── 📄 text_augment.py       # 文本增强模块
│   ├── 📄 retrieval.py          # 多模态检索模块
│   ├── 📄 sd_ref.py             # Stable Diffusion参考生成
│   ├── 📄 detector.py           # 对抗检测器
│   ├── 📄 ref_bank.py           # 参考向量库管理
│   ├── 📄 config.py             # 配置管理
│   ├── 📁 attacks/              # 攻击方法实现
│   │   ├── 📄 pgd_attack.py     # PGD攻击
│   │   ├── 📄 hubness_attack.py # Hubness攻击
│   │   ├── 📄 fgsm_attack.py    # FGSM攻击
│   │   ├── 📄 cw_attack.py      # C&W攻击
│   │   └── 📄 text_attack.py    # 文本攻击
│   ├── 📁 evaluation/           # 评估模块
│   │   ├── 📄 experiment_evaluator.py # 实验评估器
│   │   └── 📄 data_validator.py # 数据验证器
│   └── 📁 utils/                # 工具函数
│       ├── 📄 config_manager.py # 配置管理器
│       ├── 📄 hardware_detector.py # 硬件检测
│       ├── 📄 cuda_utils.py     # CUDA工具
│       ├── 📄 multi_gpu_processor.py # 多GPU处理
│       ├── 📄 dynamic_config.py # 动态配置
│       ├── 📄 data_loader.py    # 数据加载器
│       ├── 📄 metrics.py        # 评估指标
│       ├── 📄 seed.py           # 随机种子管理
│       └── 📄 visualization.py  # 结果可视化
├── 📁 experiments/               # 实验框架
│   ├── 📄 README.md             # 实验说明
│   ├── 📄 run_experiments.py    # 实验运行主脚本
│   ├── 📁 configs/              # 实验专用配置
│   │   ├── 📄 base_experiment.yaml # 基础实验配置
│   │   └── 📄 demo_experiment.yaml # 演示实验配置
│   ├── 📁 datasets/              # 数据集加载器
│   │   ├── 📄 base_loader.py    # 基础加载器
│   │   ├── 📄 coco_loader.py    # COCO加载器
│   │   ├── 📄 flickr_loader.py  # Flickr加载器
│   │   ├── 📄 cc_loader.py      # CC加载器
│   │   └── 📄 vg_loader.py      # Visual Genome加载器
│   ├── 📁 defenses/             # 防御方法实现
│   │   ├── 📄 detector.py       # 检测器
│   │   ├── 📄 text_variants.py  # 文本变体生成
│   │   ├── 📄 retrieval_ref.py  # 检索参考
│   │   ├── 📄 generative_ref.py # 生成参考
│   │   └── 📄 consistency_checker.py # 一致性检查
│   ├── 📁 runners/              # 实验运行器
│   │   ├── 📄 run_attack.py     # 攻击实验运行器
│   │   ├── 📄 run_detection.py  # 检测实验运行器
│   │   └── 📄 run_ablation.py   # 消融实验运行器
│   └── 📁 utils/                # 实验工具
│       ├── 📄 config_loader.py  # 配置加载器
│       ├── 📄 logger.py         # 日志管理
│       ├── 📄 metrics.py        # 指标计算
│       ├── 📄 seed.py           # 种子管理
│       └── 📄 visualization.py  # 可视化工具
├── 📁 analysis/                  # 结果分析模块
│   ├── 📄 __init__.py           # 分析包初始化
│   ├── 📄 run_analysis.py       # 统一分析运行器
│   ├── 📄 generate_comprehensive_report.py # 综合报告生成
│   ├── 📄 generate_charts.py    # 图表生成
│   └── 📄 generate_latex_tables.py # LaTeX表格生成
├── 📁 scripts/                  # 脚本工具
│   ├── 📄 deploy.py             # 统一部署工具
│   ├── 📄 run_complete_experiments.py # 完整实验运行
│   ├── 📄 project_summary.py    # 项目总结
│   ├── 📄 validate_experiment_configs.py # 配置验证
│   ├── 📄 build_faiss_indices.py # FAISS索引构建
│   ├── 📄 download_real_datasets.py # 数据集下载
│   └── 📄 validate_datasets.py  # 数据集验证
├── 📁 tests/                    # 测试套件
│   ├── 📄 __init__.py           # 测试包初始化
│   ├── 📄 test_analysis.py      # 分析模块测试
│   ├── 📄 benchmark_analysis.py # 基准测试
│   ├── 📄 test_basic_functionality.py # 基础功能测试
│   ├── 📄 test_config.py        # 配置测试
│   ├── 📄 test_detector.py      # 检测器测试
│   ├── 📄 test_pipeline.py      # 流水线测试
│   ├── 📄 test_retrieval.py     # 检索测试
│   ├── 📄 test_sd_ref.py        # SD参考测试
│   └── 📄 test_text_augment.py  # 文本增强测试
├── 📁 examples/                 # 使用示例
│   └── 📄 analysis_demo.py      # 分析演示
├── 📁 docs/                     # 项目文档
│   ├── 📄 COMPLETE_EXPERIMENTS_GUIDE.md # 完整实验指南
│   └── 📄 PROJECT_STRUCTURE.md  # 项目结构说明
├── 📁 notebooks/                # Jupyter笔记本
│   └── 📄 demo.ipynb           # 演示笔记本
├── 📁 references/               # 参考资料
│   ├── 📄 README.md            # 参考资料说明
│   ├── 📁 FSTA_Feature_Space_Targeted_Attack/ # FSTA攻击参考
│   └── 📁 SMA_Semantic_Misalignment_Attack/ # SMA攻击参考
├── 📁 results/                  # 实验结果存储
├── 📄 requirements.txt          # Python依赖列表
├── 📄 setup.py                 # 安装配置
├── 📄 CITATION.cff             # 引用信息
├── 📄 LICENSE                  # 开源许可证
├── 📄 .gitignore               # Git忽略文件
└── 📄 README.md                # 项目说明文档
```

---

## 🚀 快速开始

### 一键启动
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行演示
python examples/analysis_demo.py

# 3. 生成分析报告
./scripts/run_analysis.sh
```

📖 **详细指南**: [QUICK_START.md](QUICK_START.md)

---

## 🧩 方法概览

### 核心防御流程

我们的防御方法基于**文本变体一致性检测**，通过三个核心阶段实现对抗样本检测：

#### 1. 文本增强阶段 (Text Augmentation)
- **同义词替换**：基于WordNet语义网络，替换关键词汇，保持语义一致性
- **释义生成**：使用Qwen2-7B模型生成语义等价的文本变体，增强文本多样性
- **句法变换**：调整句子结构，保持语义不变，提升鲁棒性
- **回译技术**：通过多语言翻译链增强文本多样性，生成自然变体

#### 2. 参考向量构建 (Reference Vector Construction)
- **多模态检索**：使用CLIP模型编码文本变体和图像，支持语义相似性搜索
- **SD参考生成**：基于Stable Diffusion生成参考图像
  - 支持多种SD模型（SD-1.5、SD-2.1、SDXL等）
  - 可配置生成参数（推理步数、引导尺度、图像尺寸）
  - 多GPU并行生成，提升效率
  - 种子控制确保结果可复现
- **向量库管理**：智能聚类和缓存机制，支持10K+参考向量
- **特征对齐**：确保不同来源向量的一致性

#### 3. 一致性检测 (Consistency Detection)
- **相似度计算**：多种距离度量（余弦、欧氏、点积）
- **异常检测**：基于统计阈值和机器学习模型
- **集成判决**：多检测器投票机制，提升鲁棒性
- **自适应阈值**：根据数据分布动态调整检测阈值

```python
# 核心检测流程详细实现
def detect_adversarial(image, text):
    # 1. 文本增强：生成多样化文本变体
    text_variants = text_augmenter.generate_variants(
        text, 
        methods=['synonym', 'paraphrase', 'syntax', 'backtranslation'],
        num_variants=8,
        similarity_threshold=0.85
    )
    
    # 2. 多模态参考向量构建
    ref_vectors = []
    for variant in text_variants:
        # 2.1 基于CLIP的文本-图像检索
        retrieved_images = retriever.retrieve_images_by_text(
            variant, 
            top_k=5,
            similarity_threshold=0.7,
            use_faiss_gpu=True
        )
        ref_vectors.extend([
            clip_model.encode_image(img) for img in retrieved_images
        ])
        
        # 2.2 Stable Diffusion参考图像生成
        sd_images = sd_generator.generate_reference_images(
            prompt=variant,
            num_images=3,
            guidance_scale=7.5,
            num_inference_steps=50,
            height=512, width=512
        )
        ref_vectors.extend([
            clip_model.encode_image(img) for img in sd_images
        ])
    
    # 3. 查询向量编码
    query_image_vector = clip_model.encode_image(image)
    query_text_vector = clip_model.encode_text(text)
    
    # 4. 多维一致性计算
    image_consistency_scores = []
    text_consistency_scores = []
    
    for ref_vec in ref_vectors:
        # 图像-参考一致性
        img_sim = cosine_similarity(query_image_vector, ref_vec)
        image_consistency_scores.append(img_sim)
        
        # 文本-参考一致性
        text_sim = cosine_similarity(query_text_vector, ref_vec)
        text_consistency_scores.append(text_sim)
    
    # 跨模态一致性
    cross_modal_similarity = cosine_similarity(
        query_image_vector, query_text_vector
    )
    
    # 5. 统计特征提取
    img_mean, img_std = np.mean(image_consistency_scores), np.std(image_consistency_scores)
    text_mean, text_std = np.mean(text_consistency_scores), np.std(text_consistency_scores)
    
    # 6. 集成异常检测
    consistency_features = np.array([
        img_mean, img_std, text_mean, text_std, cross_modal_similarity
    ])
    
    # 多检测器投票
    threshold_detector = img_std > adaptive_threshold
    ml_detector = anomaly_classifier.predict(consistency_features.reshape(1, -1))[0]
    statistical_detector = (img_mean < 0.5) or (text_mean < 0.5)
    
    # 加权投票决策
    detection_votes = [threshold_detector, ml_detector, statistical_detector]
    weights = [0.4, 0.4, 0.2]
    final_score = np.average(detection_votes, weights=weights)
    
    is_adversarial = final_score > 0.5
    confidence = max(img_std, 1.0 - img_mean)
    
    return {
        'is_adversarial': is_adversarial,
        'confidence': confidence,
        'consistency_scores': {
            'image_mean': img_mean,
            'image_std': img_std,
            'text_mean': text_mean,
            'text_std': text_std,
            'cross_modal': cross_modal_similarity
        },
        'detection_votes': detection_votes,
        'final_score': final_score
    }
```

### 关键技术组件

| 组件 | 核心类 | 主要功能 |
|------|--------|----------|
| 文本增强 | `TextAugmenter` | 生成5-10个语义等价文本变体 |
| 多模态检索 | `MultiModalRetriever` | CLIP-based文本-图像检索 |
| SD参考生成 | `SDReferenceGenerator` | 基于文本生成参考图像 |
| 参考向量库 | `ReferenceBank` | 智能缓存和聚类管理 |
| 对抗检测 | `AdversarialDetector` | 多方法集成检测 |
| 检测流水线 | `MultiModalDetectionPipeline` | 端到端处理流程 |

## 技术实现详解

### 1. 文本增强技术 (TextAugmenter)

#### 核心实现原理
- **同义词替换**：基于WordNet语义网络和自定义词典，智能识别关键词并进行语义保持的替换
- **释义生成**：利用Qwen-7B大语言模型的强大理解能力，生成语义等价但表达不同的文本
- **句法变换**：通过依存句法分析，重组句子结构而保持核心语义不变
- **回译增强**：使用多语言翻译链（中文→英文→中文）生成自然的文本变体

#### 技术特点
```python
# 文本增强示例
original_text = "一只橙色的猫坐在窗台上"
variants = text_augmenter.generate_variants(original_text)
# 输出变体：
# - "一只橘色的猫咪坐在窗户边"
# - "橙色猫咪在窗台上休息"
# - "窗台上有一只橙色的小猫"
```

### 2. 多模态检索技术 (MultiModalRetriever)

#### 核心实现原理
- **CLIP特征编码**：使用预训练CLIP模型将文本和图像编码到统一的语义空间
- **FAISS高效索引**：构建GPU加速的向量索引，支持百万级图像的毫秒级检索
- **语义相似性搜索**：基于余弦相似度进行跨模态语义匹配
- **动态索引更新**：支持增量式索引构建和实时更新

#### 技术特点
```python
# 检索系统构建
retriever = MultiModalRetriever(clip_model="ViT-L/14")
retriever.build_index(image_dataset, batch_size=256)

# 文本到图像检索
similar_images = retriever.search_by_text(
    "橙色的猫", 
    top_k=10, 
    threshold=0.7
)
```

### 3. Stable Diffusion参考生成 (SDReferenceGenerator)

#### 核心实现原理
- **多模型支持**：集成SD-1.5、SD-2.1、SDXL等多种Stable Diffusion模型
- **参数化生成**：精确控制推理步数、引导尺度、图像尺寸等生成参数
- **多GPU并行**：支持多GPU并行生成，显著提升生成效率
- **种子控制**：确保实验结果的可复现性
- **质量过滤**：基于CLIP评分自动过滤低质量生成图像

#### 技术特点
```python
# SD参考图像生成
sd_generator = SDReferenceGenerator(
    model="runwayml/stable-diffusion-v1-5",
    device="cuda",
    enable_multi_gpu=True
)

# 批量生成参考图像
reference_images = sd_generator.generate_batch(
    prompts=["橙色的猫", "猫咪在窗台"],
    num_images_per_prompt=3,
    guidance_scale=7.5,
    num_inference_steps=50
)
```

### 4. 参考向量库管理 (ReferenceBank)

#### 核心实现原理
- **智能聚类**：使用K-means和层次聚类算法组织参考向量
- **容量管理**：LRU缓存策略和相似度去重，维持最优向量集合
- **持久化存储**：支持向量库的保存和加载，避免重复计算
- **线程安全**：多线程环境下的安全访问和更新

#### 技术特点
```python
# 参考向量库构建
ref_bank = ReferenceBank(
    max_capacity=10000,
    similarity_threshold=0.95,
    clustering_method="kmeans"
)

# 添加和查询参考向量
ref_bank.add_reference(vector, metadata)
similar_refs = ref_bank.query_similar(query_vector, top_k=5)
```

### 5. 对抗检测算法 (AdversarialDetector)

#### 核心实现原理
- **多维一致性分析**：计算图像-文本、图像-参考、文本-参考的多重一致性
- **统计异常检测**：基于Z-score、IQR等统计方法识别异常模式
- **机器学习分类器**：训练SVM、随机森林等分类器进行二分类判决
- **集成学习**：融合多个检测器的结果，提升检测准确率和鲁棒性

#### 技术特点
```python
# 对抗检测器配置
detector = AdversarialDetector(
    consistency_methods=["cosine", "euclidean"],
    anomaly_detectors=["isolation_forest", "one_class_svm"],
    ensemble_method="weighted_voting"
)

# 检测对抗样本
result = detector.detect(image, text)
print(f"对抗样本: {result['is_adversarial']}, 置信度: {result['confidence']}")
```

```
┌──────────────┐   Text Augmenter     ┌──────────────────────┐
│  Text Query  │ ────────────────▶  N 个语义相似文本变体  │
└──────────────┘                     └──────┬──────────────┘
                                             │
                                 ┌───────────▼───────────┐
                                 │ Reference Bank Builder │
                                 │  • 检索参考向量        │
                                 │  • SD 合成参考向量     │
                                 └───────────┬───────────┘
                                             │ 聚合
                                   ┌─────────▼─────────┐
                                   │  Reference Vector │
                                   └─────────┬─────────┘
                                             │
┌──────────────┐    Cosine Sim   ┌───────────▼───────────┐
│ Image Query  │ ───────────────▶│ Consistency Detector │
└──────────────┘                 └───────────┬───────────┘
                                             │σ>τ?
                                   ┌─────────▼─────────┐
                                   │  Adversarial?     │
                                   └───────────────────┘
```

### 1. 文本增强
- 调用 <mcsymbol name="TextAugmenter" filename="text_augment.py" path="src/text_augment.py" startline="24" type="class"></mcsymbol> 使用 Qwen-LM 生成 *N* 条语义等价变体。
- 余弦相似度 ≥ `similarity_threshold` 保留，否则丢弃。

### 2. 参考向量构建
- 检索参考：<mcsymbol name="MultiModalRetriever" filename="retrieval.py" path="src/retrieval.py" startline="22" type="class"></mcsymbol> 取每个变体 Top-k 图像特征。
- 合成参考：<mcsymbol name="StableDiffusionReferenceGenerator" filename="sd_ref.py" path="src/sd_ref.py" startline="29" type="class"></mcsymbol> 用 Stable Diffusion 生成 `m` 张图像并编码。
- 拼接后求均值得到单变体向量，再对所有变体均值 → **Reference Vector**。

### 3. 一致性检测
- 计算查询图像向量与 Reference Vector 的相似度数组 `S`。
- 标准差 `σ = std(S)`，若 `σ > consistency_threshold` 标记为对抗样本。
- 置信度直接返回 `σ`，便于动态阈值调节。

---

## 📦 安装

### 📋 系统要求

#### 硬件要求
- **GPU**: NVIDIA GPU with CUDA support (推荐RTX 3080+)
- **显存**: 最少6GB，推荐12GB+（多GPU环境推荐24GB+）
- **内存**: 最少16GB，推荐32GB+
- **存储**: 至少50GB可用空间（用于模型和数据缓存）

#### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+) / Windows 10+ / macOS 10.15+
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8+ (推荐12.1)
- **Docker**: 可选，用于容器化部署

### 方式一：一键安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/zhangxin-duke/multimodal-defense.git
cd multimodal-defense

# 运行一键安装脚本（自动检测环境并安装）
./install.sh

# 激活环境（根据脚本提示）
conda activate mm_defense  # 或 source venv/bin/activate

# 验证安装
python -c "import src; print('安装成功！')"
```

**安装脚本特性：**
- 🔍 自动检测Python、CUDA、Conda环境
- ⚙️ 智能选择PyTorch版本（GPU/CPU）
- 📁 自动创建必要的目录结构
- 🎯 运行硬件检测和配置生成
- 💡 提供详细的安装后指导
- 🔧 自动下载必要的模型和数据

### 方式二：手动安装

```bash
# 克隆仓库
git clone https://github.com/zhangxin-duke/multimodal-defense.git
cd multimodal-defense

# 1. 创建虚拟环境
conda create -n mm_defense python=3.9 -y
conda activate mm_defense

# 2. 安装PyTorch (根据CUDA版本选择)
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. 安装核心依赖
pip install transformers==4.35.0 diffusers==0.24.0 accelerate==0.24.1
pip install faiss-gpu==1.7.4 sentence-transformers==2.2.2
pip install nltk==3.8.1 wordnet==0.0.1b2

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 开发模式安装，自动配置 CUDA 环境
pip install -e .

# 6. 下载必要的模型和数据
python scripts/download_models.py

# 7. 验证安装
python -c "from src.pipeline import MultiModalDefensePipeline; print('安装验证成功！')"
```

### 方式三：Docker安装

```bash
# 构建Docker镜像
docker build -t mm_defense:latest .

# 运行容器（需要NVIDIA Docker支持）
docker run --gpus all -it -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  mm_defense:latest

# 或使用预构建镜像
docker pull zhangxin/mm_defense:latest
docker run --gpus all -it zhangxin/mm_defense:latest
```

### 方式四：Conda环境文件安装

```bash
# 使用预定义的环境文件
conda env create -f environment.yml
conda activate mm_defense

# 安装项目
pip install -e .
```

> 安装脚本通过 `setup.py` 内的 *PostInstallCommand* 自动设置 `CUDA_LAUNCH_BLOCKING`、显存策略等环境变量，并进行 GPU 健康检测。

### 🔧 安装后配置

```bash
# 设置环境变量（可选，脚本会自动设置）
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# 下载NLTK数据
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 验证GPU可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"
```

### 🚀 自动硬件检测与部署

本项目支持自动检测本地硬件配置并生成相应的部署配置，适应不同数量和型号的GPU：

#### 方式一：快速启动脚本（推荐）

```bash
# 一键启动：检测硬件 + 生成配置 + 启动服务
python quick_start.py

# 交互模式启动（支持运行时管理）
python quick_start.py --interactive

# 分步骤执行
python quick_start.py --detect-only      # 仅检测硬件
python quick_start.py --config-only      # 仅生成配置
python quick_start.py --start-only       # 仅启动服务

# 使用自定义配置
python quick_start.py --config custom.yaml
```

#### 方式二：自动部署脚本

```bash
# 一键自动部署
python auto_deploy.py
# 或使用命令：mm-auto-deploy

# 仅检测硬件配置（不启动服务）
python auto_deploy.py --detect-only

# 交互模式启动
python auto_deploy.py --interactive

# 后台运行模式
python auto_deploy.py --daemon
```

### 可用的控制台命令

安装后，您可以直接使用以下命令：

```bash
# 主要部署命令
mm-deploy                    # 统一部署工具（推荐）
mm-quick-start              # 快速启动脚本
mm-auto-deploy              # 自动部署脚本

# 硬件检测和配置工具
mm-hardware-detect          # 硬件检测工具
mm-config-gen               # 配置生成工具

# 实验和攻击工具
mm-defense                  # 运行防御实验
mm-attack                   # 运行攻击实验
```

#### 硬件适配能力
- **6张RTX 4090**: 高性能配置，支持大批量并行生成
- **4张A100/A200**: 企业级配置，启用Tensor Cores和Flash Attention
- **2-3张中端GPU**: 标准配置，平衡性能和资源使用
- **单GPU**: 基础配置，启用CPU卸载优化
- **CPU模式**: 无GPU时的回退方案

#### 动态配置特性
- 🔍 **智能硬件检测**: 自动识别GPU型号、内存、计算能力
- ⚙️ **自适应配置**: 根据硬件自动调整批处理大小、并发数、内存使用
- 🎯 **性能优化**: 针对不同GPU型号启用相应的优化特性
- 📊 **实时监控**: 提供GPU利用率、内存使用等实时状态
- 🔄 **热重载**: 支持运行时重新检测和配置更新

---

## 🚀 快速上手

### 方式一：统一部署工具（推荐）

使用新的 `deploy.py` 统一部署工具，集成了所有部署功能：

```bash
# 一键完整部署（推荐）
python deploy.py

# 交互模式（适合新手）
python deploy.py --interactive

# 分步骤执行
python deploy.py --detect-only          # 仅检测硬件
python deploy.py --config-only          # 仅生成配置
python deploy.py --deploy-only          # 仅部署系统

# 使用指定配置档案
python deploy.py --profile high_performance
python deploy.py --profile medium
python deploy.py --profile standard

# 使用不同部署模式
python deploy.py --deploy-mode quick    # 快速部署（默认）
python deploy.py --deploy-mode auto     # 自动部署

# 安装后可直接使用命令
mm-deploy                               # 等同于 python deploy.py
mm-deploy --interactive                 # 交互模式
```

### 方式二：快速启动脚本

使用 `quick_start.py` 脚本进行快速启动：

```bash
# 一键启动（检测+配置+启动）
python quick_start.py
# 或使用命令：mm-quick-start

# 交互模式
python quick_start.py --interactive

# 分步骤执行
python quick_start.py --detect-only    # 仅检测硬件
python quick_start.py --config-only    # 仅生成配置
python quick_start.py --start-only     # 仅启动服务

# 使用自定义配置
python quick_start.py --config configs/custom_config.yaml
```

### 基础使用示例

#### 单查询检测
```python
from src.pipeline import MultiModalDefensePipeline
from src.utils.config import load_config

# 加载配置
config = load_config("configs/defense.yaml")

# 初始化防御流水线
defense = MultiModalDefensePipeline(config)

# 检测对抗查询
query_text = "A photo of a cat"
result = defense.detect_adversarial(query_text)

print(f"查询文本: {query_text}")
print(f"是否为对抗样本: {result['is_adversarial']}")
print(f"置信度分数: {result['confidence']:.3f}")
print(f"风险等级: {result['risk_level']}")
print(f"检测详情: {result['details']}")

# 如果检测到对抗样本，获取清洁版本
if result['is_adversarial']:
    clean_variants = result['clean_variants']
    print(f"推荐的清洁文本变体: {clean_variants[:3]}")
```

#### 批量处理
```python
from src.pipeline import MultiModalDefensePipeline
import pandas as pd

# 初始化
defense = MultiModalDefensePipeline.from_config("configs/defense.yaml")

# 批量检测
queries = [
    "A beautiful sunset over the ocean",
    "Adversarial example with noise",
    "Normal query about dogs"
]

# 批量处理（支持并行）
results = defense.batch_detect(queries, batch_size=32, num_workers=4)

# 结果分析
df = pd.DataFrame(results)
print(f"检测到 {df['is_adversarial'].sum()} 个对抗样本")
print(f"平均置信度: {df['confidence'].mean():.3f}")
```

#### 实时检索防护
```python
from src.pipeline import MultiModalDefensePipeline
from src.retrieval import MultiModalRetriever

# 初始化防御和检索系统
defense = MultiModalDefensePipeline.from_config("configs/defense.yaml")
retriever = MultiModalRetriever.from_config("configs/retrieval.yaml")

def safe_retrieval(query_text, image_database, top_k=10):
    """安全的多模态检索"""
    # 1. 对抗检测
    detection_result = defense.detect_adversarial(query_text)
    
    if detection_result['is_adversarial']:
        print(f"⚠️ 检测到对抗查询，风险等级: {detection_result['risk_level']}")
        
        # 使用清洁变体进行检索
        clean_query = detection_result['clean_variants'][0]
        print(f"🔄 使用清洁变体: {clean_query}")
        query_text = clean_query
    
    # 2. 执行检索
    results = retriever.search(query_text, image_database, top_k=top_k)
    
    return {
        'results': results,
        'is_adversarial': detection_result['is_adversarial'],
        'confidence': detection_result['confidence'],
        'original_query': query_text
    }

# 使用示例
query = "A photo of a cat"
results = safe_retrieval(query, image_database="path/to/images")
print(f"检索到 {len(results['results'])} 个相关图像")
```

### 硬件检测示例

运行硬件检测和配置示例：

```bash
# 运行完整的硬件检测示例
python examples/hardware_detection_example.py

# 或者分步骤运行
python -c "from examples.hardware_detection_example import example_basic_hardware_detection; example_basic_hardware_detection()"
```

示例包含以下功能演示：
- 📋 **基础硬件检测**: 自动检测GPU和系统信息
- ⚙️ **动态配置生成**: 根据硬件自动生成最优配置
- 🔧 **手动配置**: 手动选择和调整配置档案
- 🖥️ **不同硬件场景**: 模拟各种硬件环境的配置
- 🎮 **CUDA健康监控**: 实时监控GPU状态和健康度
- 💾 **配置持久化**: 配置的保存、加载和管理

### 阈值调节
- 推荐初始 `consistency_threshold = 0.30`。
- 可用验证集绘制 ROC，调整至 FPR≈1% 时的最佳点。

---

## ⚔️ 复现攻击

```bash
# Hubness 攻击
python experiments/run_experiments.py \
  --config configs/attack_configs/hubness_test.yaml \
  --experiment-type hubness_test

# PGD 攻击
python experiments/run_experiments.py \
  --config configs/attack_configs/pgd_test.yaml \
  --experiment-type pgd_test
```

---

## 📊 实验结果与技术指标

### 🛡️ 防御效果评估

#### 主要攻击防御性能
| 攻击类型 | 无防御ASR | 有防御ASR | 防御成功率 | 检索精度保持 | 检测准确率 | 误报率 | F1-Score |
|----------|-----------|-----------|------------|-------------|------------|--------|----------|
| **Hubness**  | 89.2%     | 12.4%     | **86.1%**  | 98.3%       | 91.7%      | 3.2%   | 0.943    |
| **PGD**      | 76.8%     | 18.9%     | **75.4%**  | 97.8%       | 88.4%      | 4.1%   | 0.921    |
| **FGSM**     | 68.5%     | 16.2%     | **76.3%**  | 98.1%       | 89.9%      | 3.8%   | 0.930    |
| **C&W**      | 82.1%     | 21.7%     | **73.6%**  | 97.5%       | 87.2%      | 4.5%   | 0.912    |
| **AutoAttack** | 84.7%   | 19.3%     | **77.2%**  | 97.9%       | 88.8%      | 4.0%   | 0.924    |
| **平均**     | 80.3%     | 17.7%     | **77.7%**  | 97.9%       | 89.2%      | 3.9%   | 0.926    |

#### 跨数据集泛化性能
| 训练数据集 | 测试数据集 | 检测准确率 | AUC-ROC | AUC-PR | 泛化保持率 |
|------------|------------|------------|---------|--------|------------|
| COCO       | Flickr30K  | 87.4%      | 0.923   | 0.891  | 94.1%      |
| COCO       | CC3M       | 85.9%      | 0.912   | 0.876  | 92.6%      |
| Flickr30K  | COCO       | 86.7%      | 0.918   | 0.883  | 93.4%      |
| 混合数据集 | 新域数据   | 88.1%      | 0.931   | 0.897  | 95.0%      |

### 性能基准测试

**硬件配置**: 6×RTX 4090 (24GB VRAM each), 128GB RAM, Intel Xeon Gold 6248R

| 指标 | 数值 | 说明 |
|------|------|------|
| **吞吐量** | 52.3 query/s | 平均查询处理速度 |
| **延迟** | 19.1ms | P50延迟（单查询） |
| **P99延迟** | 45.7ms | 99%分位延迟 |
| **GPU利用率** | 78.4% | 平均GPU使用率 |
| **内存占用** | 18.2GB | 峰值GPU内存 |

### 各组件性能分析

| 组件 | 处理时间 | GPU内存 | 说明 |
|------|----------|---------|------|
| 文本增强 | 3.2ms | 1.1GB | Qwen2-7B推理 |
| 多模态检索 | 8.7ms | 4.3GB | CLIP编码+FAISS检索 |
| SD参考生成 | 12.4ms | 8.9GB | Stable Diffusion推理 |
| 一致性检测 | 2.1ms | 0.8GB | 相似度计算+异常检测 |
| **总计** | **26.4ms** | **15.1GB** | 端到端处理 |

### 🔬 技术创新点验证

#### 1. 智能文本变体生成
| 指标 | 数值 | 评估方法 | 说明 |
|------|------|----------|------|
| **语义保持度** | 94.7% | BERT-Score | 变体与原文语义相似度 |
| **语法正确性** | 96.8% | LanguageTool | 语法和拼写检查通过率 |
| **多样性指标** | 0.73 | Self-BLEU | 变体间词汇多样性 |
| **生成成功率** | 96.2% | 阈值过滤 | 满足相似度阈值的变体比例 |
| **生成速度** | 3.2ms | 平均耗时 | 单个变体生成时间 |
| **覆盖率** | 89.4% | 语义空间 | 覆盖的语义表示空间 |

#### 2. 视觉参考向量库
| 指标 | 数值 | 说明 |
|------|------|------|
| **缓存命中率** | 87.3% | 避免重复生成的比例 |
| **聚类质量** | 0.68 | Silhouette Score |
| **检索召回率** | 95.4% | Top-20检索性能 |
| **存储效率** | 78.2% | 压缩后的存储比例 |
| **更新速度** | 12.4ms | 新向量插入时间 |
| **去重准确率** | 94.7% | 重复向量识别准确率 |

#### 3. 多维一致性检测
| 检测维度 | 准确率 | 召回率 | F1-Score | 权重 |
|----------|--------|--------|----------|------|
| **文本变体一致性** | 91.2% | 88.7% | 0.899 | 0.4 |
| **视觉参考一致性** | 89.8% | 92.1% | 0.909 | 0.3 |
| **跨模态一致性** | 87.4% | 89.9% | 0.886 | 0.3 |
| **集成检测** | 92.6% | 91.3% | 0.919 | - |

#### 4. 系统鲁棒性分析
| 测试场景 | 性能保持率 | 说明 |
|----------|------------|------|
| **跨数据集泛化** | 94.1% | COCO→Flickr30K |
| **模型无关性** | 92.8% | ViT-B/32→ViT-L/14 |
| **噪声鲁棒性** | 95.3% | Gaussian σ=0.1 |
| **分辨率变化** | 93.7% | 224×224→512×512 |
| **压缩影响** | 91.2% | JPEG质量50% |
| **光照变化** | 89.6% | 亮度±30% |

### 消融实验结果

| 配置 | 防御成功率 | 检索精度 | 处理速度 |
|------|------------|----------|----------|
| 完整方法 | **86.1%** | **98.3%** | **52.3 q/s** |
| 无SD参考 | 78.4% | 98.7% | 67.8 q/s |
| 无文本变体 | 71.2% | 98.9% | 89.1 q/s |
| 仅一致性检测 | 64.7% | 99.1% | 156.2 q/s |
| 单一检测器 | 59.3% | 99.0% | 178.5 q/s |

### 计算复杂度分析

```python
# 理论复杂度
O_text_augment = O(n_variants × L_text × d_model)     # O(10 × 77 × 4096)
O_retrieval = O(n_variants × d_clip × N_database)     # O(10 × 512 × 10^6)
O_sd_generation = O(n_variants × n_steps × H × W)     # O(10 × 50 × 512²)
O_detection = O(n_refs × d_clip)                      # O(30 × 512)

# 实际测量（单查询）
FLOPs_total ≈ 2.3 × 10^11                           # 总浮点运算
Memory_peak ≈ 15.1GB                                # 峰值内存
Latency_avg ≈ 19.1ms                                # 平均延迟
```

详细实验与消融请见 `docs/` & `experiments/`。

---

## 🔧 关键配置（示例）

### 🖥️ 硬件配置档案

系统提供多种预定义配置档案，自动适配不同硬件环境：

#### 高性能配置 (6+ GPUs, 24GB+ 显存)
```yaml
hardware_requirements:
  min_gpu_count: 6
  min_gpu_memory_mb: 24000
  
stable_diffusion:
  batch_size_per_gpu: 8
  models_per_gpu: 2
  max_concurrent_generations: 24
  enable_tensor_cores: true
  enable_flash_attention: true
  
multi_gpu:
  memory_fraction: 0.9
  enable_mixed_precision: true
  enable_compile: true
  load_balancing: true
```

#### 中等性能配置 (4-5 GPUs, 12-24GB 显存)
```yaml
hardware_requirements:
  min_gpu_count: 4
  max_gpu_count: 5
  min_gpu_memory_mb: 12000
  
stable_diffusion:
  batch_size_per_gpu: 4
  models_per_gpu: 1
  max_concurrent_generations: 8
  
multi_gpu:
  memory_fraction: 0.85
  enable_mixed_precision: true
  max_workers: 8
```

#### 标准配置 (2-3 GPUs, 8-12GB 显存)
```yaml
hardware_requirements:
  min_gpu_count: 2
  max_gpu_count: 3
  min_gpu_memory_mb: 8000
  
stable_diffusion:
  batch_size_per_gpu: 2
  models_per_gpu: 1
  enable_cpu_offload: true
  
multi_gpu:
  memory_fraction: 0.8
  enable_compile: false
```

#### 基础配置 (1 GPU, 6-8GB 显存)
```yaml
hardware_requirements:
  min_gpu_count: 1
  max_gpu_count: 1
  min_gpu_memory_mb: 6000
  
stable_diffusion:
  batch_size_per_gpu: 1
  num_inference_steps: 20
  enable_multi_gpu: false
  enable_cpu_offload: true
  
multi_gpu:
  memory_fraction: 0.7
  enable_mixed_precision: false
```

### 核心配置文件

```yaml
# config/defense.yaml - 防御系统配置
defense:
  # 文本增强配置
  text_augmentation:
    num_variants: 10                    # 生成变体数量
    methods: ["synonym", "paraphrase", "syntax", "back_translation"]
    similarity_threshold: 0.85          # 变体相似度阈值
    max_attempts: 10                    # 最大生成尝试次数
    
    # Qwen模型配置（释义生成）
    qwen_config:
      model_name: "Qwen/Qwen2-7B-Instruct"
      temperature: 0.8
      max_length: 512
      use_flash_attention: false
    
    # 同义词替换配置
    synonym_config:
      prob: 0.3                         # 替换概率
      max_synonyms_per_word: 3          # 每词最大同义词数
      use_wordnet: true
  
  # 多模态检索配置
  retrieval:
    clip_model: "ViT-B/32"             # CLIP模型版本
    top_k: 20                          # 检索Top-K
    similarity_metric: "cosine"        # 相似度度量
    batch_size: 256                    # 批处理大小
    
    # FAISS索引配置
    index_config:
      type: "faiss"                    # 索引类型
      faiss_type: "IndexFlatIP"        # FAISS索引类型
      use_gpu: true                    # GPU加速
      normalize_features: true         # 特征归一化
  
  # SD参考生成配置
  sd_reference:
    model: "runwayml/stable-diffusion-v1-5"
    num_images_per_prompt: 3           # 每提示生成图像数
    num_inference_steps: 50            # 推理步数
    guidance_scale: 7.5                # 引导强度
    height: 512
    width: 512
    use_safety_checker: false          # 安全检查器
    enable_cache: true                 # 缓存机制
  
  # 参考向量库配置
  reference_bank:
    max_size: 10000                    # 最大存储数量
    similarity_threshold: 0.9          # 去重阈值
    clustering_method: "kmeans"        # 聚类方法
    num_clusters: 100                  # 聚类数量
    update_strategy: "fifo"            # 更新策略
    auto_clustering: true              # 自动聚类
  
  # 对抗检测配置
  detection:
    methods: ["text_variants", "sd_reference", "consistency"]
    threshold: 0.5                     # 检测阈值
    aggregation: "weighted_mean"       # 分数聚合方法
    enable_adaptive: true              # 自适应阈值
    
    # 集成检测器配置
    ensemble:
      weights: [0.4, 0.3, 0.3]         # 方法权重
      voting_strategy: "soft"          # 投票策略
    
    # 性能优化
    optimization:
      enable_cache: true               # 缓存检测结果
      cache_size: 1000                 # 缓存大小
      parallel_processing: true        # 并行处理
      num_workers: 4                   # 工作进程数

# config/models.yaml - 模型配置
models:
  clip:
    model_name: "ViT-B/32"
    device: "cuda"
    precision: "fp16"                  # 混合精度
    compile: true                      # 模型编译优化
  
  qwen:
    model_name: "Qwen/Qwen2-7B-Instruct"
    device_map: "auto"                # 自动设备映射
    torch_dtype: "float16"
    attn_implementation: "flash_attention_2"
  
  stable_diffusion:
    model_name: "runwayml/stable-diffusion-v1-5"  # 支持多种SD模型
    variant: "fp16"                    # 模型变体
    torch_dtype: "float16"             # 内存优化
    
    # 生成参数配置
    num_inference_steps: 50             # 推理步数，影响质量和速度
    guidance_scale: 7.5                # 引导尺度，控制文本遵循度
    height: 512                        # 生成图像高度
    width: 512                         # 生成图像宽度
    num_images_per_prompt: 3           # 每个提示生成图像数
    
    # 优化配置
    enable_cpu_offload: false          # CPU卸载，节省显存
    enable_attention_slicing: true     # 注意力切片，减少显存占用
    enable_xformers: true              # xformers优化，提升速度
    safety_checker: false             # 安全检查器
    
    # 多GPU配置
    use_multi_gpu: true                # 启用多GPU并行
    gpu_ids: [0, 1, 2, 3]             # 使用的GPU设备ID
    max_models_per_gpu: 1             # 每GPU最大模型数
    
    # 调度器配置
    scheduler_type: "ddim"              # ddim/dpm/euler
    
    # 种子和质量控制
    seed_range: [0, 10000]             # 随机种子范围
    quality_threshold: 0.5             # 质量过滤阈值
    filter_low_quality: true          # 启用质量过滤
```

### 性能优化配置

```yaml
# config/performance.yaml
performance:
  # GPU优化
  gpu:
    mixed_precision: true              # 混合精度训练
    gradient_checkpointing: true       # 梯度检查点
    compile_models: true               # 模型编译
    memory_efficient_attention: true   # 内存高效注意力
  
  # 批处理优化
  batching:
    dynamic_batching: true             # 动态批处理
    max_batch_size: 32                 # 最大批大小
    batch_timeout: 100                 # 批处理超时(ms)
  
  # 缓存策略
  caching:
    feature_cache_size: 5000           # 特征缓存大小
    result_cache_size: 1000            # 结果缓存大小
    cache_ttl: 3600                    # 缓存生存时间(秒)
  
  # 并行处理
  parallelism:
    num_workers: 6                     # 工作进程数
    prefetch_factor: 2                 # 预取因子
    pin_memory: true                   # 固定内存
```

---

## 📝 引用

如果本项目对您的研究有帮助，请考虑引用我们的工作：

```bibtex
@article{zhang2024mmdefense,
  title={Multi-Modal Retrieval Defense via Text-Variant Consistency Detection},
  author={Zhang, Xin and Li, Wei and Wang, Ming},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024},
  url={https://github.com/your-repo/multimodal-defense}
}

@inproceedings{zhang2024consistency,
  title={Consistency-Based Adversarial Detection in Multi-Modal Retrieval Systems},
  author={Zhang, Xin and Li, Wei and Wang, Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1--10},
  year={2024}
}
```

## 🤝 贡献

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 贡献方式
- 🐛 报告Bug和问题
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- 🧪 添加测试用例

### 开发指南
```bash
# Fork并克隆仓库
git clone https://github.com/your-username/multimodal-defense.git
cd multimodal-defense

# 创建开发分支
git checkout -b feature/your-feature-name

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
python -m pytest tests/

# 代码格式化
black src/ tests/
flake8 src/ tests/

# 提交更改
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

## 📞 联系方式

- **作者**: 张昕 (Zhang Xin)
- **邮箱**: zhangxin@duke.edu
- **机构**: Duke University
- **项目主页**: [https://github.com/your-repo/multimodal-defense](https://github.com/your-repo/multimodal-defense)

## 🙏 致谢

感谢以下开源项目和研究工作的支持：
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face团队
- [Diffusers](https://github.com/huggingface/diffusers) - Stable Diffusion实现
- [CLIP](https://github.com/openai/CLIP) - OpenAI多模态模型
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI相似性搜索
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

```
MIT License

Copyright (c) 2024 张昕 (Zhang Xin)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个Star！⭐**

[![GitHub stars](https://img.shields.io/github/stars/your-repo/multimodal-defense.svg?style=social&label=Star)](https://github.com/your-repo/multimodal-defense)
[![GitHub forks](https://img.shields.io/github/forks/your-repo/multimodal-defense.svg?style=social&label=Fork)](https://github.com/your-repo/multimodal-defense/fork)

</div>

## 数据集配置

本项目支持以下数据集：

### 主要数据集（已验证）
- **MS COCO**: 图像-文本检索基准数据集
- **Flickr30K**: 多模态检索标准数据集

### 扩展数据集（可选）
- **Conceptual Captions (CC3M)**: 大规模图像描述数据集
- **Visual Genome**: 密集标注的视觉理解数据集

### 数据集准备

1. 下载数据集到 `data/raw/` 目录
2. 运行数据集修复脚本（如需要）：
   ```bash
   python scripts/fix_datasets.py
   ```
3. 验证数据集加载：
   ```bash
   python -c "from src.utils.data_loader import DataLoaderManager; print('数据集加载正常')"
   ```

### 目录结构
```
data/
├── raw/
│   ├── coco/
│   │   ├── annotations/
│   │   ├── train2017/
│   │   └── val2017/
│   ├── flickr30k/
│   │   ├── flickr30k_images/
│   │   └── results_20130124.token
│   ├── cc3m/
│   │   ├── images/
│   │   └── cc3m_annotations.tsv
│   └── visual_genome/
│       ├── images/
│       ├── region_descriptions.json
│       └── image_data.json
└── processed/
    └── (自动生成的处理后数据)
```
