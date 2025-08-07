"""多模态检测一致性实验代码包

本包提供了多模态对抗检测的完整实现，包括：
- 文本变体生成
- 多模态检索
- Stable Diffusion参考生成
- 参考向量库管理
- 对抗检测
- 实验流水线
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# 导入配置管理
from .config import Config, GlobalConfig

# 导入主要模块
from . import text_augment
from . import retrieval
from . import sd_ref
from . import ref_bank
from . import detector
from . import pipeline

# 导入子包
from . import attacks
from . import utils
from . import evaluation

# 导入主要类（这些类需要在相应模块中实现）
# from .text_augment import TextAugmenter, TextAugmentConfig
# from .retrieval import MultiModalRetriever, RetrievalConfig
# from .sd_ref import SDReferenceGenerator, SDReferenceConfig
# from .ref_bank import ReferenceBank, ReferenceBankConfig
# from .detector import AdversarialDetector, DetectorConfig
# from .pipeline import MultiModalDetectionPipeline, PipelineConfig

# 模型相关导入（需要时再添加）
# from .models import CLIPModel, CLIPConfig
# from .models import QwenModel, QwenConfig
# from .models import StableDiffusionModel, StableDiffusionConfig

# 攻击模块导入（需要时再添加）
# from .attacks import PGDAttacker, HubnessAttacker, TextAttacker, FGSMAttacker

# 导入数据集加载器（需要时再添加）
# from .datasets import COCOLoader, FlickrLoader, CCLoader, VGLoader

# 评估模块导入（需要时再添加）
# from .evaluation import DataValidator, ExperimentEvaluator

# 工具模块导入（需要时再添加）
# from .utils import (
#     ConfigManager,
#     HardwareDetector,
#     MultiGPUProcessor,
#     CUDADeviceManager,
#     SimilarityCalculator,
#     DetectionEvaluator,
#     RetrievalEvaluator,
#     MetricsAggregator,
#     VisualizationManager
# )

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    
    # 配置管理
    "Config",
    "GlobalConfig",
    
    # 子模块
    "text_augment",
    "retrieval",
    "sd_ref",
    "ref_bank",
    "detector",
    "pipeline",
    "attacks",
    "utils",
    "evaluation"
]

# 包级别的配置
DEFAULT_CONFIG = {
    "device": "cuda",
    "batch_size": 32,
    "num_workers": 4,
    "random_seed": 42,
    "log_level": "INFO"
}

# 支持的模型列表
SUPPORTED_CLIP_MODELS = [
    "ViT-B/32",
    "ViT-B/16", 
    "ViT-L/14",
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64"
]

SUPPORTED_SD_MODELS = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0"
]

SUPPORTED_QWEN_MODELS = [
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct"
]

# 支持的数据集
SUPPORTED_DATASETS = [
    "coco",
    "flickr30k",
    "conceptual_captions",
    "visual_genome"
]

# 支持的攻击方法
SUPPORTED_ATTACKS = [
    "pgd",
    "hubness",
    "fsta",
    "sma"
]

def get_version():
    """获取版本信息"""
    return __version__

def get_supported_models():
    """获取支持的模型列表"""
    return {
        "clip": SUPPORTED_CLIP_MODELS,
        "stable_diffusion": SUPPORTED_SD_MODELS,
        "qwen": SUPPORTED_QWEN_MODELS
    }

def get_supported_datasets():
    """获取支持的数据集列表"""
    return SUPPORTED_DATASETS

def get_supported_attacks():
    """获取支持的攻击方法列表"""
    return SUPPORTED_ATTACKS

def get_default_config():
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()