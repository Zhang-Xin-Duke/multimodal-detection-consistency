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

# 导入主要模块
from . import text_augment
from . import retrieval
from . import sd_ref
from . import ref_bank
from . import detector
# 延迟导入pipeline以避免循环导入

# 导入子包
from . import attacks
from . import models
from . import utils
from . import evaluation

# 导入主要类
from .text_augment import TextAugmenter, TextAugmentConfig
from .retrieval import MultiModalRetriever, RetrievalConfig
from .sd_ref import SDReferenceGenerator, SDReferenceConfig
from .ref_bank import ReferenceBank, ReferenceBankConfig
from .detector import AdversarialDetector, DetectorConfig
# 延迟导入pipeline类以避免循环导入

# 导入工厂函数
from .text_augment import create_text_augmenter
from .retrieval import create_retriever
from .sd_ref import create_sd_reference_generator
from .ref_bank import create_reference_bank
from .detector import create_adversarial_detector
# 延迟导入pipeline工厂函数以避免循环导入

# 导入模型
from .models import CLIPModel, CLIPConfig
from .models import create_clip_model

# 导入攻击模块
from .attacks import HubnessAttacker, HubnessAttackConfig
from .attacks import create_hubness_attacker

# 导入评估模块
from .evaluation import DataValidator, ExperimentEvaluator
from .evaluation import create_data_validator, create_experiment_evaluator

# 导入工具
from .utils import load_config, save_config
from .utils import create_metrics_calculator, create_experiment_visualizer

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    
    # 主要配置类
    "TextAugmentConfig",
    "RetrievalConfig", 
    "SDReferenceConfig",
    "ReferenceBankConfig",
    "DetectorConfig",
    "CLIPConfig",
    "HubnessAttackConfig",
    
    # 主要功能类
    "TextAugmenter",
    "MultiModalRetriever",
    "SDReferenceGenerator",
    "ReferenceBank",
    "AdversarialDetector",
    "CLIPModel",
    "HubnessAttacker",
    "DataValidator",
    "ExperimentEvaluator",
    
    # 工厂函数
    "create_text_augmenter",
    "create_retriever",
    "create_sd_reference_generator",
    "create_reference_bank",
    "create_adversarial_detector",
    "create_clip_model",
    "create_hubness_attacker",
    "create_data_validator",
    "create_experiment_evaluator",
    
    # 工具函数
    "load_config",
    "save_config",
    "create_metrics_calculator",
    "create_experiment_visualizer",
    
    # 子模块
    "text_augment",
    "retrieval",
    "sd_ref",
    "ref_bank",
    "detector",
    "attacks",
    "models",
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

# 支持的数据集
SUPPORTED_DATASETS = [
    "coco",
    "flickr30k",
    "conceptual_captions",
    "visual_genome"
]

def get_version():
    """获取版本信息"""
    return __version__

def get_supported_models():
    """获取支持的模型列表"""
    return {
        "clip": SUPPORTED_CLIP_MODELS,
        "stable_diffusion": SUPPORTED_SD_MODELS
    }

def get_supported_datasets():
    """获取支持的数据集列表"""
    return SUPPORTED_DATASETS

def get_default_config():
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()