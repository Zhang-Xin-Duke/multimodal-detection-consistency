"""配置管理模块

提供统一的配置加载、验证和管理功能。
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    # CLIP配置
    clip_model_name: str = "ViT-B/32"
    clip_device: str = "cuda"
    clip_batch_size: int = 256
    
    # Qwen配置
    qwen_api_key: str = "sk-fda43c38cc2f4012ac44da3848e79586"
    qwen_model_name: str = "qwen-turbo"
    qwen_max_tokens: int = 512
    qwen_temperature: float = 0.7
    
    # Stable Diffusion配置
    sd_model_name: str = "runwayml/stable-diffusion-v1-5"
    sd_device: str = "cuda"
    sd_batch_size: int = 4
    sd_num_inference_steps: int = 20
    sd_guidance_scale: float = 7.5
    sd_height: int = 512
    sd_width: int = 512


@dataclass
class DataConfig:
    """数据配置"""
    # 数据集路径
    coco_root: str = "./data/coco"
    flickr30k_root: str = "./data/flickr30k"
    processed_root: str = "./data/processed"
    
    # 数据加载配置
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    
    # 预处理配置
    image_size: int = 224
    normalize_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 基础配置
    experiment_name: str = "multimodal_consistency_experiment"
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    seed: int = 42
    
    # 硬件配置
    device: str = "cuda"
    num_gpus: int = 6
    gpu_memory_fraction: float = 0.8
    
    # 缓存配置
    enable_cache: bool = True
    cache_dir: str = "./cache"
    cache_size_limit: str = "50GB"


@dataclass
class AttackConfig:
    """攻击配置"""
    # Hubness攻击
    hubness_k: int = 10
    hubness_alpha: float = 0.1
    hubness_max_iterations: int = 100
    
    # PGD攻击
    pgd_epsilon: float = 8.0 / 255.0
    pgd_alpha: float = 2.0 / 255.0
    pgd_steps: int = 10
    
    # 文本攻击
    text_attack_method: str = "textfooler"
    text_attack_max_candidates: int = 50


@dataclass
class DefenseConfig:
    """防御配置"""
    # 文本变体生成
    num_text_variants: int = 10
    text_similarity_threshold: float = 0.85
    text_max_retries: int = 3
    
    # 检索配置
    retrieval_top_k: int = 20
    retrieval_similarity_metric: str = "cosine"
    
    # SD参考生成
    sd_num_images_per_text: int = 3
    sd_enable_cache: bool = True
    
    # 检测配置
    detection_threshold: float = 0.5
    detection_aggregation_method: str = "mean"
    detection_enable_adaptive: bool = True


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估指标
    metrics: list = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc"])
    
    # 评估设置
    test_split_ratio: float = 0.2
    validation_split_ratio: float = 0.1
    cross_validation_folds: int = 5
    
    # 可视化配置
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为None
        """
        self.config_path = config_path
        self.config = {}
        
        # 加载默认配置
        self._load_default_config()
        
        # 如果提供了配置文件路径，则加载配置文件
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self):
        """加载默认配置"""
        self.config = {
            'model': ModelConfig(),
            'models': {},  # 支持新的models配置节
            'data': DataConfig(),
            'experiment': ExperimentConfig(),
            'attack': AttackConfig(),
            'defense': DefenseConfig(),
            'evaluation': EvaluationConfig(),
            # 支持default.yaml中的新配置节
            'text_augment': {},
            'retrieval': {},
            'sd_reference': {},
            'reference_bank': {},
            'detector': {},
            'pipeline': {},
            'logging': {},
            'visualization': {},
            'cache': {}
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            加载的配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            # 更新配置
            self._update_config(yaml_config)
            
            logger.info(f"成功加载配置文件: {config_path}")
            return self.config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _update_config(self, yaml_config: Dict[str, Any]):
        """更新配置"""
        for section, values in yaml_config.items():
            if section in self.config:
                if isinstance(values, dict):
                    # 如果是dataclass对象，更新其字段
                    config_obj = self.config[section]
                    if hasattr(config_obj, '__dict__'):
                        for key, value in values.items():
                            if hasattr(config_obj, key):
                                setattr(config_obj, key, value)
                            else:
                                logger.debug(f"配置项 {section}.{key} 不在预定义字段中，将直接存储")
                    else:
                        # 如果是字典，直接更新
                        self.config[section].update(values)
                else:
                    self.config[section] = values
            else:
                # 对于新的配置节，直接添加
                self.config[section] = values
                logger.debug(f"添加新配置节: {section}")
    
    def save_config(self, output_path: str):
        """
        保存当前配置到YAML文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            # 转换dataclass为字典
            config_dict = {}
            for section, config_obj in self.config.items():
                if hasattr(config_obj, '__dict__'):
                    config_dict[section] = config_obj.__dict__
                else:
                    config_dict[section] = config_obj
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            logger.info(f"配置已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def get_config(self, section: str) -> Any:
        """
        获取指定节的配置
        
        Args:
            section: 配置节名称
            
        Returns:
            配置对象
            
        Raises:
            KeyError: 配置节不存在
        """
        if section not in self.config:
            raise KeyError(f"配置节不存在: {section}")
        return self.config[section]
    
    def update_config(self, section: str, key: str, value: Any):
        """
        更新指定配置项
        
        Args:
            section: 配置节名称
            key: 配置项名称
            value: 新值
        """
        if section not in self.config:
            raise KeyError(f"配置节不存在: {section}")
        
        config_obj = self.config[section]
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)
            logger.info(f"更新配置: {section}.{key} = {value}")
        else:
            raise KeyError(f"配置项不存在: {section}.{key}")
    
    def validate_config(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
        """
        try:
            # 验证设备配置
            import torch
            if 'cuda' in self.config['experiment'].device and not torch.cuda.is_available():
                logger.warning("CUDA不可用，将使用CPU")
                self.config['experiment'].device = 'cpu'
                self.config['model'].clip_device = 'cpu'
                self.config['model'].sd_device = 'cpu'
            
            # 验证路径
            data_config = self.config['data']
            for path_attr in ['coco_root', 'flickr30k_root', 'processed_root']:
                path = getattr(data_config, path_attr)
                Path(path).mkdir(parents=True, exist_ok=True)
            
            # 验证输出目录
            exp_config = self.config['experiment']
            Path(exp_config.output_dir).mkdir(parents=True, exist_ok=True)
            Path(exp_config.cache_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def get_env_config(self) -> Dict[str, str]:
        """
        获取环境变量配置
        
        Returns:
            环境变量字典
        """
        env_config = {}
        
        # 从.env文件加载环境变量
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_config[key.strip()] = value.strip()
        
        # 添加系统环境变量
        for key in ['CUDA_VISIBLE_DEVICES', 'PYTHONPATH']:
            if key in os.environ:
                env_config[key] = os.environ[key]
        
        return env_config


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config(section: str = None) -> Union[Dict[str, Any], Any]:
    """
    获取配置
    
    Args:
        section: 配置节名称，如果为None则返回所有配置
        
    Returns:
        配置对象或配置字典
    """
    if section is None:
        return config_manager.config
    return config_manager.get_config(section)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    return config_manager.load_config(config_path)


def save_config(output_path: str):
    """
    保存配置到文件
    
    Args:
        output_path: 输出文件路径
    """
    config_manager.save_config(output_path)


def update_config(section: str, key: str, value: Any):
    """
    更新配置项
    
    Args:
        section: 配置节名称
        key: 配置项名称
        value: 新值
    """
    config_manager.update_config(section, key, value)


def validate_config() -> bool:
    """
    验证配置有效性
    
    Returns:
        配置是否有效
    """
    return config_manager.validate_config()


def get_qwen_cache_dir() -> str:
    """
    获取Qwen模型的缓存目录
    
    Returns:
        Qwen模型缓存目录路径
    """
    try:
        # 尝试从配置中获取
        if hasattr(config_manager.config.get('models', {}), 'get'):
            qwen_config = config_manager.config['models'].get('qwen', {})
            if isinstance(qwen_config, dict) and 'cache_dir' in qwen_config:
                return qwen_config['cache_dir']
        
        # 如果配置中没有，返回默认值
        return "./cache/qwen"
        
    except Exception as e:
        logger.warning(f"获取Qwen缓存目录失败，使用默认值: {e}")
        return "./cache/qwen"