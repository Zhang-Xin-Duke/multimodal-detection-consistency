"""
全局配置管理模块

这个模块提供统一的配置管理接口，聚合utils/config_manager的功能。
支持多级配置文件合并、环境变量覆盖、命令行参数等。
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import ConfigManager
from src.utils.hardware_detector import HardwareDetector


@dataclass
class GlobalConfig:
    """全局配置类
    
    聚合所有子系统的配置，提供统一的访问接口。
    """
    
    # 基础配置
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    config_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "configs")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "cache")
    
    # 设备配置
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    memory_fraction: float = 0.8
    
    # 批处理配置
    batch_size: int = 32
    num_workers: int = 4
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # 实验配置
    seed: int = 42
    reproducible: bool = True
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 86400
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保路径存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 硬件检测和设备配置
        hardware = HardwareDetector()
        if self.device == "cuda" and not hardware.has_cuda():
            self.device = "cpu"
            print("Warning: CUDA not available, falling back to CPU")
        
        # 自动检测可用GPU
        if self.device == "cuda":
            available_gpus = hardware.get_available_gpus()
            if available_gpus:
                self.gpu_ids = available_gpus[:len(self.gpu_ids)]


class Config:
    """全局配置管理器
    
    提供统一的配置访问接口，支持：
    - 多级配置文件合并
    - 环境变量覆盖
    - 命令行参数覆盖
    - 动态配置更新
    """
    
    _instance: Optional['Config'] = None
    _config_manager: Optional[ConfigManager] = None
    _global_config: Optional[GlobalConfig] = None
    _loaded_configs: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls) -> 'Config':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        if self._config_manager is None:
            self._config_manager = ConfigManager()
            self._global_config = GlobalConfig()
            self._load_default_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        default_config_path = self._global_config.config_dir / "default.yaml"
        if default_config_path.exists():
            self._loaded_configs["default"] = self._config_manager.load_config(
                str(default_config_path)
            )
    
    def load_config(self, config_path: Union[str, Path], name: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            name: 配置名称，用于缓存
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self._global_config.config_dir / config_path
        
        if name is None:
            name = config_path.stem
        
        config = self._config_manager.load_config(str(config_path))
        self._loaded_configs[name] = config
        return config
    
    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """加载实验配置
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            合并后的实验配置
        """
        experiment_config_path = self._global_config.config_dir / "experiments" / f"{experiment_name}.yaml"
        
        if not experiment_config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {experiment_config_path}")
        
        # 加载实验配置
        experiment_config = self._config_manager.load_config(str(experiment_config_path))
        
        # 处理继承关系
        if "inherits" in experiment_config:
            base_configs = []
            for inherit_path in experiment_config["inherits"]:
                inherit_config_path = self._global_config.config_dir / inherit_path.lstrip("../")
                base_config = self._config_manager.load_config(str(inherit_config_path))
                base_configs.append(base_config)
            
            # 合并配置
            merged_config = self._config_manager.merge_configs(*base_configs, experiment_config)
        else:
            merged_config = experiment_config
        
        # 应用覆盖配置
        if "overrides" in merged_config:
            merged_config = self._config_manager.apply_overrides(
                merged_config, merged_config["overrides"]
            )
        
        self._loaded_configs[experiment_name] = merged_config
        return merged_config
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """获取已加载的配置
        
        Args:
            name: 配置名称
            
        Returns:
            配置字典
        """
        if name not in self._loaded_configs:
            raise KeyError(f"Config '{name}' not loaded")
        return self._loaded_configs[name]
    
    def get_global_config(self) -> GlobalConfig:
        """获取全局配置
        
        Returns:
            全局配置对象
        """
        return self._global_config
    
    def update_config(self, name: str, updates: Dict[str, Any]):
        """更新配置
        
        Args:
            name: 配置名称
            updates: 更新内容
        """
        if name in self._loaded_configs:
            self._loaded_configs[name] = self._config_manager.merge_configs(
                self._loaded_configs[name], updates
            )
        else:
            self._loaded_configs[name] = updates
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集配置
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集配置
        """
        config_path = f"datasets/{dataset_name}.yaml"
        return self.load_config(config_path, f"dataset_{dataset_name}")
    
    def get_attack_config(self, attack_name: str) -> Dict[str, Any]:
        """获取攻击配置
        
        Args:
            attack_name: 攻击名称
            
        Returns:
            攻击配置
        """
        config_path = f"attacks/{attack_name}.yaml"
        return self.load_config(config_path, f"attack_{attack_name}")
    
    def get_defense_config(self, defense_name: str) -> Dict[str, Any]:
        """获取防御配置
        
        Args:
            defense_name: 防御名称
            
        Returns:
            防御配置
        """
        config_path = f"defenses/{defense_name}.yaml"
        return self.load_config(config_path, f"defense_{defense_name}")
    
    def set_device(self, device: str, gpu_ids: Optional[List[int]] = None):
        """设置设备配置
        
        Args:
            device: 设备类型 ('cuda' 或 'cpu')
            gpu_ids: GPU ID列表
        """
        self._global_config.device = device
        if gpu_ids is not None:
            self._global_config.gpu_ids = gpu_ids
    
    def set_batch_size(self, batch_size: int):
        """设置批处理大小
        
        Args:
            batch_size: 批处理大小
        """
        self._global_config.batch_size = batch_size
    
    def set_seed(self, seed: int):
        """设置随机种子
        
        Args:
            seed: 随机种子
        """
        self._global_config.seed = seed
    
    def enable_cache(self, enabled: bool = True):
        """启用/禁用缓存
        
        Args:
            enabled: 是否启用缓存
        """
        self._global_config.enable_cache = enabled
    
    def get_cache_dir(self, subdir: Optional[str] = None) -> Path:
        """获取缓存目录
        
        Args:
            subdir: 子目录名称
            
        Returns:
            缓存目录路径
        """
        cache_dir = self._global_config.cache_dir
        if subdir:
            cache_dir = cache_dir / subdir
            cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_data_dir(self, subdir: Optional[str] = None) -> Path:
        """获取数据目录
        
        Args:
            subdir: 子目录名称
            
        Returns:
            数据目录路径
        """
        data_dir = self._global_config.data_dir
        if subdir:
            data_dir = data_dir / subdir
            data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def list_available_configs(self) -> Dict[str, List[str]]:
        """列出可用的配置文件
        
        Returns:
            配置文件列表，按类型分组
        """
        config_dir = self._global_config.config_dir
        
        available_configs = {
            "datasets": [],
            "attacks": [],
            "defenses": [],
            "experiments": []
        }
        
        for config_type in available_configs.keys():
            type_dir = config_dir / config_type
            if type_dir.exists():
                for config_file in type_dir.glob("*.yaml"):
                    available_configs[config_type].append(config_file.stem)
        
        return available_configs
    
    def validate_config(self, config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
        """验证配置
        
        Args:
            config: 待验证的配置
            schema: 配置模式（可选）
            
        Returns:
            是否有效
        """
        return self._config_manager.validate_config(config, schema)
    
    def export_config(self, name: str, output_path: Union[str, Path]):
        """导出配置到文件
        
        Args:
            name: 配置名称
            output_path: 输出文件路径
        """
        if name not in self._loaded_configs:
            raise KeyError(f"Config '{name}' not loaded")
        
        self._config_manager.save_config(
            self._loaded_configs[name], str(output_path)
        )


# 全局配置实例
config = Config()


# 便捷函数
def get_config() -> Config:
    """获取全局配置实例"""
    return config


def load_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """加载实验配置的便捷函数"""
    return config.load_experiment_config(experiment_name)


def get_global_config() -> GlobalConfig:
    """获取全局配置的便捷函数"""
    return config.get_global_config()