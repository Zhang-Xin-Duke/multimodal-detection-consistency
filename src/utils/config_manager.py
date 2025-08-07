"""配置管理器模块

提供配置文件的加载、合并、验证和保存功能。
支持YAML格式配置文件，环境变量覆盖，配置继承等。
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器
    
    负责配置文件的加载、合并、验证和保存。
    """
    
    def __init__(self):
        """初始化配置管理器"""
        self.loaded_configs: Dict[str, Dict[str, Any]] = {}
        self.config_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # 检查缓存
        cache_key = str(config_path.absolute())
        if cache_key in self.config_cache:
            logger.debug(f"Loading config from cache: {config_path}")
            return deepcopy(self.config_cache[cache_key])
        
        logger.info(f"Loading config from file: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            if config is None:
                config = {}
            
            # 应用环境变量覆盖
            config = self._apply_env_overrides(config)
            
            # 缓存配置
            self.config_cache[cache_key] = deepcopy(config)
            
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config {config_path}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON config {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], output_path: Union[str, Path]):
        """保存配置到文件
        
        Args:
            config: 配置字典
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving config to: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
                elif output_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported output format: {output_path.suffix}")
        except Exception as e:
            logger.error(f"Error saving config to {output_path}: {e}")
            raise
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个配置
        
        Args:
            *configs: 配置字典列表，后面的配置会覆盖前面的
            
        Returns:
            合并后的配置字典
        """
        if not configs:
            return {}
        
        merged = deepcopy(configs[0])
        
        for config in configs[1:]:
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
            
        Returns:
            合并后的字典
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """应用配置覆盖
        
        Args:
            config: 基础配置
            overrides: 覆盖配置
            
        Returns:
            应用覆盖后的配置
        """
        return self._deep_merge(config, overrides)
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖
        
        环境变量格式: CONFIG_SECTION_KEY=value
        例如: CONFIG_MODEL_DEVICE=cpu
        
        Args:
            config: 配置字典
            
        Returns:
            应用环境变量覆盖后的配置
        """
        env_prefix = "CONFIG_"
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue
            
            # 解析环境变量键
            config_key = env_key[len(env_prefix):].lower()
            key_parts = config_key.split('_')
            
            # 转换环境变量值
            value = self._convert_env_value(env_value)
            
            # 设置配置值
            self._set_nested_value(config, key_parts, value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """转换环境变量值到合适的类型
        
        Args:
            value: 环境变量字符串值
            
        Returns:
            转换后的值
        """
        # 布尔值
        if value.lower() in ['true', 'yes', '1', 'on']:
            return True
        elif value.lower() in ['false', 'no', '0', 'off']:
            return False
        
        # 数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON格式
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # 逗号分隔的列表
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # 字符串
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key_parts: List[str], value: Any):
        """设置嵌套配置值
        
        Args:
            config: 配置字典
            key_parts: 键路径列表
            value: 要设置的值
        """
        current = config
        
        for key_part in key_parts[:-1]:
            if key_part not in current:
                current[key_part] = {}
            elif not isinstance(current[key_part], dict):
                # 如果不是字典，创建新的字典
                current[key_part] = {}
            current = current[key_part]
        
        current[key_parts[-1]] = value
    
    def validate_config(self, config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
        """验证配置
        
        Args:
            config: 待验证的配置
            schema: 配置模式（可选）
            
        Returns:
            是否有效
        """
        if schema is None:
            # 基本验证：检查配置是否为字典
            return isinstance(config, dict)
        
        # 使用模式验证（简单实现）
        return self._validate_against_schema(config, schema)
    
    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """根据模式验证配置
        
        Args:
            config: 配置字典
            schema: 模式字典
            
        Returns:
            是否有效
        """
        try:
            # 检查必需字段
            if 'required' in schema:
                for required_field in schema['required']:
                    if required_field not in config:
                        logger.error(f"Missing required field: {required_field}")
                        return False
            
            # 检查字段类型
            if 'properties' in schema:
                for field, field_schema in schema['properties'].items():
                    if field in config:
                        if not self._validate_field_type(config[field], field_schema):
                            logger.error(f"Invalid type for field {field}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False
    
    def _validate_field_type(self, value: Any, field_schema: Dict[str, Any]) -> bool:
        """验证字段类型
        
        Args:
            value: 字段值
            field_schema: 字段模式
            
        Returns:
            是否有效
        """
        if 'type' not in field_schema:
            return True
        
        expected_type = field_schema['type']
        
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        if expected_type in type_mapping:
            return isinstance(value, type_mapping[expected_type])
        
        return True
    
    def get_config_template(self, config_type: str) -> Dict[str, Any]:
        """获取配置模板
        
        Args:
            config_type: 配置类型
            
        Returns:
            配置模板
        """
        templates = {
            'dataset': {
                'name': 'dataset_name',
                'type': 'image_caption',
                'data_dir': './data/raw',
                'processed_dir': './data/processed',
                'splits': {
                    'train': {'size': 1000},
                    'val': {'size': 100},
                    'test': {'size': 100}
                },
                'preprocessing': {
                    'image': {'resize': [224, 224]},
                    'text': {'max_length': 77}
                }
            },
            'attack': {
                'name': 'attack_name',
                'type': 'gradient_based',
                'attack_params': {
                    'epsilon': 0.03137,
                    'num_steps': 100
                },
                'batch': {'size': 32}
            },
            'defense': {
                'name': 'defense_name',
                'type': 'consistency_based',
                'consistency': {
                    'direct': {'enabled': True, 'threshold': 0.5}
                },
                'detector': {
                    'type': 'threshold_based',
                    'confidence_threshold': 0.8
                }
            },
            'experiment': {
                'name': 'experiment_name',
                'description': 'Experiment description',
                'inherits': [],
                'overrides': {},
                'workflow': {
                    'stages': ['data_preparation', 'evaluation']
                }
            }
        }
        
        return templates.get(config_type, {})
    
    def clear_cache(self):
        """清空配置缓存"""
        self.config_cache.clear()
        logger.info("Config cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息
        
        Returns:
            缓存信息字典
        """
        return {
            'cached_configs': len(self.config_cache),
            'cache_keys': list(self.config_cache.keys())
        }