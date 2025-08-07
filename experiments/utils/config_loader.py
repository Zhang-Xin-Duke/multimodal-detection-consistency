"""配置加载模块

提供YAML配置文件的加载、保存、合并和验证功能
"""

import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import copy

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    # 基本信息
    experiment_name: str
    description: str
    version: str = "1.0"
    
    # 数据集配置
    dataset: Dict[str, Any] = None
    
    # 攻击配置
    attack: Dict[str, Any] = None
    
    # 防御配置
    defense: Dict[str, Any] = None
    
    # 模型配置
    models: Dict[str, Any] = None
    
    # 评估配置
    evaluation: Dict[str, Any] = None
    
    # 输出配置
    output: Dict[str, Any] = None
    
    # 硬件配置
    hardware: Dict[str, Any] = None
    
    # 调试配置
    debug: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 设置默认值
        if self.dataset is None:
            self.dataset = {}
        if self.attack is None:
            self.attack = {}
        if self.defense is None:
            self.defense = {}
        if self.models is None:
            self.models = {}
        if self.evaluation is None:
            self.evaluation = {}
        if self.output is None:
            self.output = {}
        if self.hardware is None:
            self.hardware = {}
        if self.debug is None:
            self.debug = {}

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
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
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info(f"成功加载配置文件: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"YAML解析错误: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {e}")
        raise
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

def save_config(config: Dict[str, Any], 
               save_path: Union[str, Path],
               format: str = 'yaml') -> bool:
    """保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
        format: 保存格式 ('yaml' 或 'json')
        
    Returns:
        是否保存成功
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml':
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            elif format.lower() == 'json':
                json.dump(config, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的保存格式: {format}")
        
        logger.info(f"配置文件已保存: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        return False

def merge_configs(*configs: Dict[str, Any], 
                 strategy: str = 'deep') -> Dict[str, Any]:
    """合并多个配置字典
    
    Args:
        *configs: 要合并的配置字典
        strategy: 合并策略 ('shallow' 或 'deep')
        
    Returns:
        合并后的配置字典
    """
    if not configs:
        return {}
    
    if len(configs) == 1:
        return copy.deepcopy(configs[0])
    
    if strategy == 'shallow':
        # 浅合并：后面的配置覆盖前面的
        merged = {}
        for config in configs:
            merged.update(config)
        return merged
    
    elif strategy == 'deep':
        # 深合并：递归合并嵌套字典
        merged = copy.deepcopy(configs[0])
        
        for config in configs[1:]:
            merged = _deep_merge(merged, config)
        
        return merged
    
    else:
        raise ValueError(f"不支持的合并策略: {strategy}")

def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并两个字典"""
    result = copy.deepcopy(dict1)
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result

def validate_config(config: Dict[str, Any], 
                   schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """验证配置文件
    
    Args:
        config: 配置字典
        schema: 验证模式（可选）
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 基本验证
    required_fields = ['experiment_name', 'dataset', 'attack', 'defense']
    
    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必需字段: {field}")
    
    # 数据集验证
    if 'dataset' in config:
        dataset_config = config['dataset']
        if 'name' not in dataset_config:
            errors.append("数据集配置缺少 'name' 字段")
        
        valid_datasets = ['coco', 'flickr30k', 'cc3m', 'visual_genome']
        if dataset_config.get('name') not in valid_datasets:
            errors.append(f"不支持的数据集: {dataset_config.get('name')}")
    
    # 攻击验证
    if 'attack' in config:
        attack_config = config['attack']
        if 'method' not in attack_config:
            errors.append("攻击配置缺少 'method' 字段")
        
        valid_attacks = ['pgd', 'hubness', 'fsta', 'sma']
        if attack_config.get('method') not in valid_attacks:
            errors.append(f"不支持的攻击方法: {attack_config.get('method')}")
    
    # 防御验证
    if 'defense' in config:
        defense_config = config['defense']
        
        # 检查防御组件
        components = ['text_variants', 'retrieval_reference', 
                     'generative_reference', 'consistency_detection']
        
        for component in components:
            if component in defense_config:
                comp_config = defense_config[component]
                if not isinstance(comp_config, dict):
                    errors.append(f"防御组件 '{component}' 配置应为字典")
                elif 'enabled' not in comp_config:
                    errors.append(f"防御组件 '{component}' 缺少 'enabled' 字段")
    
    # 模型验证
    if 'models' in config:
        models_config = config['models']
        
        required_models = ['clip', 'qwen', 'stable_diffusion']
        for model in required_models:
            if model not in models_config:
                errors.append(f"缺少模型配置: {model}")
            else:
                model_config = models_config[model]
                if 'model_name' not in model_config:
                    errors.append(f"模型 '{model}' 缺少 'model_name' 字段")
    
    # 硬件验证
    if 'hardware' in config:
        hardware_config = config['hardware']
        
        if 'device' in hardware_config:
            device = hardware_config['device']
            if device not in ['cpu', 'cuda', 'auto']:
                errors.append(f"不支持的设备类型: {device}")
        
        if 'gpu_ids' in hardware_config:
            gpu_ids = hardware_config['gpu_ids']
            if not isinstance(gpu_ids, list):
                errors.append("gpu_ids 应为列表")
            elif not all(isinstance(gpu_id, int) for gpu_id in gpu_ids):
                errors.append("gpu_ids 中的所有元素应为整数")
    
    # 自定义模式验证
    if schema:
        schema_errors = _validate_with_schema(config, schema)
        errors.extend(schema_errors)
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info("配置验证通过")
    else:
        logger.warning(f"配置验证失败，发现 {len(errors)} 个错误")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return is_valid, errors

def _validate_with_schema(config: Dict[str, Any], 
                         schema: Dict[str, Any]) -> List[str]:
    """使用自定义模式验证配置"""
    errors = []
    
    # 这里可以实现更复杂的模式验证逻辑
    # 目前只做简单的类型检查
    
    for key, expected_type in schema.items():
        if key in config:
            actual_value = config[key]
            
            if isinstance(expected_type, type):
                if not isinstance(actual_value, expected_type):
                    errors.append(f"字段 '{key}' 类型错误，期望 {expected_type.__name__}，实际 {type(actual_value).__name__}")
            
            elif isinstance(expected_type, dict):
                if isinstance(actual_value, dict):
                    nested_errors = _validate_with_schema(actual_value, expected_type)
                    errors.extend([f"{key}.{error}" for error in nested_errors])
                else:
                    errors.append(f"字段 '{key}' 应为字典类型")
    
    return errors

def create_default_config(experiment_name: str, 
                         dataset_name: str,
                         attack_method: str) -> Dict[str, Any]:
    """创建默认配置
    
    Args:
        experiment_name: 实验名称
        dataset_name: 数据集名称
        attack_method: 攻击方法
        
    Returns:
        默认配置字典
    """
    config = {
        'experiment_name': experiment_name,
        'description': f'{dataset_name}数据集上的{attack_method}攻击防御实验',
        'version': '1.0',
        
        'dataset': {
            'name': dataset_name,
            'split': 'test',
            'max_samples': 1000,
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 4
        },
        
        'attack': {
            'method': attack_method,
            'enabled': True,
            'parameters': _get_default_attack_params(attack_method)
        },
        
        'defense': {
            'text_variants': {
                'enabled': True,
                'count': 5,
                'strategies': ['synonym', 'paraphrase', 'reorder'],
                'quality_threshold': 0.8
            },
            'retrieval_reference': {
                'enabled': True,
                'count': 5,
                'similarity_threshold': 0.3,
                'database_path': 'data/reference_db'
            },
            'generative_reference': {
                'enabled': True,
                'count': 3,
                'guidance_scale': 7.5,
                'num_inference_steps': 20
            },
            'consistency_detection': {
                'enabled': True,
                'voting_strategy': 'weighted',
                'weights': {
                    'text_variants': 0.3,
                    'retrieval_reference': 0.4,
                    'generative_reference': 0.3
                },
                'threshold': 0.5,
                'adaptive_threshold': True
            }
        },
        
        'models': {
            'clip': {
                'model_name': 'ViT-B/32',
                'device': 'cuda',
                'batch_size': 64
            },
            'qwen': {
                'model_name': 'Qwen/Qwen2-VL-7B-Instruct',
                'device': 'cuda',
                'max_length': 512,
                'temperature': 0.7
            },
            'stable_diffusion': {
                'model_name': 'runwayml/stable-diffusion-v1-5',
                'device': 'cuda',
                'torch_dtype': 'float16'
            }
        },
        
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc'],
            'save_predictions': True,
            'save_scores': True,
            'compute_confidence_intervals': True
        },
        
        'output': {
            'base_dir': 'experiments/results',
            'save_models': False,
            'save_visualizations': True,
            'save_logs': True,
            'export_latex_tables': True
        },
        
        'hardware': {
            'device': 'cuda',
            'gpu_ids': [0],
            'mixed_precision': True,
            'pin_memory': True
        },
        
        'debug': {
            'enabled': False,
            'log_level': 'INFO',
            'save_intermediate_results': False,
            'profile_performance': False
        }
    }
    
    return config

def _get_default_attack_params(attack_method: str) -> Dict[str, Any]:
    """获取默认攻击参数"""
    if attack_method == 'pgd':
        return {
            'epsilon': 8/255,
            'alpha': 2/255,
            'num_iter': 10,
            'random_start': True,
            'targeted': False
        }
    elif attack_method == 'hubness':
        return {
            'target_hub_ratio': 0.1,
            'perturbation_strength': 0.1,
            'max_iterations': 50,
            'feature_space_attack': True,
            'input_space_attack': True
        }
    elif attack_method == 'fsta':
        return {
            'epsilon': 8/255,
            'alpha': 2/255,
            'num_iter': 10,
            'momentum': 0.9,
            'targeted': False
        }
    elif attack_method == 'sma':
        return {
            'epsilon': 8/255,
            'num_iter': 10,
            'momentum': 0.9,
            'variance': 0.1,
            'targeted': False
        }
    else:
        return {}

def load_config_with_overrides(base_config_path: Union[str, Path],
                              overrides: Optional[Dict[str, Any]] = None,
                              override_files: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
    """加载配置并应用覆盖
    
    Args:
        base_config_path: 基础配置文件路径
        overrides: 覆盖参数字典
        override_files: 覆盖配置文件列表
        
    Returns:
        最终配置字典
    """
    # 加载基础配置
    config = load_config(base_config_path)
    
    # 应用覆盖文件
    if override_files:
        override_configs = []
        for override_file in override_files:
            override_config = load_config(override_file)
            override_configs.append(override_config)
        
        # 合并所有配置
        all_configs = [config] + override_configs
        config = merge_configs(*all_configs, strategy='deep')
    
    # 应用覆盖参数
    if overrides:
        config = merge_configs(config, overrides, strategy='deep')
    
    return config

def export_config_template(save_path: Union[str, Path],
                          experiment_name: str = "template_experiment",
                          dataset_name: str = "coco",
                          attack_method: str = "pgd") -> bool:
    """导出配置模板
    
    Args:
        save_path: 保存路径
        experiment_name: 实验名称
        dataset_name: 数据集名称
        attack_method: 攻击方法
        
    Returns:
        是否导出成功
    """
    template_config = create_default_config(experiment_name, dataset_name, attack_method)
    
    # 添加注释
    template_config['_comments'] = {
        'experiment_name': '实验名称，用于标识实验',
        'dataset': '数据集配置，支持 coco, flickr30k, cc3m, visual_genome',
        'attack': '攻击配置，支持 pgd, hubness, fsta, sma',
        'defense': '防御配置，包含四个主要组件',
        'models': '模型配置，包含 CLIP, Qwen, Stable Diffusion',
        'evaluation': '评估配置，指定计算的指标',
        'output': '输出配置，指定结果保存位置和格式',
        'hardware': '硬件配置，指定设备和GPU设置',
        'debug': '调试配置，用于开发和调试'
    }
    
    return save_config(template_config, save_path, format='yaml')

def get_config_summary(config: Dict[str, Any]) -> str:
    """获取配置摘要
    
    Args:
        config: 配置字典
        
    Returns:
        配置摘要字符串
    """
    summary_lines = []
    
    # 基本信息
    summary_lines.append(f"实验名称: {config.get('experiment_name', 'Unknown')}")
    summary_lines.append(f"描述: {config.get('description', 'No description')}")
    summary_lines.append(f"版本: {config.get('version', 'Unknown')}")
    
    # 数据集信息
    if 'dataset' in config:
        dataset = config['dataset']
        summary_lines.append(f"数据集: {dataset.get('name', 'Unknown')}")
        summary_lines.append(f"样本数: {dataset.get('max_samples', 'All')}")
    
    # 攻击信息
    if 'attack' in config:
        attack = config['attack']
        summary_lines.append(f"攻击方法: {attack.get('method', 'Unknown')}")
        summary_lines.append(f"攻击启用: {attack.get('enabled', False)}")
    
    # 防御信息
    if 'defense' in config:
        defense = config['defense']
        enabled_components = []
        for component in ['text_variants', 'retrieval_reference', 
                         'generative_reference', 'consistency_detection']:
            if component in defense and defense[component].get('enabled', False):
                enabled_components.append(component)
        summary_lines.append(f"启用的防御组件: {', '.join(enabled_components)}")
    
    # 硬件信息
    if 'hardware' in config:
        hardware = config['hardware']
        summary_lines.append(f"设备: {hardware.get('device', 'Unknown')}")
        summary_lines.append(f"GPU IDs: {hardware.get('gpu_ids', [])}")
    
    return '\n'.join(summary_lines)