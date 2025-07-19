"""工具模块包

提供配置管理、数据加载、指标计算、可视化等工具功能。
"""

from .config import (
    ModelConfig, DataConfig, ExperimentConfig, AttackConfig, DefenseConfig, EvaluationConfig,
    ConfigManager, config_manager,
    get_config, load_config, save_config, update_config, validate_config
)

from .data_loader import (
    DatasetInfo, ImageTextDataset, COCODataLoader, Flickr30kDataLoader,
    DataLoaderManager, create_data_loader_manager, collate_image_text_batch
)

from .metrics import (
    MetricResult, RetrievalMetrics, DetectionMetrics,
    RetrievalEvaluator, DetectionEvaluator, SimilarityMetrics,
    MetricsCalculator, create_metrics_calculator
)

from .visualization import (
    PlotConfig, MetricsVisualizer, InteractiveVisualizer,
    ExperimentVisualizer, create_experiment_visualizer
)

__all__ = [
    # 配置管理
    'ModelConfig', 'DataConfig', 'ExperimentConfig', 'AttackConfig', 'DefenseConfig', 'EvaluationConfig',
    'ConfigManager', 'config_manager',
    'get_config', 'load_config', 'save_config', 'update_config', 'validate_config',
    
    # 数据加载
    'DatasetInfo', 'ImageTextDataset', 'COCODataLoader', 'Flickr30kDataLoader',
    'DataLoaderManager', 'create_data_loader_manager', 'collate_image_text_batch',
    
    # 指标计算
    'MetricResult', 'RetrievalMetrics', 'DetectionMetrics',
    'RetrievalEvaluator', 'DetectionEvaluator', 'SimilarityMetrics',
    'MetricsCalculator', 'create_metrics_calculator',
    
    # 可视化
    'PlotConfig', 'MetricsVisualizer', 'InteractiveVisualizer',
    'ExperimentVisualizer', 'create_experiment_visualizer'
]