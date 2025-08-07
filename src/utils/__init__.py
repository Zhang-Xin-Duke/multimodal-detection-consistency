"""工具模块

提供配置管理、硬件检测、多GPU处理、CUDA工具、评估指标和可视化等功能。
"""

from .config_manager import ConfigManager
from .hardware_detector import HardwareDetector
from .multi_gpu_processor import MultiGPUProcessor, GPUTask, GPUResult
from .cuda_utils import (
    CUDADeviceManager, 
    GPUMonitor, 
    CUDAErrorHandler,
    get_device_manager,
    check_cuda_available,
    estimate_model_memory,
    optimize_batch_size
)
from .metrics import (
    SimilarityCalculator,
    DetectionEvaluator, 
    RetrievalEvaluator,
    MetricsAggregator,
    DetectionMetrics,
    RetrievalMetrics,
    SimilarityMetrics
)
from .visualization import (
    VisualizationManager,
    ROCVisualizer,
    PRVisualizer,
    DistributionVisualizer,
    DimensionalityVisualizer,
    ConfusionMatrixVisualizer,
    MetricsVisualizer,
    create_visualization_manager
)
from .seed import (
    set_random_seed,
    get_random_seed,
    create_reproducible_generator,
    SeedContext,
    DEFAULT_SEED
)

__all__ = [
    # 配置管理
    'ConfigManager',
    
    # 硬件检测
    'HardwareDetector',
    
    # 多GPU处理
    'MultiGPUProcessor',
    'GPUTask',
    'GPUResult',
    
    # CUDA工具
    'CUDADeviceManager',
    'GPUMonitor',
    'CUDAErrorHandler',
    'get_device_manager',
    'check_cuda_available',
    'estimate_model_memory',
    'optimize_batch_size',
    
    # 评估指标
    'SimilarityCalculator',
    'DetectionEvaluator',
    'RetrievalEvaluator',
    'MetricsAggregator',
    'DetectionMetrics',
    'RetrievalMetrics',
    'SimilarityMetrics',
    
    # 可视化
    'VisualizationManager',
    'ROCVisualizer',
    'PRVisualizer',
    'DistributionVisualizer',
    'DimensionalityVisualizer',
    'ConfusionMatrixVisualizer',
    'MetricsVisualizer',
    'create_visualization_manager',
    
    # 随机种子
    'set_random_seed',
    'get_random_seed',
    'create_reproducible_generator',
    'SeedContext',
    'DEFAULT_SEED'
]