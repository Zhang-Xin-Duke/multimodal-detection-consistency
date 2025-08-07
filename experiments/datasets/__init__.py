"""数据集加载模块

提供统一的数据集加载接口，支持多种多模态数据集的加载和预处理。
"""

from .coco_loader import COCOLoader
from .flickr_loader import Flickr30KLoader
from .vg_loader import VisualGenomeLoader
from .cc_loader import ConceptualCaptionsLoader, CC3MLoader, CC12MLoader
from .base_loader import BaseDatasetLoader, DatasetConfig, MultiModalDataset

__all__ = [
    'COCOLoader',
    'Flickr30KLoader', 
    'VisualGenomeLoader',
    'ConceptualCaptionsLoader',
    'CC3MLoader',
    'CC12MLoader',
    'BaseDatasetLoader',
    'DatasetConfig',
    'MultiModalDataset',
    'get_dataset_loader',
    'list_available_datasets'
]

def get_dataset_loader(dataset_name: str, **kwargs):
    """根据数据集名称获取对应的加载器
    
    Args:
        dataset_name: 数据集名称 ('coco', 'flickr30k', 'visual_genome', 'cc3m', 'cc12m')
        **kwargs: 传递给加载器的参数
    
    Returns:
        对应的数据集加载器实例
    
    Raises:
        ValueError: 不支持的数据集名称
    """
    loaders = {
        'coco': COCOLoader,
        'flickr30k': Flickr30KLoader,
        'visual_genome': VisualGenomeLoader,
        'cc3m': CC3MLoader,
        'cc12m': CC12MLoader,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"不支持的数据集: {dataset_name}. 支持的数据集: {list(loaders.keys())}")
    
    return loaders[dataset_name](**kwargs)

def list_available_datasets():
    """列出所有可用的数据集
    
    Returns:
        可用数据集名称列表
    """
    return ['coco', 'flickr30k', 'visual_genome', 'cc3m', 'cc12m']