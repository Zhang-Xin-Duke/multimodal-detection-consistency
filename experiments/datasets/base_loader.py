"""基础数据集加载器

提供所有数据集加载器的基础类和通用功能
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    data_root: str
    split: str = 'test'
    max_samples: Optional[int] = None
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = False
    pin_memory: bool = True
    image_size: Tuple[int, int] = (224, 224)
    normalize: bool = True
    cache_features: bool = False
    cache_dir: Optional[str] = None
    
class BaseDatasetLoader(ABC):
    """基础数据集加载器"""
    
    def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
        """初始化数据集加载器
        
        Args:
            config: 数据集配置
        """
        if isinstance(config, dict):
            self.config = DatasetConfig(**config)
        else:
            self.config = config
            
        self.data_root = Path(self.config.data_root)
        self.cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None
        
        # 验证数据路径
        if not self.data_root.exists():
            raise FileNotFoundError(f"数据根目录不存在: {self.data_root}")
        
        # 创建缓存目录
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据
        self.samples = []
        self.image_paths = []
        self.captions = []
        self.metadata = {}
        
        logger.info(f"初始化 {self.__class__.__name__} 数据集加载器")
        logger.info(f"数据根目录: {self.data_root}")
        logger.info(f"数据集分割: {self.config.split}")
        logger.info(f"最大样本数: {self.config.max_samples}")
    
    @abstractmethod
    def load_annotations(self) -> List[Dict[str, Any]]:
        """加载数据集标注
        
        Returns:
            标注列表，每个元素包含图像路径和对应的文本描述
        """
        pass
    
    @abstractmethod
    def get_image_path(self, annotation: Dict[str, Any]) -> str:
        """获取图像路径
        
        Args:
            annotation: 标注信息
            
        Returns:
            图像文件路径
        """
        pass
    
    @abstractmethod
    def get_captions(self, annotation: Dict[str, Any]) -> List[str]:
        """获取图像描述
        
        Args:
            annotation: 标注信息
            
        Returns:
            图像描述列表
        """
        pass
    
    def load_data(self) -> None:
        """加载数据集"""
        logger.info(f"开始加载 {self.config.name} 数据集...")
        
        # 加载标注
        annotations = self.load_annotations()
        logger.info(f"加载了 {len(annotations)} 个标注")
        
        # 处理样本
        valid_samples = 0
        for i, annotation in enumerate(annotations):
            if self.config.max_samples and valid_samples >= self.config.max_samples:
                break
                
            try:
                # 获取图像路径
                image_path = self.get_image_path(annotation)
                full_image_path = self.data_root / image_path
                
                # 检查图像文件是否存在
                if not full_image_path.exists():
                    logger.warning(f"图像文件不存在: {full_image_path}")
                    continue
                
                # 获取描述
                captions = self.get_captions(annotation)
                if not captions:
                    logger.warning(f"样本 {i} 没有有效的描述")
                    continue
                
                # 添加样本
                for caption in captions:
                    if caption.strip():  # 确保描述不为空
                        self.samples.append({
                            'image_path': str(full_image_path),
                            'caption': caption.strip(),
                            'image_id': annotation.get('image_id', i),
                            'annotation_id': annotation.get('id', f"{i}_{len(self.samples)}"),
                            'metadata': annotation
                        })
                        valid_samples += 1
                        
                        if self.config.max_samples and valid_samples >= self.config.max_samples:
                            break
                            
            except Exception as e:
                logger.warning(f"处理样本 {i} 时出错: {e}")
                continue
        
        logger.info(f"成功加载 {len(self.samples)} 个有效样本")
        
        # 提取图像路径和描述
        self.image_paths = [sample['image_path'] for sample in self.samples]
        self.captions = [sample['caption'] for sample in self.samples]
        
        # 保存元数据
        self.metadata = {
            'dataset_name': self.config.name,
            'split': self.config.split,
            'total_samples': len(self.samples),
            'data_root': str(self.data_root),
            'config': self.config.__dict__
        }
    
    def get_dataset(self) -> 'MultiModalDataset':
        """获取PyTorch数据集
        
        Returns:
            MultiModalDataset实例
        """
        if not self.samples:
            self.load_data()
        
        return MultiModalDataset(
            samples=self.samples,
            image_size=self.config.image_size,
            normalize=self.config.normalize
        )
    
    def get_dataloader(self, **kwargs) -> DataLoader:
        """获取PyTorch数据加载器
        
        Args:
            **kwargs: DataLoader的额外参数
            
        Returns:
            DataLoader实例
        """
        dataset = self.get_dataset()
        
        # 合并配置和传入的参数
        dataloader_kwargs = {
            'batch_size': self.config.batch_size,
            'shuffle': self.config.shuffle,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory,
            'collate_fn': self._collate_fn
        }
        dataloader_kwargs.update(kwargs)
        
        return DataLoader(dataset, **dataloader_kwargs)
    
    def _collate_fn(self, batch):
        """自定义批处理函数"""
        images = []
        captions = []
        image_paths = []
        metadata = []
        
        for item in batch:
            images.append(item['image'])
            captions.append(item['caption'])
            image_paths.append(item['image_path'])
            metadata.append(item['metadata'])
        
        return {
            'images': torch.stack(images),
            'captions': captions,
            'image_paths': image_paths,
            'metadata': metadata
        }
    
    def get_sample_by_index(self, index: int) -> Dict[str, Any]:
        """根据索引获取样本
        
        Args:
            index: 样本索引
            
        Returns:
            样本信息
        """
        if not self.samples:
            self.load_data()
        
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"索引超出范围: {index}")
        
        return self.samples[index]
    
    def get_samples_by_image_id(self, image_id: Union[int, str]) -> List[Dict[str, Any]]:
        """根据图像ID获取所有相关样本
        
        Args:
            image_id: 图像ID
            
        Returns:
            相关样本列表
        """
        if not self.samples:
            self.load_data()
        
        return [sample for sample in self.samples if sample['image_id'] == image_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        if not self.samples:
            self.load_data()
        
        # 计算描述长度统计
        caption_lengths = [len(sample['caption'].split()) for sample in self.samples]
        
        # 计算唯一图像数量
        unique_images = len(set(sample['image_path'] for sample in self.samples))
        
        # 计算每张图像的平均描述数
        image_caption_counts = {}
        for sample in self.samples:
            image_path = sample['image_path']
            image_caption_counts[image_path] = image_caption_counts.get(image_path, 0) + 1
        
        avg_captions_per_image = np.mean(list(image_caption_counts.values()))
        
        return {
            'total_samples': len(self.samples),
            'unique_images': unique_images,
            'avg_captions_per_image': avg_captions_per_image,
            'caption_length_stats': {
                'mean': np.mean(caption_lengths),
                'std': np.std(caption_lengths),
                'min': np.min(caption_lengths),
                'max': np.max(caption_lengths),
                'median': np.median(caption_lengths)
            },
            'dataset_name': self.config.name,
            'split': self.config.split
        }
    
    def save_cache(self, cache_path: Optional[str] = None) -> bool:
        """保存缓存
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            是否保存成功
        """
        if not self.config.cache_features:
            return False
        
        if cache_path is None:
            if self.cache_dir is None:
                return False
            cache_path = self.cache_dir / f"{self.config.name}_{self.config.split}_cache.json"
        
        try:
            cache_data = {
                'samples': self.samples,
                'metadata': self.metadata,
                'config': self.config.__dict__
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"缓存已保存: {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
            return False
    
    def load_cache(self, cache_path: Optional[str] = None) -> bool:
        """加载缓存
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            是否加载成功
        """
        if cache_path is None:
            if self.cache_dir is None:
                return False
            cache_path = self.cache_dir / f"{self.config.name}_{self.config.split}_cache.json"
        
        if not Path(cache_path).exists():
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.samples = cache_data['samples']
            self.metadata = cache_data['metadata']
            
            # 更新图像路径和描述列表
            self.image_paths = [sample['image_path'] for sample in self.samples]
            self.captions = [sample['caption'] for sample in self.samples]
            
            logger.info(f"缓存已加载: {cache_path}")
            logger.info(f"加载了 {len(self.samples)} 个样本")
            return True
            
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return False
    
    def __len__(self) -> int:
        """获取数据集大小"""
        if not self.samples:
            self.load_data()
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取单个样本"""
        return self.get_sample_by_index(index)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(dataset={self.config.name}, split={self.config.split}, samples={len(self.samples)})"

class MultiModalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(self, 
                 samples: List[Dict[str, Any]],
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True):
        """初始化数据集
        
        Args:
            samples: 样本列表
            image_size: 图像尺寸
            normalize: 是否标准化
        """
        self.samples = samples
        self.image_size = image_size
        self.normalize = normalize
        
        # 设置图像变换
        from torchvision import transforms
        
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor()
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """获取数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取单个样本"""
        sample = self.samples[index]
        
        # 加载图像
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.error(f"加载图像失败 {sample['image_path']}: {e}")
            # 创建空白图像作为备用
            image = torch.zeros(3, *self.image_size)
        
        return {
            'image': image,
            'caption': sample['caption'],
            'image_path': sample['image_path'],
            'image_id': sample['image_id'],
            'annotation_id': sample['annotation_id'],
            'metadata': sample['metadata']
        }

def create_dataset_loader(dataset_name: str, 
                         data_root: str,
                         split: str = 'test',
                         **kwargs) -> BaseDatasetLoader:
    """创建数据集加载器的便捷函数
    
    Args:
        dataset_name: 数据集名称
        data_root: 数据根目录
        split: 数据集分割
        **kwargs: 其他配置参数
        
    Returns:
        数据集加载器实例
    """
    from . import get_dataset_loader
    
    config = DatasetConfig(
        name=dataset_name,
        data_root=data_root,
        split=split,
        **kwargs
    )
    
    return get_dataset_loader(dataset_name, config=config)