"""COCO数据集加载器

提供COCO数据集的加载、预处理和批量处理功能。
支持图像-文本对的加载，用于多模态检索实验。
"""

import os
import json
import random
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class COCODataset(Dataset):
    """COCO数据集类"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "val2017",
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 max_samples: Optional[int] = None):
        """
        初始化COCO数据集
        
        Args:
            data_dir: COCO数据集根目录
            split: 数据集分割 ('train2017', 'val2017', 'test2017')
            image_size: 图像尺寸
            normalize: 是否标准化图像
            max_samples: 最大样本数量，None表示使用全部样本
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples
        
        # 设置路径
        self.image_dir = self.data_dir / "images" / split
        self.annotation_file = self.data_dir / "annotations" / f"captions_{split}.json"
        
        # 验证路径存在
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_file}")
        
        # 加载标注
        self._load_annotations()
        
        # 设置图像变换
        self.transform = self._get_transform(normalize)
        
        logger.info(f"COCO数据集加载完成: {len(self.samples)} 个样本")
    
    def _load_annotations(self):
        """加载COCO标注文件"""
        logger.info(f"加载COCO标注文件: {self.annotation_file}")
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 构建图像ID到文件名的映射
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # 构建样本列表
        self.samples = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in id_to_filename:
                image_path = self.image_dir / id_to_filename[image_id]
                if image_path.exists():
                    self.samples.append({
                        'image_id': image_id,
                        'image_path': str(image_path),
                        'caption': ann['caption'].strip(),
                        'annotation_id': ann['id']
                    })
        
        # 限制样本数量
        if self.max_samples and len(self.samples) > self.max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:self.max_samples]
        
        logger.info(f"加载了 {len(self.samples)} 个图像-文本对")
    
    def _get_transform(self, normalize: bool):
        """获取图像变换"""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ]
        
        if normalize:
            # CLIP标准化参数
            transform_list.append(
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            logger.error(f"加载图像失败: {sample['image_path']}, 错误: {e}")
            # 返回黑色图像作为fallback
            image_tensor = torch.zeros(3, *self.image_size)
        
        return {
            'image': image_tensor,
            'caption': sample['caption'],
            'image_id': sample['image_id'],
            'annotation_id': sample['annotation_id'],
            'image_path': sample['image_path']
        }
    
    def get_image_caption_pairs(self) -> List[Tuple[str, str]]:
        """获取所有图像路径和标题对"""
        return [(sample['image_path'], sample['caption']) for sample in self.samples]

class COCOLoader:
    """COCO数据集加载器"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "val2017",
                 batch_size: int = 32,
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 num_workers: int = 4,
                 shuffle: bool = False,
                 max_samples: Optional[int] = None):
        """
        初始化COCO加载器
        
        Args:
            data_dir: COCO数据集根目录
            split: 数据集分割
            batch_size: 批次大小
            image_size: 图像尺寸
            normalize: 是否标准化
            num_workers: 数据加载进程数
            shuffle: 是否打乱数据
            max_samples: 最大样本数量
        """
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # 创建数据集
        self.dataset = COCODataset(
            data_dir=data_dir,
            split=split,
            image_size=image_size,
            normalize=normalize,
            max_samples=max_samples
        )
        
        # 创建数据加载器
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """批次整理函数"""
        images = torch.stack([item['image'] for item in batch])
        captions = [item['caption'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        annotation_ids = [item['annotation_id'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        
        return {
            'images': images,
            'captions': captions,
            'image_ids': image_ids,
            'annotation_ids': annotation_ids,
            'image_paths': image_paths
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        return len(self.dataloader)
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'name': 'COCO',
            'split': self.split,
            'num_samples': len(self.dataset),
            'num_batches': len(self.dataloader),
            'batch_size': self.batch_size,
            'data_dir': self.data_dir
        }
    
    def get_sample_by_id(self, image_id: int) -> Optional[Dict]:
        """根据图像ID获取样本"""
        for sample in self.dataset.samples:
            if sample['image_id'] == image_id:
                idx = self.dataset.samples.index(sample)
                return self.dataset[idx]
        return None
    
    def get_random_samples(self, num_samples: int = 10) -> List[Dict]:
        """获取随机样本"""
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        return [self.dataset[idx] for idx in indices]

def create_coco_loader(config: Dict) -> COCOLoader:
    """根据配置创建COCO加载器"""
    dataset_config = config.get('dataset', {})
    
    return COCOLoader(
        data_dir=dataset_config.get('data_dir'),
        split=dataset_config.get('split', 'val2017'),
        batch_size=dataset_config.get('batch_size', 32),
        image_size=tuple(dataset_config.get('image_size', [224, 224])),
        normalize=dataset_config.get('normalize', True),
        num_workers=config.get('hardware', {}).get('num_workers', 4),
        shuffle=False,  # 实验中通常不打乱
        max_samples=dataset_config.get('num_samples')
    )