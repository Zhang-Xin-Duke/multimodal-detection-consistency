"""Flickr30K数据集加载器

提供Flickr30K数据集的加载、预处理和批量处理功能。
支持图像-文本对的加载，用于多模态检索实验。
"""

import os
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

class Flickr30KDataset(Dataset):
    """Flickr30K数据集类"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "test",
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 max_samples: Optional[int] = None):
        """
        初始化Flickr30K数据集
        
        Args:
            data_dir: Flickr30K数据集根目录
            split: 数据集分割 ('train', 'val', 'test')
            image_size: 图像尺寸
            normalize: 是否标准化图像
            max_samples: 最大样本数量，None表示使用全部样本
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples
        
        # 设置路径
        self.image_dir = self.data_dir / "flickr30k_images"
        self.caption_file = self.data_dir / "results_20130124.token"
        self.split_file = self.data_dir / f"{split}.txt"
        
        # 验证路径存在
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not self.caption_file.exists():
            raise FileNotFoundError(f"标题文件不存在: {self.caption_file}")
        
        # 加载数据
        self._load_data()
        
        # 设置图像变换
        self.transform = self._get_transform(normalize)
        
        logger.info(f"Flickr30K数据集加载完成: {len(self.samples)} 个样本")
    
    def _load_data(self):
        """加载Flickr30K数据"""
        logger.info(f"加载Flickr30K数据，分割: {self.split}")
        
        # 加载分割文件（如果存在）
        split_images = set()
        if self.split_file.exists():
            with open(self.split_file, 'r') as f:
                split_images = set(line.strip() for line in f)
        
        # 加载标题文件
        image_captions = {}
        with open(self.caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_caption_id = parts[0]  # 格式: image_name#caption_id
                    caption = parts[1]
                    
                    # 提取图像名称
                    image_name = image_caption_id.split('#')[0]
                    
                    # 如果指定了分割且图像不在分割中，跳过
                    if split_images and image_name not in split_images:
                        continue
                    
                    if image_name not in image_captions:
                        image_captions[image_name] = []
                    image_captions[image_name].append(caption)
        
        # 构建样本列表
        self.samples = []
        for image_name, captions in image_captions.items():
            image_path = self.image_dir / image_name
            if image_path.exists():
                # 为每个标题创建一个样本
                for i, caption in enumerate(captions):
                    self.samples.append({
                        'image_name': image_name,
                        'image_path': str(image_path),
                        'caption': caption.strip(),
                        'caption_id': i
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
            'image_name': sample['image_name'],
            'caption_id': sample['caption_id'],
            'image_path': sample['image_path']
        }
    
    def get_image_caption_pairs(self) -> List[Tuple[str, str]]:
        """获取所有图像路径和标题对"""
        return [(sample['image_path'], sample['caption']) for sample in self.samples]

class Flickr30KLoader:
    """Flickr30K数据集加载器"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "test",
                 batch_size: int = 32,
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 num_workers: int = 4,
                 shuffle: bool = False,
                 max_samples: Optional[int] = None):
        """
        初始化Flickr30K加载器
        
        Args:
            data_dir: Flickr30K数据集根目录
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
        self.dataset = Flickr30KDataset(
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
        image_names = [item['image_name'] for item in batch]
        caption_ids = [item['caption_id'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        
        return {
            'images': images,
            'captions': captions,
            'image_names': image_names,
            'caption_ids': caption_ids,
            'image_paths': image_paths
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        return len(self.dataloader)
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'name': 'Flickr30K',
            'split': self.split,
            'num_samples': len(self.dataset),
            'num_batches': len(self.dataloader),
            'batch_size': self.batch_size,
            'data_dir': self.data_dir
        }
    
    def get_sample_by_name(self, image_name: str) -> Optional[List[Dict]]:
        """根据图像名称获取所有相关样本"""
        samples = []
        for i, sample in enumerate(self.dataset.samples):
            if sample['image_name'] == image_name:
                samples.append(self.dataset[i])
        return samples if samples else None
    
    def get_random_samples(self, num_samples: int = 10) -> List[Dict]:
        """获取随机样本"""
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        return [self.dataset[idx] for idx in indices]
    
    def get_unique_images(self) -> List[str]:
        """获取所有唯一图像名称"""
        unique_images = set()
        for sample in self.dataset.samples:
            unique_images.add(sample['image_name'])
        return list(unique_images)

def create_flickr30k_loader(config: Dict) -> Flickr30KLoader:
    """根据配置创建Flickr30K加载器"""
    dataset_config = config.get('dataset', {})
    
    return Flickr30KLoader(
        data_dir=dataset_config.get('data_dir'),
        split=dataset_config.get('split', 'test'),
        batch_size=dataset_config.get('batch_size', 32),
        image_size=tuple(dataset_config.get('image_size', [224, 224])),
        normalize=dataset_config.get('normalize', True),
        num_workers=config.get('hardware', {}).get('num_workers', 4),
        shuffle=False,
        max_samples=dataset_config.get('num_samples')
    )