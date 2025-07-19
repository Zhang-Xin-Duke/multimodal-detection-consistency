"""数据加载工具模块

提供统一的数据加载、预处理和批处理功能。
"""

import os
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    root_path: str
    split: str
    num_samples: int
    image_dir: str
    annotation_file: str


class ImageTextDataset(Dataset):
    """图像-文本数据集"""
    
    def __init__(self, 
                 image_paths: List[str],
                 texts: List[str],
                 image_ids: Optional[List[str]] = None,
                 transform: Optional[transforms.Compose] = None):
        """
        初始化数据集
        
        Args:
            image_paths: 图像路径列表
            texts: 文本列表
            image_ids: 图像ID列表
            transform: 图像变换
        """
        assert len(image_paths) == len(texts), "图像和文本数量不匹配"
        
        self.image_paths = image_paths
        self.texts = texts
        self.image_ids = image_ids or [f"img_{i}" for i in range(len(image_paths))]
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            包含图像、文本和元数据的字典
        """
        try:
            # 加载图像
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'text': self.texts[idx],
                'image_id': self.image_ids[idx],
                'image_path': image_path,
                'index': idx
            }
            
        except Exception as e:
            logger.error(f"加载数据项失败 (idx={idx}): {e}")
            # 返回默认数据
            return {
                'image': torch.zeros(3, 224, 224),
                'text': "",
                'image_id': f"error_{idx}",
                'image_path': "",
                'index': idx
            }


class COCODataLoader:
    """COCO数据集加载器"""
    
    def __init__(self, root_path: str):
        """
        初始化COCO数据加载器
        
        Args:
            root_path: COCO数据集根目录
        """
        self.root_path = Path(root_path)
        self.image_dir = self.root_path / "images"
        self.annotation_dir = self.root_path / "annotations"
    
    def load_annotations(self, split: str = "train") -> Tuple[List[str], List[str], List[str]]:
        """
        加载COCO注释
        
        Args:
            split: 数据集分割 (train/val/test)
            
        Returns:
            (image_paths, captions, image_ids)
        """
        try:
            # 加载注释文件
            if split == "train":
                ann_file = self.annotation_dir / "captions_train2017.json"
                img_dir = self.image_dir / "train2017"
            elif split == "val":
                ann_file = self.annotation_dir / "captions_val2017.json"
                img_dir = self.image_dir / "val2017"
            else:
                raise ValueError(f"不支持的数据集分割: {split}")
            
            if not ann_file.exists():
                raise FileNotFoundError(f"注释文件不存在: {ann_file}")
            
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # 构建图像ID到文件名的映射
            id_to_filename = {img['id']: img['file_name'] for img in data['images']}
            
            # 提取图像路径和标题
            image_paths = []
            captions = []
            image_ids = []
            
            for ann in data['annotations']:
                image_id = ann['image_id']
                if image_id in id_to_filename:
                    image_path = img_dir / id_to_filename[image_id]
                    if image_path.exists():
                        image_paths.append(str(image_path))
                        captions.append(ann['caption'])
                        image_ids.append(str(image_id))
            
            logger.info(f"加载COCO {split}集: {len(image_paths)}个样本")
            return image_paths, captions, image_ids
            
        except Exception as e:
            logger.error(f"加载COCO注释失败: {e}")
            raise
    
    def create_dataset(self, split: str = "train", transform: Optional[transforms.Compose] = None) -> ImageTextDataset:
        """
        创建COCO数据集
        
        Args:
            split: 数据集分割
            transform: 图像变换
            
        Returns:
            ImageTextDataset实例
        """
        image_paths, captions, image_ids = self.load_annotations(split)
        return ImageTextDataset(image_paths, captions, image_ids, transform)


class Flickr30kDataLoader:
    """Flickr30k数据集加载器"""
    
    def __init__(self, root_path: str):
        """
        初始化Flickr30k数据加载器
        
        Args:
            root_path: Flickr30k数据集根目录
        """
        self.root_path = Path(root_path)
        self.image_dir = self.root_path / "flickr30k_images"
        self.annotation_file = self.root_path / "results_20130124.token"
    
    def load_annotations(self) -> Tuple[List[str], List[str], List[str]]:
        """
        加载Flickr30k注释
        
        Returns:
            (image_paths, captions, image_ids)
        """
        try:
            if not self.annotation_file.exists():
                raise FileNotFoundError(f"注释文件不存在: {self.annotation_file}")
            
            image_paths = []
            captions = []
            image_ids = []
            
            with open(self.annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        image_filename = parts[0].split('#')[0]
                        caption = parts[1]
                        
                        image_path = self.image_dir / image_filename
                        if image_path.exists():
                            image_paths.append(str(image_path))
                            captions.append(caption)
                            image_ids.append(image_filename.split('.')[0])
            
            logger.info(f"加载Flickr30k: {len(image_paths)}个样本")
            return image_paths, captions, image_ids
            
        except Exception as e:
            logger.error(f"加载Flickr30k注释失败: {e}")
            raise
    
    def create_dataset(self, transform: Optional[transforms.Compose] = None) -> ImageTextDataset:
        """
        创建Flickr30k数据集
        
        Args:
            transform: 图像变换
            
        Returns:
            ImageTextDataset实例
        """
        image_paths, captions, image_ids = self.load_annotations()
        return ImageTextDataset(image_paths, captions, image_ids, transform)


class DataLoaderManager:
    """数据加载管理器"""
    
    def __init__(self, config):
        """
        初始化数据加载管理器
        
        Args:
            config: 数据配置对象
        """
        self.config = config
        self.datasets = {}
        self.dataloaders = {}
        
        # 创建图像变换
        self.transform = self._create_transform()
    
    def _create_transform(self) -> transforms.Compose:
        """
        创建图像变换
        
        Returns:
            图像变换组合
        """
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def load_dataset(self, dataset_name: str, split: str = "train") -> ImageTextDataset:
        """
        加载指定数据集
        
        Args:
            dataset_name: 数据集名称 (coco/flickr30k)
            split: 数据集分割
            
        Returns:
            数据集实例
        """
        dataset_key = f"{dataset_name}_{split}"
        
        if dataset_key in self.datasets:
            return self.datasets[dataset_key]
        
        try:
            if dataset_name.lower() == "coco":
                loader = COCODataLoader(self.config.coco_root)
                dataset = loader.create_dataset(split, self.transform)
            elif dataset_name.lower() == "flickr30k":
                loader = Flickr30kDataLoader(self.config.flickr30k_root)
                dataset = loader.create_dataset(self.transform)
            else:
                raise ValueError(f"不支持的数据集: {dataset_name}")
            
            self.datasets[dataset_key] = dataset
            logger.info(f"成功加载数据集: {dataset_key}")
            return dataset
            
        except Exception as e:
            logger.error(f"加载数据集失败 {dataset_key}: {e}")
            raise
    
    def create_dataloader(self, dataset: ImageTextDataset, 
                         batch_size: Optional[int] = None,
                         shuffle: Optional[bool] = None,
                         num_workers: Optional[int] = None) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集实例
            batch_size: 批大小
            shuffle: 是否打乱
            num_workers: 工作进程数
            
        Returns:
            数据加载器
        """
        batch_size = batch_size or self.config.batch_size
        shuffle = shuffle if shuffle is not None else self.config.shuffle
        num_workers = num_workers or self.config.num_workers
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批处理函数
        
        Args:
            batch: 批数据
            
        Returns:
            批处理后的数据
        """
        images = torch.stack([item['image'] for item in batch])
        texts = [item['text'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        indices = torch.tensor([item['index'] for item in batch])
        
        return {
            'images': images,
            'texts': texts,
            'image_ids': image_ids,
            'image_paths': image_paths,
            'indices': indices
        }
    
    def save_processed_data(self, data: Any, filename: str):
        """
        保存预处理数据
        
        Args:
            data: 要保存的数据
            filename: 文件名
        """
        try:
            output_path = Path(self.config.processed_root) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if filename.endswith('.pkl'):
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)
            elif filename.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif filename.endswith('.npy'):
                np.save(output_path, data)
            else:
                raise ValueError(f"不支持的文件格式: {filename}")
            
            logger.info(f"数据已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            raise
    
    def load_processed_data(self, filename: str) -> Any:
        """
        加载预处理数据
        
        Args:
            filename: 文件名
            
        Returns:
            加载的数据
        """
        try:
            file_path = Path(self.config.processed_root) / filename
            
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            if filename.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            elif filename.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif filename.endswith('.npy'):
                data = np.load(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {filename}")
            
            logger.info(f"数据已加载: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def get_dataset_info(self, dataset_name: str, split: str = "train") -> DatasetInfo:
        """
        获取数据集信息
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            
        Returns:
            数据集信息
        """
        dataset = self.load_dataset(dataset_name, split)
        
        if dataset_name.lower() == "coco":
            root_path = self.config.coco_root
            image_dir = f"images/{split}2017"
            annotation_file = f"annotations/captions_{split}2017.json"
        elif dataset_name.lower() == "flickr30k":
            root_path = self.config.flickr30k_root
            image_dir = "flickr30k_images"
            annotation_file = "results_20130124.token"
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        return DatasetInfo(
            name=dataset_name,
            root_path=root_path,
            split=split,
            num_samples=len(dataset),
            image_dir=image_dir,
            annotation_file=annotation_file
        )


def create_data_loader_manager(config) -> DataLoaderManager:
    """
    创建数据加载管理器
    
    Args:
        config: 数据配置
        
    Returns:
        数据加载管理器实例
    """
    return DataLoaderManager(config)


def collate_image_text_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    图像-文本批处理函数
    
    Args:
        batch: 批数据
        
    Returns:
        批处理后的数据
    """
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'images': images,
        'texts': texts
    }