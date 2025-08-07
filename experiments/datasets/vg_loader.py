"""Visual Genome 数据集加载器

支持加载Visual Genome数据集的图像和区域描述
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .base_loader import BaseDatasetLoader, DatasetConfig

logger = logging.getLogger(__name__)

class VisualGenomeLoader(BaseDatasetLoader):
    """Visual Genome数据集加载器"""
    
    def __init__(self, config: DatasetConfig):
        """初始化VG加载器
        
        Args:
            config: 数据集配置
        """
        super().__init__(config)
        
        # VG特定配置
        self.region_descriptions_file = self.data_root / "region_descriptions.json"
        self.image_data_file = self.data_root / "image_data.json"
        self.images_dir = self.data_root / "images"
        
        # 验证文件存在性
        if not self.region_descriptions_file.exists():
            raise FileNotFoundError(f"区域描述文件不存在: {self.region_descriptions_file}")
        
        if not self.image_data_file.exists():
            raise FileNotFoundError(f"图像数据文件不存在: {self.image_data_file}")
        
        if not self.images_dir.exists():
            logger.warning(f"图像目录不存在: {self.images_dir}")
        
        # 加载图像元数据
        self.image_metadata = self._load_image_metadata()
        
        # 分割配置
        self.split_config = self._get_split_config()
    
    def _load_image_metadata(self) -> Dict[int, Dict[str, Any]]:
        """加载图像元数据
        
        Returns:
            图像ID到元数据的映射
        """
        logger.info(f"加载图像元数据: {self.image_data_file}")
        
        try:
            with open(self.image_data_file, 'r', encoding='utf-8') as f:
                image_data = json.load(f)
            
            # 转换为字典格式，以image_id为键
            metadata_dict = {}
            for img_info in image_data:
                image_id = img_info['image_id']
                metadata_dict[image_id] = img_info
            
            logger.info(f"加载了 {len(metadata_dict)} 个图像的元数据")
            return metadata_dict
            
        except Exception as e:
            logger.error(f"加载图像元数据失败: {e}")
            return {}
    
    def _get_split_config(self) -> Dict[str, List[int]]:
        """获取数据集分割配置
        
        Returns:
            分割配置字典
        """
        # VG通常没有官方的train/val/test分割
        # 这里提供一个简单的分割策略
        all_image_ids = list(self.image_metadata.keys())
        all_image_ids.sort()  # 确保一致性
        
        total_images = len(all_image_ids)
        train_end = int(total_images * 0.8)
        val_end = int(total_images * 0.9)
        
        split_config = {
            'train': all_image_ids[:train_end],
            'val': all_image_ids[train_end:val_end],
            'test': all_image_ids[val_end:]
        }
        
        logger.info(f"数据集分割: train={len(split_config['train'])}, "
                   f"val={len(split_config['val'])}, test={len(split_config['test'])}")
        
        return split_config
    
    def load_annotations(self) -> List[Dict[str, Any]]:
        """加载VG标注
        
        Returns:
            标注列表
        """
        logger.info(f"加载VG区域描述: {self.region_descriptions_file}")
        
        try:
            with open(self.region_descriptions_file, 'r', encoding='utf-8') as f:
                region_data = json.load(f)
            
            logger.info(f"读取到 {len(region_data)} 个图像的区域描述")
            
        except Exception as e:
            logger.error(f"加载区域描述失败: {e}")
            raise
        
        # 获取当前分割的图像ID
        if self.config.split in self.split_config:
            valid_image_ids = set(self.split_config[self.config.split])
        else:
            logger.warning(f"未知的数据集分割: {self.config.split}，使用所有数据")
            valid_image_ids = set(self.image_metadata.keys())
        
        annotations = []
        
        for img_regions in region_data:
            image_id = img_regions['id']
            
            # 检查是否在当前分割中
            if image_id not in valid_image_ids:
                continue
            
            # 检查图像元数据是否存在
            if image_id not in self.image_metadata:
                logger.warning(f"图像 {image_id} 的元数据不存在")
                continue
            
            # 处理区域描述
            regions = img_regions.get('regions', [])
            
            for region_idx, region in enumerate(regions):
                try:
                    # 提取区域描述
                    phrase = region.get('phrase', '').strip()
                    
                    if not phrase or len(phrase) < 3:
                        continue
                    
                    # 获取区域坐标（可选）
                    x = region.get('x', 0)
                    y = region.get('y', 0)
                    width = region.get('width', 0)
                    height = region.get('height', 0)
                    
                    annotation = {
                        'id': f"{image_id}_{region_idx}",
                        'image_id': image_id,
                        'region_id': region.get('region_id', region_idx),
                        'phrase': phrase,
                        'bbox': [x, y, width, height],
                        'split': self.config.split,
                        'image_metadata': self.image_metadata[image_id]
                    }
                    
                    annotations.append(annotation)
                    
                except Exception as e:
                    logger.warning(f"处理图像 {image_id} 区域 {region_idx} 时出错: {e}")
                    continue
        
        logger.info(f"成功加载 {len(annotations)} 个有效标注")
        return annotations
    
    def get_image_path(self, annotation: Dict[str, Any]) -> str:
        """获取图像路径
        
        Args:
            annotation: 标注信息
            
        Returns:
            相对于数据根目录的图像路径
        """
        image_id = annotation['image_id']
        
        # 从图像元数据中获取文件名
        if image_id in self.image_metadata:
            image_info = self.image_metadata[image_id]
            
            # 尝试不同的文件名字段
            filename = None
            for field in ['url', 'filename', 'image_url']:
                if field in image_info:
                    url_or_filename = image_info[field]
                    if isinstance(url_or_filename, str):
                        # 如果是URL，提取文件名
                        if url_or_filename.startswith('http'):
                            filename = os.path.basename(url_or_filename)
                        else:
                            filename = url_or_filename
                        break
            
            if not filename:
                # 使用默认文件名格式
                filename = f"{image_id}.jpg"
        else:
            filename = f"{image_id}.jpg"
        
        # 尝试不同的子目录结构
        possible_paths = [
            f"images/{filename}",
            f"VG_100K/{filename}",
            f"VG_100K_2/{filename}",
            filename
        ]
        
        for path in possible_paths:
            full_path = self.data_root / path
            if full_path.exists():
                return path
        
        # 如果都不存在，返回默认路径
        return f"images/{filename}"
    
    def get_captions(self, annotation: Dict[str, Any]) -> List[str]:
        """获取图像描述
        
        Args:
            annotation: 标注信息
            
        Returns:
            描述列表
        """
        phrase = annotation['phrase']
        
        # 清理和验证描述
        if not phrase or len(phrase.strip()) < 3:
            return []
        
        # 基本清理
        phrase = phrase.strip()
        
        # 移除可能的特殊字符
        import re
        phrase = re.sub(r'[\n\r\t]+', ' ', phrase)
        phrase = re.sub(r'\s+', ' ', phrase)
        
        return [phrase]
    
    def get_region_info(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """获取区域信息
        
        Args:
            annotation: 标注信息
            
        Returns:
            区域信息字典
        """
        return {
            'region_id': annotation.get('region_id'),
            'bbox': annotation.get('bbox', [0, 0, 0, 0]),
            'phrase': annotation.get('phrase', ''),
            'image_id': annotation.get('image_id')
        }
    
    def get_samples_by_image_id(self, image_id: int) -> List[Dict[str, Any]]:
        """根据图像ID获取所有区域描述
        
        Args:
            image_id: 图像ID
            
        Returns:
            该图像的所有区域描述样本
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
        
        # 基础统计
        base_stats = super().get_statistics()
        
        # VG特定统计
        unique_images = len(set(sample['image_id'] for sample in self.samples))
        
        # 区域大小统计
        bbox_areas = []
        for sample in self.samples:
            bbox = sample['metadata'].get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                area = bbox[2] * bbox[3]  # width * height
                if area > 0:
                    bbox_areas.append(area)
        
        import numpy as np
        
        vg_stats = {
            'unique_images': unique_images,
            'regions_per_image': len(self.samples) / unique_images if unique_images > 0 else 0,
            'bbox_area_stats': {
                'mean': np.mean(bbox_areas) if bbox_areas else 0,
                'std': np.std(bbox_areas) if bbox_areas else 0,
                'min': np.min(bbox_areas) if bbox_areas else 0,
                'max': np.max(bbox_areas) if bbox_areas else 0
            } if bbox_areas else None
        }
        
        # 合并统计信息
        base_stats.update(vg_stats)
        
        return base_stats

def create_vg_loader(data_root: str, 
                    split: str = 'test',
                    **kwargs) -> VisualGenomeLoader:
    """创建Visual Genome加载器的便捷函数
    
    Args:
        data_root: 数据根目录
        split: 数据集分割
        **kwargs: 其他配置参数
        
    Returns:
        VG数据集加载器实例
    """
    config = DatasetConfig(
        name='visual_genome',
        data_root=data_root,
        split=split,
        **kwargs
    )
    
    return VisualGenomeLoader(config)