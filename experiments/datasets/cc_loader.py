"""Conceptual Captions (CC3M/CC12M) 数据集加载器

支持加载Conceptual Captions 3M和12M数据集
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

from .base_loader import BaseDatasetLoader, DatasetConfig

logger = logging.getLogger(__name__)

class CC3MLoader(BaseDatasetLoader):
    """Conceptual Captions 3M数据集加载器"""
    
    def __init__(self, config: DatasetConfig):
        """初始化CC3M加载器
        
        Args:
            config: 数据集配置
        """
        super().__init__(config)
        
        # CC3M特定配置
        self.annotation_file = self._get_annotation_file()
        self.image_dir = self.data_root / "images"
        
        # 验证文件存在性
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_file}")
        
        if not self.image_dir.exists():
            logger.warning(f"图像目录不存在: {self.image_dir}")
    
    def _get_annotation_file(self) -> Path:
        """获取标注文件路径"""
        # 支持多种文件格式和命名方式
        possible_files = [
            f"cc3m_{self.config.split}.tsv",
            f"cc3m_{self.config.split}.csv",
            f"conceptual_captions_{self.config.split}.tsv",
            f"conceptual_captions_{self.config.split}.csv",
            f"{self.config.split}.tsv",
            f"{self.config.split}.csv",
            "Train_GCC-training.tsv",  # 官方训练集文件名
            "Validation_GCC-1.1.0-Validation.tsv",  # 官方验证集文件名
        ]
        
        for filename in possible_files:
            file_path = self.data_root / filename
            if file_path.exists():
                return file_path
        
        # 如果都不存在，返回默认路径
        return self.data_root / f"cc3m_{self.config.split}.tsv"
    
    def load_annotations(self) -> List[Dict[str, Any]]:
        """加载CC3M标注
        
        Returns:
            标注列表
        """
        logger.info(f"加载CC3M标注文件: {self.annotation_file}")
        
        annotations = []
        
        try:
            # 根据文件扩展名选择读取方式
            if self.annotation_file.suffix.lower() == '.tsv':
                df = pd.read_csv(self.annotation_file, sep='\t', 
                                names=['caption', 'url'], header=None)
            elif self.annotation_file.suffix.lower() == '.csv':
                df = pd.read_csv(self.annotation_file)
            else:
                # 尝试自动检测分隔符
                df = pd.read_csv(self.annotation_file, sep=None, engine='python')
            
            logger.info(f"读取到 {len(df)} 条记录")
            
            # 处理数据
            for idx, row in df.iterrows():
                try:
                    # 提取描述和URL
                    if 'caption' in row:
                        caption = str(row['caption']).strip()
                    elif 'text' in row:
                        caption = str(row['text']).strip()
                    else:
                        # 假设第一列是描述
                        caption = str(row.iloc[0]).strip()
                    
                    if 'url' in row:
                        url = str(row['url']).strip()
                    elif 'image_url' in row:
                        url = str(row['image_url']).strip()
                    else:
                        # 假设第二列是URL
                        url = str(row.iloc[1]).strip()
                    
                    # 跳过无效记录
                    if not caption or caption == 'nan' or not url or url == 'nan':
                        continue
                    
                    # 生成图像文件名（通常基于URL或索引）
                    image_filename = self._generate_image_filename(url, idx)
                    
                    annotation = {
                        'id': idx,
                        'image_id': idx,
                        'caption': caption,
                        'url': url,
                        'image_filename': image_filename,
                        'split': self.config.split
                    }
                    
                    annotations.append(annotation)
                    
                except Exception as e:
                    logger.warning(f"处理第 {idx} 行时出错: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"读取标注文件失败: {e}")
            raise
        
        logger.info(f"成功加载 {len(annotations)} 个有效标注")
        return annotations
    
    def _generate_image_filename(self, url: str, index: int) -> str:
        """生成图像文件名
        
        Args:
            url: 图像URL
            index: 索引
            
        Returns:
            图像文件名
        """
        # 方法1: 基于URL生成文件名
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            # 如果URL中有有效的文件名
            if filename and '.' in filename:
                return filename
        except:
            pass
        
        # 方法2: 基于URL的哈希值
        try:
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()
            return f"{url_hash}.jpg"
        except:
            pass
        
        # 方法3: 基于索引
        return f"{index:08d}.jpg"
    
    def get_image_path(self, annotation: Dict[str, Any]) -> str:
        """获取图像路径
        
        Args:
            annotation: 标注信息
            
        Returns:
            相对于数据根目录的图像路径
        """
        image_filename = annotation['image_filename']
        
        # 尝试不同的子目录结构
        possible_paths = [
            f"images/{image_filename}",
            f"train/{image_filename}" if self.config.split == 'train' else f"val/{image_filename}",
            f"{self.config.split}/{image_filename}",
            image_filename
        ]
        
        for path in possible_paths:
            full_path = self.data_root / path
            if full_path.exists():
                return path
        
        # 如果都不存在，返回默认路径
        return f"images/{image_filename}"
    
    def get_captions(self, annotation: Dict[str, Any]) -> List[str]:
        """获取图像描述
        
        Args:
            annotation: 标注信息
            
        Returns:
            描述列表
        """
        caption = annotation['caption']
        
        # 清理和验证描述
        if not caption or len(caption.strip()) < 3:
            return []
        
        # 基本清理
        caption = caption.strip()
        
        # 移除可能的HTML标签
        import re
        caption = re.sub(r'<[^>]+>', '', caption)
        
        # 移除多余的空白字符
        caption = re.sub(r'\s+', ' ', caption)
        
        return [caption]

class CC12MLoader(CC3MLoader):
    """Conceptual Captions 12M数据集加载器
    
    继承自CC3MLoader，主要区别在于数据规模和可能的文件结构
    """
    
    def _get_annotation_file(self) -> Path:
        """获取CC12M标注文件路径"""
        possible_files = [
            f"cc12m_{self.config.split}.tsv",
            f"cc12m_{self.config.split}.csv",
            f"conceptual_captions_12m_{self.config.split}.tsv",
            f"conceptual_captions_12m_{self.config.split}.csv",
            f"{self.config.split}.tsv",
            f"{self.config.split}.csv",
            "cc12m_train.tsv",
            "cc12m_val.tsv",
        ]
        
        for filename in possible_files:
            file_path = self.data_root / filename
            if file_path.exists():
                return file_path
        
        return self.data_root / f"cc12m_{self.config.split}.tsv"

class ConceptualCaptionsLoader(CC3MLoader):
    """通用Conceptual Captions加载器
    
    自动检测是CC3M还是CC12M
    """
    
    def __init__(self, config: DatasetConfig):
        """初始化通用CC加载器
        
        Args:
            config: 数据集配置
        """
        # 检测数据集类型
        self.dataset_type = self._detect_dataset_type(config.data_root)
        logger.info(f"检测到数据集类型: {self.dataset_type}")
        
        super().__init__(config)
    
    def _detect_dataset_type(self, data_root: str) -> str:
        """检测数据集类型
        
        Args:
            data_root: 数据根目录
            
        Returns:
            数据集类型 ('cc3m' 或 'cc12m')
        """
        data_root = Path(data_root)
        
        # 检查文件名中的标识
        for file_path in data_root.glob("*.tsv"):
            filename = file_path.name.lower()
            if 'cc12m' in filename or '12m' in filename:
                return 'cc12m'
            elif 'cc3m' in filename or '3m' in filename:
                return 'cc3m'
        
        for file_path in data_root.glob("*.csv"):
            filename = file_path.name.lower()
            if 'cc12m' in filename or '12m' in filename:
                return 'cc12m'
            elif 'cc3m' in filename or '3m' in filename:
                return 'cc3m'
        
        # 检查目录结构或文件大小来推断
        annotation_files = list(data_root.glob("*.tsv")) + list(data_root.glob("*.csv"))
        if annotation_files:
            # 简单的启发式：文件大小大于100MB可能是CC12M
            largest_file = max(annotation_files, key=lambda x: x.stat().st_size)
            file_size_mb = largest_file.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 100:  # 100MB阈值
                return 'cc12m'
            else:
                return 'cc3m'
        
        # 默认返回CC3M
        return 'cc3m'
    
    def _get_annotation_file(self) -> Path:
        """获取标注文件路径"""
        if self.dataset_type == 'cc12m':
            return CC12MLoader._get_annotation_file(self)
        else:
            return super()._get_annotation_file()

def create_cc_loader(data_root: str, 
                    split: str = 'test',
                    dataset_type: str = 'auto',
                    **kwargs) -> BaseDatasetLoader:
    """创建Conceptual Captions加载器的便捷函数
    
    Args:
        data_root: 数据根目录
        split: 数据集分割
        dataset_type: 数据集类型 ('cc3m', 'cc12m', 'auto')
        **kwargs: 其他配置参数
        
    Returns:
        CC数据集加载器实例
    """
    config = DatasetConfig(
        name=f"cc_{dataset_type}" if dataset_type != 'auto' else 'cc',
        data_root=data_root,
        split=split,
        **kwargs
    )
    
    if dataset_type == 'cc3m':
        return CC3MLoader(config)
    elif dataset_type == 'cc12m':
        return CC12MLoader(config)
    else:  # auto
        return ConceptualCaptionsLoader(config)