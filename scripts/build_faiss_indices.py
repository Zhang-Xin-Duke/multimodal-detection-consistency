#!/usr/bin/env python3
"""构建和持久化FAISS索引

该脚本用于为不同数据集构建FAISS索引，支持多种索引类型和配置。
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import torch
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.models import CLIPModel
from src.datasets import COCOLoader, FlickrLoader, CCLoader, VGLoader
from src.utils import ConfigManager, HardwareDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSIndexBuilder:
    """FAISS索引构建器"""
    
    def __init__(self, config_path: str):
        """初始化索引构建器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = Config.load_config(config_path)
        self.device = torch.device(self.config.device)
        
        # 初始化模型
        self.clip_model = CLIPModel(self.config.models.clip)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # 硬件检测
        self.hardware = HardwareDetector()
        
        # 索引配置
        self.index_configs = {
            'flat': {'type': 'Flat', 'params': {}},
            'ivf': {'type': 'IVF', 'params': {'nlist': 1024}},
            'hnsw': {'type': 'HNSW', 'params': {'M': 16}},
            'pq': {'type': 'PQ', 'params': {'m': 8, 'nbits': 8}}
        }
    
    def extract_features(self, dataset_name: str, split: str = 'val') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """提取数据集特征
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            
        Returns:
            图像特征、文本特征、图像ID列表
        """
        logger.info(f"提取 {dataset_name} 数据集特征...")
        
        # 选择数据加载器
        if dataset_name == 'coco':
            loader = COCOLoader(self.config.datasets.coco)
        elif dataset_name == 'flickr30k':
            loader = FlickrLoader(self.config.datasets.flickr30k)
        elif dataset_name == 'cc3m':
            loader = CCLoader(self.config.datasets.cc3m)
        elif dataset_name == 'visual_genome':
            loader = VGLoader(self.config.datasets.visual_genome)
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 获取数据
        dataset = loader.load_dataset(split=split)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        image_features = []
        text_features = []
        image_ids = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取特征"):
                images = batch['image'].to(self.device)
                texts = batch['text']
                ids = batch['image_id']
                
                # 提取CLIP特征
                img_feat = self.clip_model.encode_image(images)
                txt_feat = self.clip_model.encode_text(texts)
                
                # 归一化特征
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                
                image_features.append(img_feat.cpu().numpy())
                text_features.append(txt_feat.cpu().numpy())
                image_ids.extend(ids)
        
        # 合并特征
        image_features = np.vstack(image_features)
        text_features = np.vstack(text_features)
        
        logger.info(f"提取完成: {len(image_features)} 个样本")
        return image_features, text_features, image_ids
    
    def build_index(self, features: np.ndarray, index_type: str = 'flat') -> faiss.Index:
        """构建FAISS索引
        
        Args:
            features: 特征向量
            index_type: 索引类型
            
        Returns:
            FAISS索引
        """
        logger.info(f"构建 {index_type} 索引...")
        
        d = features.shape[1]  # 特征维度
        config = self.index_configs[index_type]
        
        if config['type'] == 'Flat':
            index = faiss.IndexFlatIP(d)  # 内积索引
        elif config['type'] == 'IVF':
            nlist = config['params']['nlist']
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            # 训练索引
            index.train(features)
        elif config['type'] == 'HNSW':
            M = config['params']['M']
            index = faiss.IndexHNSWFlat(d, M)
        elif config['type'] == 'PQ':
            m = config['params']['m']
            nbits = config['params']['nbits']
            index = faiss.IndexPQ(d, m, nbits)
            # 训练索引
            index.train(features)
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
        
        # 添加向量到索引
        index.add(features)
        
        logger.info(f"索引构建完成: {index.ntotal} 个向量")
        return index
    
    def save_index(self, index: faiss.Index, save_path: str):
        """保存索引到文件
        
        Args:
            index: FAISS索引
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(index, save_path)
        logger.info(f"索引已保存到: {save_path}")
    
    def build_dataset_indices(self, dataset_name: str, index_types: List[str]):
        """为数据集构建多种索引
        
        Args:
            dataset_name: 数据集名称
            index_types: 索引类型列表
        """
        logger.info(f"为 {dataset_name} 数据集构建索引...")
        
        # 提取特征
        image_features, text_features, image_ids = self.extract_features(dataset_name)
        
        # 保存特征
        cache_dir = Path(self.config.cache_dir) / 'features' / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(cache_dir / 'image_features.npy', image_features)
        np.save(cache_dir / 'text_features.npy', text_features)
        np.save(cache_dir / 'image_ids.npy', np.array(image_ids))
        
        # 构建不同类型的索引
        for index_type in index_types:
            logger.info(f"构建 {index_type} 索引...")
            
            # 图像索引
            img_index = self.build_index(image_features, index_type)
            img_index_path = cache_dir / f'image_index_{index_type}.faiss'
            self.save_index(img_index, str(img_index_path))
            
            # 文本索引
            txt_index = self.build_index(text_features, index_type)
            txt_index_path = cache_dir / f'text_index_{index_type}.faiss'
            self.save_index(txt_index, str(txt_index_path))
            
            logger.info(f"{index_type} 索引构建完成")
    
    def benchmark_indices(self, dataset_name: str, index_types: List[str], k: int = 10):
        """基准测试不同索引的性能
        
        Args:
            dataset_name: 数据集名称
            index_types: 索引类型列表
            k: 检索的top-k数量
        """
        logger.info(f"基准测试 {dataset_name} 数据集索引性能...")
        
        cache_dir = Path(self.config.cache_dir) / 'features' / dataset_name
        
        # 加载特征
        image_features = np.load(cache_dir / 'image_features.npy')
        text_features = np.load(cache_dir / 'text_features.npy')
        
        # 选择测试查询
        n_queries = min(1000, len(text_features))
        query_indices = np.random.choice(len(text_features), n_queries, replace=False)
        query_features = text_features[query_indices]
        
        results = {}
        
        for index_type in index_types:
            logger.info(f"测试 {index_type} 索引...")
            
            # 加载索引
            index_path = cache_dir / f'image_index_{index_type}.faiss'
            index = faiss.read_index(str(index_path))
            
            # 测试检索时间
            import time
            start_time = time.time()
            
            distances, indices = index.search(query_features, k)
            
            end_time = time.time()
            search_time = end_time - start_time
            
            results[index_type] = {
                'search_time': search_time,
                'qps': n_queries / search_time,
                'avg_time_per_query': search_time / n_queries * 1000  # ms
            }
            
            logger.info(f"{index_type} 性能: {results[index_type]['qps']:.2f} QPS, "
                       f"{results[index_type]['avg_time_per_query']:.2f} ms/query")
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建FAISS索引')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset', type=str, choices=['coco', 'flickr30k', 'cc3m', 'visual_genome'],
                       help='数据集名称（如果不指定则构建所有数据集）')
    parser.add_argument('--index-types', nargs='+', default=['flat', 'ivf', 'hnsw'],
                       choices=['flat', 'ivf', 'hnsw', 'pq'], help='索引类型')
    parser.add_argument('--benchmark', action='store_true', help='运行基准测试')
    
    args = parser.parse_args()
    
    # 初始化构建器
    builder = FAISSIndexBuilder(args.config)
    
    # 确定要处理的数据集
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['coco', 'flickr30k', 'cc3m', 'visual_genome']
    
    # 构建索引
    for dataset in datasets:
        try:
            builder.build_dataset_indices(dataset, args.index_types)
            
            if args.benchmark:
                builder.benchmark_indices(dataset, args.index_types)
                
        except Exception as e:
            logger.error(f"处理数据集 {dataset} 时出错: {e}")
            continue
    
    logger.info("所有索引构建完成！")

if __name__ == '__main__':
    main()