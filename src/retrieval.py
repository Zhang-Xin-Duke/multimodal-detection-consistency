"""
检索模块

提供多模态检索功能，支持文本-图像检索和相似度计算。
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import faiss
from .models import CLIPModel, CLIPConfig
from PIL import Image
import time

logger = logging.getLogger(__name__)


class RetrievalIndex:
    """检索索引管理器"""
    
    def __init__(self, index_type: str = "faiss", dimension: int = 512):
        """
        初始化检索索引
        
        Args:
            index_type: 索引类型
            dimension: 特征维度
        """
        self.index_type = index_type
        self.dimension = dimension
        self.index = None
        self.features = None
        
    def build_index(self, features: np.ndarray):
        """
        构建索引
        
        Args:
            features: 特征矩阵
        """
        try:
            self.features = features.astype(np.float32)
            
            if self.index_type == "faiss":
                import faiss
                # 创建FAISS索引
                self.index = faiss.IndexFlatIP(features.shape[1])
                self.index.add(self.features)
            else:
                # 使用简单的numpy实现
                self.index = features
                
        except Exception as e:
            logger.error(f"构建索引失败: {e}")
            raise
    
    def search(self, query_features: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最相似的项目
        
        Args:
            query_features: 查询特征
            top_k: 返回数量
            
        Returns:
            (索引数组, 分数数组)
        """
        try:
            if self.index is None:
                raise ValueError("索引未构建")
                
            query_features = query_features.astype(np.float32)
            
            if self.index_type == "faiss":
                distances, indices = self.index.search(query_features, top_k)
                return indices, distances
            else:
                # 使用余弦相似度
                similarities = cosine_similarity(query_features, self.index)[0]
                indices = np.argsort(similarities)[::-1][:top_k]
                scores = similarities[indices]
                return indices, scores
                
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise
    
    def add_items(self, features: np.ndarray):
        """
        向索引添加新项目
        
        Args:
            features: 新特征
        """
        try:
            features = features.astype(np.float32)
            
            if self.index_type == "faiss" and self.index is not None:
                self.index.add(features)
            
            # 更新特征矩阵
            if self.features is not None:
                self.features = np.vstack([self.features, features])
            else:
                self.features = features
                
        except Exception as e:
            logger.error(f"添加项目失败: {e}")
            raise


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 模型配置
    clip_model: str = "ViT-B/32"
    device: str = "cuda"
    batch_size: int = 256
    
    # 检索配置
    top_k: int = 10
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product
    
    # 索引配置
    index_type: str = "faiss"  # faiss, sklearn, exact
    faiss_index_type: str = "IndexFlatIP"  # IndexFlatIP, IndexIVFFlat, IndexHNSWFlat
    n_clusters: int = 100  # for IVF index
    
    # 缓存配置
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    
    # 性能配置
    normalize_features: bool = True
    use_gpu_index: bool = True


class MultiModalRetriever:
    """多模态检索器"""
    
    def __init__(self, config: RetrievalConfig):
        """
        初始化多模态检索器
        
        Args:
            config: 检索配置
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化CLIP模型
        self.clip_model = self._initialize_clip_model()
        
        # 数据存储
        self.image_features = None
        self.text_features = None
        self.image_paths = []
        self.texts = []
        
        # 索引
        self.image_index = None
        self.text_index = None
        
        # 缓存
        self.feature_cache = {}
        self.retrieval_cache = {}
        
        logger.info("多模态检索器初始化完成")
    
    def _initialize_clip_model(self) -> CLIPModel:
        """
        初始化CLIP模型
        
        Returns:
            CLIP模型实例
        """
        try:
            clip_config = CLIPConfig(
                model_name=self.config.clip_model,
                device=self.config.device,
                batch_size=self.config.batch_size,
                normalize=self.config.normalize_features
            )
            
            clip_model = CLIPModel(clip_config)
            logger.info(f"CLIP模型初始化完成: {self.config.clip_model}")
            
            return clip_model
            
        except Exception as e:
            logger.error(f"CLIP模型初始化失败: {e}")
            raise
    
    def build_image_index(self, image_paths: List[str], 
                         save_path: Optional[str] = None) -> np.ndarray:
        """
        构建图像索引
        
        Args:
            image_paths: 图像路径列表
            save_path: 保存路径
            
        Returns:
            图像特征矩阵
        """
        try:
            logger.info(f"开始构建图像索引: {len(image_paths)}张图像")
            start_time = time.time()
            
            # 提取图像特征
            # 加载图像
            images = []
            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                except Exception as e:
                    logger.warning(f"无法加载图像 {path}: {e}")
                    continue
            
            # 编码图像特征
            image_features = self.clip_model.encode_image(
                images, 
                normalize=self.config.normalize_features
            )
            
            # 转换为numpy数组
            self.image_features = image_features.numpy()
            
            self.image_paths = image_paths.copy()
            
            # 构建FAISS索引
            self.image_index = self._build_faiss_index(self.image_features)
            
            # 保存索引
            if save_path:
                self.save_image_index(save_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"图像索引构建完成: {self.image_features.shape}, 耗时: {elapsed_time:.2f}秒")
            
            return self.image_features
            
        except Exception as e:
            logger.error(f"构建图像索引失败: {e}")
            raise
    
    def build_text_index(self, texts: List[str], 
                        save_path: Optional[str] = None) -> np.ndarray:
        """
        构建文本索引
        
        Args:
            texts: 文本列表
            save_path: 保存路径
            
        Returns:
            文本特征矩阵
        """
        try:
            logger.info(f"开始构建文本索引: {len(texts)}个文本")
            start_time = time.time()
            
            # 提取文本特征
            text_features = self.clip_model.encode_text(
                texts, 
                normalize=self.config.normalize_features
            )
            
            # 转换为numpy数组
            self.text_features = text_features.numpy()
            
            self.texts = texts.copy()
            
            # 构建FAISS索引
            self.text_index = self._build_faiss_index(self.text_features)
            
            # 保存索引
            if save_path:
                self.save_text_index(save_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"文本索引构建完成: {self.text_features.shape}, 耗时: {elapsed_time:.2f}秒")
            
            return self.text_features
            
        except Exception as e:
            logger.error(f"构建文本索引失败: {e}")
            raise
    
    def _build_faiss_index(self, features: np.ndarray):
        """
        构建FAISS索引
        
        Args:
            features: 特征矩阵
            
        Returns:
            FAISS索引
        """
        try:
            if self.config.index_type != "faiss":
                return None
            
            dimension = features.shape[1]
            
            # 选择索引类型
            if self.config.faiss_index_type == "IndexFlatIP":
                index = faiss.IndexFlatIP(dimension)
            elif self.config.faiss_index_type == "IndexIVFFlat":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, self.config.n_clusters)
            elif self.config.faiss_index_type == "IndexHNSWFlat":
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                index = faiss.IndexFlatIP(dimension)
            
            # 使用GPU加速（如果可用）
            if self.config.use_gpu_index and torch.cuda.is_available():
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info("使用GPU加速FAISS索引")
                except Exception as e:
                    logger.warning(f"GPU索引创建失败，使用CPU索引: {e}")
            
            # 训练索引（如果需要）
            if hasattr(index, 'train') and not index.is_trained:
                index.train(features.astype(np.float32))
            
            # 添加特征
            index.add(features.astype(np.float32))
            
            logger.debug(f"FAISS索引构建完成: {index.ntotal}个向量")
            return index
            
        except Exception as e:
            logger.error(f"构建FAISS索引失败: {e}")
            return None
    
    def retrieve_images_by_text(self, query_text: str, 
                               top_k: Optional[int] = None) -> Tuple[List[str], List[float]]:
        """
        根据文本检索图像
        
        Args:
            query_text: 查询文本
            top_k: 返回的图像数量
            
        Returns:
            (图像路径列表, 相似度分数列表)
        """
        try:
            top_k = top_k or self.config.top_k
            
            # 检查缓存
            cache_key = f"text2img_{query_text}_{top_k}"
            if self.config.enable_cache and cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            if self.image_features is None or self.image_index is None:
                raise ValueError("图像索引未构建")
            
            # 编码查询文本
            query_features = self.clip_model.encode_text(
                [query_text], 
                normalize=self.config.normalize_features
            ).numpy()
            
            # 检索
            indices, scores = self._search_index(
                self.image_index, 
                query_features, 
                top_k
            )
            
            # 获取结果
            retrieved_paths = [self.image_paths[i] for i in indices]
            retrieved_scores = scores.tolist()
            
            # 缓存结果
            if self.config.enable_cache:
                self.retrieval_cache[cache_key] = (retrieved_paths, retrieved_scores)
            
            logger.debug(f"文本检索图像完成: {query_text} -> {len(retrieved_paths)}张图像")
            return retrieved_paths, retrieved_scores
            
        except Exception as e:
            logger.error(f"文本检索图像失败: {e}")
            return [], []
    
    def retrieve_texts_by_image(self, query_image: Union[str, Image.Image], 
                               top_k: Optional[int] = None) -> Tuple[List[str], List[float]]:
        """
        根据图像检索文本
        
        Args:
            query_image: 查询图像（路径或PIL图像）
            top_k: 返回的文本数量
            
        Returns:
            (文本列表, 相似度分数列表)
        """
        try:
            top_k = top_k or self.config.top_k
            
            if self.text_features is None or self.text_index is None:
                raise ValueError("文本索引未构建")
            
            # 处理图像输入
            if isinstance(query_image, str):
                image = Image.open(query_image).convert('RGB')
                cache_key = f"img2text_{query_image}_{top_k}"
            else:
                image = query_image
                cache_key = f"img2text_pil_{id(image)}_{top_k}"
            
            # 检查缓存
            if self.config.enable_cache and cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            # 编码查询图像
            query_features = self.clip_model.encode_image(
                [image], 
                normalize=self.config.normalize_features
            ).numpy()
            
            # 检索
            indices, scores = self._search_index(
                self.text_index, 
                query_features, 
                top_k
            )
            
            # 获取结果
            retrieved_texts = [self.texts[i] for i in indices]
            retrieved_scores = scores.tolist()
            
            # 缓存结果
            if self.config.enable_cache:
                self.retrieval_cache[cache_key] = (retrieved_texts, retrieved_scores)
            
            logger.debug(f"图像检索文本完成: -> {len(retrieved_texts)}个文本")
            return retrieved_texts, retrieved_scores
            
        except Exception as e:
            logger.error(f"图像检索文本失败: {e}")
            return [], []
    
    def _search_index(self, index, query_features: np.ndarray, 
                     top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        在索引中搜索
        
        Args:
            index: 搜索索引
            query_features: 查询特征
            top_k: 返回数量
            
        Returns:
            (索引数组, 分数数组)
        """
        try:
            if self.config.index_type == "faiss" and index is not None:
                # 使用FAISS搜索
                scores, indices = index.search(
                    query_features.astype(np.float32), 
                    top_k
                )
                return indices[0], scores[0]
            
            else:
                # 使用精确搜索
                if index is None:
                    # 直接计算相似度
                    if hasattr(self, 'image_features') and self.image_features is not None:
                        features = self.image_features
                    elif hasattr(self, 'text_features') and self.text_features is not None:
                        features = self.text_features
                    else:
                        raise ValueError("没有可用的特征矩阵")
                    
                    similarities = cosine_similarity(query_features, features)[0]
                    indices = np.argsort(similarities)[::-1][:top_k]
                    scores = similarities[indices]
                    
                    return indices, scores
                
                else:
                    raise ValueError(f"不支持的索引类型: {self.config.index_type}")
            
        except Exception as e:
            logger.error(f"索引搜索失败: {e}")
            return np.array([]), np.array([])
    
    def compute_similarity_matrix(self, text_features: Optional[np.ndarray] = None,
                                 image_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算文本-图像相似度矩阵
        
        Args:
            text_features: 文本特征矩阵
            image_features: 图像特征矩阵
            
        Returns:
            相似度矩阵 (N_text, N_image)
        """
        try:
            # 使用默认特征或提供的特征
            if text_features is None:
                text_features = self.text_features
            if image_features is None:
                image_features = self.image_features
            
            if text_features is None or image_features is None:
                raise ValueError("缺少文本或图像特征")
            
            # 计算相似度矩阵
            if self.config.similarity_metric == "cosine":
                similarity_matrix = cosine_similarity(text_features, image_features)
            elif self.config.similarity_metric == "dot_product":
                similarity_matrix = np.dot(text_features, image_features.T)
            elif self.config.similarity_metric == "euclidean":
                # 欧几里得距离（转换为相似度）
                from sklearn.metrics.pairwise import euclidean_distances
                distances = euclidean_distances(text_features, image_features)
                similarity_matrix = 1 / (1 + distances)  # 转换为相似度
            else:
                raise ValueError(f"不支持的相似度度量: {self.config.similarity_metric}")
            
            logger.debug(f"相似度矩阵计算完成: {similarity_matrix.shape}")
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"计算相似度矩阵失败: {e}")
            raise
    
    def batch_retrieve_images_by_texts(self, query_texts: List[str], 
                                      top_k: Optional[int] = None) -> List[Tuple[List[str], List[float]]]:
        """
        批量根据文本检索图像
        
        Args:
            query_texts: 查询文本列表
            top_k: 每个查询返回的图像数量
            
        Returns:
            检索结果列表
        """
        results = []
        
        for query_text in query_texts:
            paths, scores = self.retrieve_images_by_text(query_text, top_k)
            results.append((paths, scores))
        
        return results
    
    def batch_retrieve_texts_by_images(self, query_images: List[Union[str, Image.Image]], 
                                      top_k: Optional[int] = None) -> List[Tuple[List[str], List[float]]]:
        """
        批量根据图像检索文本
        
        Args:
            query_images: 查询图像列表
            top_k: 每个查询返回的文本数量
            
        Returns:
            检索结果列表
        """
        results = []
        
        for query_image in query_images:
            texts, scores = self.retrieve_texts_by_image(query_image, top_k)
            results.append((texts, scores))
        
        return results
    
    def save_image_index(self, save_path: str):
        """
        保存图像索引
        
        Args:
            save_path: 保存路径
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存数据
            index_data = {
                'image_features': self.image_features,
                'image_paths': self.image_paths,
                'config': self.config
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            # 保存FAISS索引
            if self.image_index is not None:
                faiss_path = save_path.with_suffix('.faiss')
                faiss.write_index(self.image_index, str(faiss_path))
            
            logger.info(f"图像索引已保存: {save_path}")
            
        except Exception as e:
            logger.error(f"保存图像索引失败: {e}")
            raise
    
    def save_text_index(self, save_path: str):
        """
        保存文本索引
        
        Args:
            save_path: 保存路径
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存数据
            index_data = {
                'text_features': self.text_features,
                'texts': self.texts,
                'config': self.config
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            # 保存FAISS索引
            if self.text_index is not None:
                faiss_path = save_path.with_suffix('.faiss')
                faiss.write_index(self.text_index, str(faiss_path))
            
            logger.info(f"文本索引已保存: {save_path}")
            
        except Exception as e:
            logger.error(f"保存文本索引失败: {e}")
            raise
    
    def load_image_index(self, load_path: str):
        """
        加载图像索引
        
        Args:
            load_path: 加载路径
        """
        try:
            load_path = Path(load_path)
            
            # 加载数据
            with open(load_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.image_features = index_data['image_features']
            self.image_paths = index_data['image_paths']
            
            # 加载FAISS索引
            faiss_path = load_path.with_suffix('.faiss')
            if faiss_path.exists():
                self.image_index = faiss.read_index(str(faiss_path))
            
            logger.info(f"图像索引已加载: {load_path}")
            
        except Exception as e:
            logger.error(f"加载图像索引失败: {e}")
            raise
    
    def load_text_index(self, load_path: str):
        """
        加载文本索引
        
        Args:
            load_path: 加载路径
        """
        try:
            load_path = Path(load_path)
            
            # 加载数据
            with open(load_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.text_features = index_data['text_features']
            self.texts = index_data['texts']
            
            # 加载FAISS索引
            faiss_path = load_path.with_suffix('.faiss')
            if faiss_path.exists():
                self.text_index = faiss.read_index(str(faiss_path))
            
            logger.info(f"文本索引已加载: {load_path}")
            
        except Exception as e:
            logger.error(f"加载文本索引失败: {e}")
            raise
    
    def clear_cache(self):
        """
        清理缓存
        """
        self.feature_cache.clear()
        self.retrieval_cache.clear()
        logger.info("检索缓存已清理")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'image_count': len(self.image_paths) if self.image_paths else 0,
            'text_count': len(self.texts) if self.texts else 0,
            'image_features_shape': self.image_features.shape if self.image_features is not None else None,
            'text_features_shape': self.text_features.shape if self.text_features is not None else None,
            'feature_cache_size': len(self.feature_cache),
            'retrieval_cache_size': len(self.retrieval_cache),
            'config': {
                'clip_model': self.config.clip_model,
                'top_k': self.config.top_k,
                'similarity_metric': self.config.similarity_metric,
                'index_type': self.config.index_type
            }
        }


def create_retriever(config: Optional[RetrievalConfig] = None) -> MultiModalRetriever:
    """
    创建多模态检索器实例
    
    Args:
        config: 检索配置
        
    Returns:
        多模态检索器实例
    """
    if config is None:
        config = RetrievalConfig()
    
    return MultiModalRetriever(config)