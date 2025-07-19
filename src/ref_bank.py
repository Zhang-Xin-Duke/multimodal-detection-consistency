"""参考向量库管理模块

提供参考向量的存储、管理、聚类和检索功能。
"""

import numpy as np
import torch
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from collections import defaultdict, deque
import logging
from threading import Lock
import time
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ReferenceBankConfig:
    """参考库配置"""
    max_size: int = 10000  # 最大存储数量
    similarity_threshold: float = 0.9  # 相似度阈值
    clustering_method: str = "kmeans"  # 聚类方法
    num_clusters: int = 100  # 聚类数量
    update_strategy: str = "fifo"  # 更新策略
    persistence_enabled: bool = True  # 是否持久化
    save_path: str = "./cache/ref_bank"  # 保存路径
    auto_clustering: bool = True  # 自动聚类
    clustering_interval: int = 1000  # 聚类间隔
    feature_dim: int = 512  # 特征维度
    
    def __post_init__(self):
        """配置后处理"""
        if self.clustering_method not in ["kmeans", "dbscan", "none"]:
            raise ValueError(f"不支持的聚类方法: {self.clustering_method}")
        
        if self.update_strategy not in ["fifo", "lru", "random", "similarity"]:
            raise ValueError(f"不支持的更新策略: {self.update_strategy}")


@dataclass
class ReferenceItem:
    """参考项"""
    vector: np.ndarray  # 特征向量
    metadata: Dict[str, Any]  # 元数据
    timestamp: float  # 时间戳
    access_count: int = 0  # 访问次数
    cluster_id: Optional[int] = None  # 聚类ID
    similarity_scores: Optional[Dict[str, float]] = None  # 相似度分数
    
    def __post_init__(self):
        """后处理"""
        if self.similarity_scores is None:
            self.similarity_scores = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "vector": self.vector.tolist(),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "cluster_id": self.cluster_id,
            "similarity_scores": self.similarity_scores
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReferenceItem':
        """从字典创建"""
        return cls(
            vector=np.array(data["vector"]),
            metadata=data["metadata"],
            timestamp=data["timestamp"],
            access_count=data.get("access_count", 0),
            cluster_id=data.get("cluster_id"),
            similarity_scores=data.get("similarity_scores", {})
        )


class ReferenceBank:
    """参考向量库管理器"""
    
    def __init__(self, config: ReferenceBankConfig):
        """
        初始化参考库
        
        Args:
            config: 参考库配置
        """
        self.config = config
        self.references: List[ReferenceItem] = []
        self.clusters: Dict[int, List[int]] = {}  # 聚类ID -> 参考索引列表
        self.cluster_centers: Optional[np.ndarray] = None
        self.access_order: deque = deque()  # LRU访问顺序
        
        # 线程安全
        self._lock = Lock()
        
        # 统计信息
        self.stats = {
            "total_added": 0,
            "total_removed": 0,
            "total_queries": 0,
            "clustering_count": 0,
            "last_clustering_time": None
        }
        
        # 创建保存目录
        if self.config.persistence_enabled:
            Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
        
        # 加载已有数据
        self._load_from_disk()
        
        logger.info(f"参考库初始化完成，当前大小: {len(self.references)}")
    
    def add_reference(self, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """
        添加参考向量
        
        Args:
            vector: 特征向量
            metadata: 元数据
        
        Returns:
            是否成功添加
        """
        try:
            with self._lock:
                # 检查相似度
                if self._is_too_similar(vector):
                    logger.debug("向量过于相似，跳过添加")
                    return False
                
                # 创建参考项
                ref_item = ReferenceItem(
                    vector=vector.copy(),
                    metadata=metadata.copy(),
                    timestamp=time.time()
                )
                
                # 检查容量
                if len(self.references) >= self.config.max_size:
                    self._remove_reference()
                
                # 添加参考
                self.references.append(ref_item)
                self.stats["total_added"] += 1
                
                # 自动聚类
                if (self.config.auto_clustering and 
                    len(self.references) % self.config.clustering_interval == 0):
                    self._perform_clustering()
                
                # 持久化
                if self.config.persistence_enabled:
                    self._save_to_disk()
                
                logger.debug(f"成功添加参考向量，当前大小: {len(self.references)}")
                return True
                
        except Exception as e:
            logger.error(f"添加参考向量失败: {e}")
            return False
    
    def query_similar(self, query_vector: np.ndarray, 
                     top_k: int = 10, 
                     similarity_threshold: Optional[float] = None) -> List[Tuple[ReferenceItem, float]]:
        """
        查询相似参考向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回top-k结果
            similarity_threshold: 相似度阈值
        
        Returns:
            (参考项, 相似度分数)列表
        """
        try:
            with self._lock:
                if not self.references:
                    return []
                
                threshold = similarity_threshold or self.config.similarity_threshold
                
                # 计算相似度
                similarities = self._compute_similarities(query_vector)
                
                # 过滤和排序
                valid_indices = np.where(similarities >= threshold)[0]
                if len(valid_indices) == 0:
                    return []
                
                # 排序并取top-k
                sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
                top_indices = sorted_indices[:top_k]
                
                # 更新访问统计
                results = []
                for idx in top_indices:
                    ref_item = self.references[idx]
                    ref_item.access_count += 1
                    
                    # 更新LRU
                    if idx in self.access_order:
                        self.access_order.remove(idx)
                    self.access_order.append(idx)
                    
                    results.append((ref_item, similarities[idx]))
                
                self.stats["total_queries"] += 1
                
                return results
                
        except Exception as e:
            logger.error(f"查询相似向量失败: {e}")
            return []
    
    def query_by_cluster(self, cluster_id: int, top_k: int = 10) -> List[ReferenceItem]:
        """
        按聚类查询参考向量
        
        Args:
            cluster_id: 聚类ID
            top_k: 返回数量
        
        Returns:
            参考项列表
        """
        try:
            with self._lock:
                if cluster_id not in self.clusters:
                    return []
                
                indices = self.clusters[cluster_id][:top_k]
                return [self.references[idx] for idx in indices]
                
        except Exception as e:
            logger.error(f"按聚类查询失败: {e}")
            return []
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        获取聚类中心
        
        Returns:
            聚类中心数组
        """
        with self._lock:
            return self.cluster_centers.copy() if self.cluster_centers is not None else None
    
    def perform_clustering(self, force: bool = False) -> bool:
        """
        执行聚类
        
        Args:
            force: 是否强制聚类
        
        Returns:
            是否成功
        """
        try:
            with self._lock:
                return self._perform_clustering(force)
        except Exception as e:
            logger.error(f"聚类失败: {e}")
            return False
    
    def _perform_clustering(self, force: bool = False) -> bool:
        """
        内部聚类实现
        
        Args:
            force: 是否强制聚类
        
        Returns:
            是否成功
        """
        if not self.references or len(self.references) < 2:
            return False
        
        if self.config.clustering_method == "none" and not force:
            return False
        
        try:
            # 提取特征向量
            vectors = np.array([ref.vector for ref in self.references])
            
            if self.config.clustering_method == "kmeans":
                n_clusters = min(self.config.num_clusters, len(self.references))
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(vectors)
                self.cluster_centers = clusterer.cluster_centers_
                
            elif self.config.clustering_method == "dbscan":
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = clusterer.fit_predict(vectors)
                
                # 计算聚类中心
                unique_labels = np.unique(cluster_labels)
                centers = []
                for label in unique_labels:
                    if label != -1:  # 排除噪声点
                        mask = cluster_labels == label
                        center = np.mean(vectors[mask], axis=0)
                        centers.append(center)
                
                self.cluster_centers = np.array(centers) if centers else None
            
            else:
                return False
            
            # 更新聚类信息
            self.clusters.clear()
            for idx, label in enumerate(cluster_labels):
                if label != -1:  # 排除噪声点
                    self.references[idx].cluster_id = int(label)
                    if label not in self.clusters:
                        self.clusters[label] = []
                    self.clusters[label].append(idx)
                else:
                    self.references[idx].cluster_id = None
            
            self.stats["clustering_count"] += 1
            self.stats["last_clustering_time"] = time.time()
            
            logger.info(f"聚类完成，生成 {len(self.clusters)} 个聚类")
            return True
            
        except Exception as e:
            logger.error(f"聚类执行失败: {e}")
            return False
    
    def _is_too_similar(self, vector: np.ndarray) -> bool:
        """
        检查向量是否过于相似
        
        Args:
            vector: 待检查向量
        
        Returns:
            是否过于相似
        """
        if not self.references:
            return False
        
        # 随机采样检查（避免计算所有相似度）
        sample_size = min(100, len(self.references))
        sample_indices = np.random.choice(len(self.references), sample_size, replace=False)
        
        for idx in sample_indices:
            similarity = self._cosine_similarity(vector, self.references[idx].vector)
            if similarity > self.config.similarity_threshold:
                return True
        
        return False
    
    def _remove_reference(self):
        """
        根据策略移除参考向量
        """
        if not self.references:
            return
        
        if self.config.update_strategy == "fifo":
            # 先进先出
            removed_ref = self.references.pop(0)
            # 更新聚类信息
            self._update_clusters_after_removal(0)
            
        elif self.config.update_strategy == "lru":
            # 最少使用
            if self.access_order:
                oldest_idx = self.access_order.popleft()
                if oldest_idx < len(self.references):
                    self.references.pop(oldest_idx)
                    self._update_clusters_after_removal(oldest_idx)
            else:
                self.references.pop(0)
                self._update_clusters_after_removal(0)
                
        elif self.config.update_strategy == "random":
            # 随机移除
            idx = np.random.randint(len(self.references))
            self.references.pop(idx)
            self._update_clusters_after_removal(idx)
            
        elif self.config.update_strategy == "similarity":
            # 移除最相似的
            self._remove_most_similar()
        
        self.stats["total_removed"] += 1
    
    def _remove_most_similar(self):
        """
        移除最相似的参考向量
        """
        if len(self.references) < 2:
            return
        
        max_similarity = -1
        remove_idx = 0
        
        # 计算所有向量间的相似度，找到最相似的一对
        for i in range(len(self.references)):
            for j in range(i + 1, len(self.references)):
                similarity = self._cosine_similarity(
                    self.references[i].vector, 
                    self.references[j].vector
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    # 移除访问次数较少的
                    if self.references[i].access_count <= self.references[j].access_count:
                        remove_idx = i
                    else:
                        remove_idx = j
        
        self.references.pop(remove_idx)
        self._update_clusters_after_removal(remove_idx)
    
    def _update_clusters_after_removal(self, removed_idx: int):
        """
        移除参考后更新聚类信息
        
        Args:
            removed_idx: 被移除的索引
        """
        # 更新聚类索引
        new_clusters = {}
        for cluster_id, indices in self.clusters.items():
            new_indices = []
            for idx in indices:
                if idx < removed_idx:
                    new_indices.append(idx)
                elif idx > removed_idx:
                    new_indices.append(idx - 1)
                # idx == removed_idx 的情况跳过
            
            if new_indices:
                new_clusters[cluster_id] = new_indices
        
        self.clusters = new_clusters
        
        # 更新访问顺序
        new_access_order = deque()
        for idx in self.access_order:
            if idx < removed_idx:
                new_access_order.append(idx)
            elif idx > removed_idx:
                new_access_order.append(idx - 1)
        
        self.access_order = new_access_order
    
    def _compute_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """
        计算查询向量与所有参考向量的相似度
        
        Args:
            query_vector: 查询向量
        
        Returns:
            相似度数组
        """
        if not self.references:
            return np.array([])
        
        ref_vectors = np.array([ref.vector for ref in self.references])
        
        # 批量计算余弦相似度
        query_norm = np.linalg.norm(query_vector)
        ref_norms = np.linalg.norm(ref_vectors, axis=1)
        
        dot_products = np.dot(ref_vectors, query_vector)
        similarities = dot_products / (ref_norms * query_norm + 1e-8)
        
        return similarities
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1, vec2: 输入向量
        
        Returns:
            余弦相似度
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _save_to_disk(self):
        """
        保存到磁盘
        """
        try:
            save_path = Path(self.config.save_path)
            
            # 保存参考数据
            refs_data = [ref.to_dict() for ref in self.references]
            with open(save_path / "references.json", "w") as f:
                json.dump(refs_data, f, indent=2)
            
            # 保存聚类信息
            cluster_data = {
                "clusters": {str(k): v for k, v in self.clusters.items()},
                "cluster_centers": self.cluster_centers.tolist() if self.cluster_centers is not None else None
            }
            with open(save_path / "clusters.json", "w") as f:
                json.dump(cluster_data, f, indent=2)
            
            # 保存统计信息
            with open(save_path / "stats.json", "w") as f:
                json.dump(self.stats, f, indent=2)
            
            # 保存配置
            with open(save_path / "config.json", "w") as f:
                json.dump(asdict(self.config), f, indent=2)
            
            logger.debug("参考库已保存到磁盘")
            
        except Exception as e:
            logger.error(f"保存到磁盘失败: {e}")
    
    def _load_from_disk(self):
        """
        从磁盘加载
        """
        try:
            save_path = Path(self.config.save_path)
            
            # 检查文件是否存在
            refs_file = save_path / "references.json"
            if not refs_file.exists():
                return
            
            # 加载参考数据
            with open(refs_file, "r") as f:
                refs_data = json.load(f)
            
            self.references = [ReferenceItem.from_dict(data) for data in refs_data]
            
            # 加载聚类信息
            cluster_file = save_path / "clusters.json"
            if cluster_file.exists():
                with open(cluster_file, "r") as f:
                    cluster_data = json.load(f)
                
                self.clusters = {int(k): v for k, v in cluster_data["clusters"].items()}
                
                if cluster_data["cluster_centers"]:
                    self.cluster_centers = np.array(cluster_data["cluster_centers"])
            
            # 加载统计信息
            stats_file = save_path / "stats.json"
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    self.stats.update(json.load(f))
            
            logger.info(f"从磁盘加载参考库，大小: {len(self.references)}")
            
        except Exception as e:
            logger.error(f"从磁盘加载失败: {e}")
    
    def clear(self):
        """
        清空参考库
        """
        with self._lock:
            self.references.clear()
            self.clusters.clear()
            self.cluster_centers = None
            self.access_order.clear()
            
            # 重置统计
            self.stats = {
                "total_added": 0,
                "total_removed": 0,
                "total_queries": 0,
                "clustering_count": 0,
                "last_clustering_time": None
            }
            
            logger.info("参考库已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                "current_size": len(self.references),
                "num_clusters": len(self.clusters),
                "max_size": self.config.max_size,
                "clustering_method": self.config.clustering_method,
                "update_strategy": self.config.update_strategy
            })
            
            if self.references:
                access_counts = [ref.access_count for ref in self.references]
                stats.update({
                    "avg_access_count": np.mean(access_counts),
                    "max_access_count": np.max(access_counts),
                    "min_access_count": np.min(access_counts)
                })
            
            return stats
    
    def export_references(self, export_path: str, format: str = "json") -> bool:
        """
        导出参考数据
        
        Args:
            export_path: 导出路径
            format: 导出格式 ("json", "numpy", "pickle")
        
        Returns:
            是否成功
        """
        try:
            with self._lock:
                export_path = Path(export_path)
                export_path.parent.mkdir(parents=True, exist_ok=True)
                
                if format == "json":
                    data = [ref.to_dict() for ref in self.references]
                    with open(export_path, "w") as f:
                        json.dump(data, f, indent=2)
                        
                elif format == "numpy":
                    vectors = np.array([ref.vector for ref in self.references])
                    metadata = [ref.metadata for ref in self.references]
                    
                    np.savez(export_path, vectors=vectors, metadata=metadata)
                    
                elif format == "pickle":
                    with open(export_path, "wb") as f:
                        pickle.dump(self.references, f)
                        
                else:
                    raise ValueError(f"不支持的导出格式: {format}")
                
                logger.info(f"参考数据已导出到: {export_path}")
                return True
                
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return False
    
    def import_references(self, import_path: str, format: str = "json") -> bool:
        """
        导入参考数据
        
        Args:
            import_path: 导入路径
            format: 导入格式
        
        Returns:
            是否成功
        """
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                raise FileNotFoundError(f"文件不存在: {import_path}")
            
            with self._lock:
                if format == "json":
                    with open(import_path, "r") as f:
                        data = json.load(f)
                    
                    imported_refs = [ReferenceItem.from_dict(item) for item in data]
                    
                elif format == "numpy":
                    data = np.load(import_path, allow_pickle=True)
                    vectors = data["vectors"]
                    metadata = data["metadata"]
                    
                    imported_refs = []
                    for vec, meta in zip(vectors, metadata):
                        ref = ReferenceItem(
                            vector=vec,
                            metadata=meta,
                            timestamp=time.time()
                        )
                        imported_refs.append(ref)
                        
                elif format == "pickle":
                    with open(import_path, "rb") as f:
                        imported_refs = pickle.load(f)
                        
                else:
                    raise ValueError(f"不支持的导入格式: {format}")
                
                # 添加导入的参考
                for ref in imported_refs:
                    if len(self.references) < self.config.max_size:
                        self.references.append(ref)
                        self.stats["total_added"] += 1
                    else:
                        break
                
                logger.info(f"成功导入 {len(imported_refs)} 个参考项")
                return True
                
        except Exception as e:
            logger.error(f"导入失败: {e}")
            return False


def create_reference_bank(max_size: int = 10000,
                         similarity_threshold: float = 0.9,
                         clustering_method: str = "kmeans",
                         **kwargs) -> ReferenceBank:
    """
    创建参考库实例
    
    Args:
        max_size: 最大容量
        similarity_threshold: 相似度阈值
        clustering_method: 聚类方法
        **kwargs: 其他配置参数
    
    Returns:
        参考库实例
    """
    config = ReferenceBankConfig(
        max_size=max_size,
        similarity_threshold=similarity_threshold,
        clustering_method=clustering_method,
        **kwargs
    )
    
    return ReferenceBank(config)


# 导出的主要类和函数
__all__ = [
    "ReferenceBankConfig",
    "ReferenceItem",
    "ReferenceBank",
    "create_reference_bank"
]