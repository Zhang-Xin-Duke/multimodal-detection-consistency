"""检索参考图像模块

该模块从预构建的图像库中检索与查询文本最相似的参考图像，
用于与输入图像进行一致性比较，检测对抗攻击。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """检索参考配置"""
    reference_count: int = 5
    similarity_threshold: float = 0.3
    use_faiss: bool = True
    faiss_index_type: str = "IVF"  # IVF, Flat, HNSW
    nlist: int = 100  # IVF聚类数
    nprobe: int = 10  # 搜索聚类数
    device: str = "cuda"
    cache_size: int = 1000
    enable_reranking: bool = True
    rerank_top_k: int = 20

class RetrievalReferenceGenerator:
    """检索参考图像生成器
    
    从预构建的图像特征库中检索与查询文本最相似的图像，
    这些图像用作参考来检测对抗攻击。
    """
    
    def __init__(self, 
                 clip_model,
                 reference_db_path: str,
                 config: RetrievalConfig = None):
        """
        初始化检索参考生成器
        
        Args:
            clip_model: CLIP模型实例
            reference_db_path: 参考数据库路径
            config: 检索配置
        """
        self.clip_model = clip_model
        self.config = config or RetrievalConfig()
        self.device = torch.device(self.config.device)
        self.reference_db_path = Path(reference_db_path)
        
        # 初始化组件
        self.reference_features = None
        self.reference_metadata = None
        self.faiss_index = None
        self.feature_cache = {}
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 检索统计
        self.retrieval_stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'cache_hits': 0,
            'average_retrieval_time': 0.0,
            'average_similarity': 0.0
        }
        
        # 加载参考数据库
        self._load_reference_database()
        
        logger.info("检索参考生成器初始化完成")
    
    def _load_reference_database(self):
        """加载参考数据库"""
        try:
            # 检查数据库文件是否存在
            features_path = self.reference_db_path / "features.npy"
            metadata_path = self.reference_db_path / "metadata.json"
            
            if not features_path.exists() or not metadata_path.exists():
                logger.warning(f"参考数据库不存在: {self.reference_db_path}")
                logger.info("将创建空的参考数据库")
                self._create_empty_database()
                return
            
            # 加载特征
            self.reference_features = np.load(features_path)
            logger.info(f"加载了 {self.reference_features.shape[0]} 个参考特征")
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.reference_metadata = json.load(f)
            
            # 初始化FAISS索引
            if self.config.use_faiss:
                self._initialize_faiss_index()
            
            logger.info("参考数据库加载完成")
            
        except Exception as e:
            logger.error(f"加载参考数据库失败: {e}")
            self._create_empty_database()
    
    def _create_empty_database(self):
        """创建空的参考数据库"""
        self.reference_features = np.empty((0, 512))  # CLIP特征维度
        self.reference_metadata = []
        
        # 创建目录
        self.reference_db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("创建了空的参考数据库")
    
    def _initialize_faiss_index(self):
        """初始化FAISS索引"""
        try:
            import faiss
            
            feature_dim = self.reference_features.shape[1]
            num_features = self.reference_features.shape[0]
            
            if num_features == 0:
                logger.warning("参考特征为空，跳过FAISS索引初始化")
                return
            
            # 选择索引类型
            if self.config.faiss_index_type == "Flat":
                self.faiss_index = faiss.IndexFlatIP(feature_dim)  # 内积索引
            elif self.config.faiss_index_type == "IVF":
                quantizer = faiss.IndexFlatIP(feature_dim)
                self.faiss_index = faiss.IndexIVFFlat(
                    quantizer, feature_dim, min(self.config.nlist, num_features)
                )
            elif self.config.faiss_index_type == "HNSW":
                self.faiss_index = faiss.IndexHNSWFlat(feature_dim, 32)
            else:
                raise ValueError(f"不支持的FAISS索引类型: {self.config.faiss_index_type}")
            
            # 训练索引（如果需要）
            if hasattr(self.faiss_index, 'train'):
                self.faiss_index.train(self.reference_features.astype(np.float32))
            
            # 添加特征
            self.faiss_index.add(self.reference_features.astype(np.float32))
            
            # 设置搜索参数
            if hasattr(self.faiss_index, 'nprobe'):
                self.faiss_index.nprobe = self.config.nprobe
            
            logger.info(f"FAISS索引初始化完成，类型: {self.config.faiss_index_type}")
            
        except ImportError:
            logger.warning("FAISS未安装，使用numpy实现检索")
            self.config.use_faiss = False
            self.faiss_index = None
        except Exception as e:
            logger.error(f"FAISS索引初始化失败: {e}")
            self.config.use_faiss = False
            self.faiss_index = None
    
    def retrieve_references(self, text: str) -> List[Dict[str, Any]]:
        """检索参考图像
        
        Args:
            text: 查询文本
            
        Returns:
            检索到的参考图像信息列表
        """
        import time
        start_time = time.time()
        
        # 检查缓存
        cache_key = hash(text)
        if cache_key in self.feature_cache:
            self.retrieval_stats['cache_hits'] += 1
            return self.feature_cache[cache_key]
        
        try:
            # 编码查询文本
            query_features = self._encode_text(text)
            
            if self.reference_features.shape[0] == 0:
                logger.warning("参考数据库为空")
                return []
            
            # 执行检索
            if self.config.use_faiss and self.faiss_index is not None:
                retrieved_refs = self._faiss_retrieve(query_features)
            else:
                retrieved_refs = self._numpy_retrieve(query_features)
            
            # 重排序（如果启用）
            if self.config.enable_reranking and len(retrieved_refs) > self.config.reference_count:
                retrieved_refs = self._rerank_results(text, retrieved_refs)
            
            # 过滤低相似度结果
            filtered_refs = [
                ref for ref in retrieved_refs 
                if ref['similarity'] >= self.config.similarity_threshold
            ]
            
            # 限制返回数量
            final_refs = filtered_refs[:self.config.reference_count]
            
            # 更新统计
            retrieval_time = time.time() - start_time
            self._update_retrieval_stats(retrieval_time, final_refs)
            
            # 缓存结果
            if len(self.feature_cache) < self.config.cache_size:
                self.feature_cache[cache_key] = final_refs
            
            self.retrieval_stats['total_queries'] += 1
            if final_refs:
                self.retrieval_stats['successful_retrievals'] += 1
            
            logger.debug(f"为文本 '{text[:50]}...' 检索到 {len(final_refs)} 个参考")
            
            return final_refs
            
        except Exception as e:
            logger.error(f"检索参考图像时出错: {e}")
            return []
    
    def _encode_text(self, text: str) -> np.ndarray:
        """编码文本为特征向量"""
        with torch.no_grad():
            text_features = self.clip_model.encode_text([text])
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().astype(np.float32)
    
    def _faiss_retrieve(self, query_features: np.ndarray) -> List[Dict[str, Any]]:
        """使用FAISS检索"""
        # 搜索最相似的项目
        search_k = min(self.config.rerank_top_k if self.config.enable_reranking else self.config.reference_count,
                      self.reference_features.shape[0])
        
        similarities, indices = self.faiss_index.search(query_features, search_k)
        
        # 构建结果
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0:  # 有效索引
                ref_info = {
                    'index': int(idx),
                    'similarity': float(sim),
                    'metadata': self.reference_metadata[idx] if idx < len(self.reference_metadata) else {},
                    'features': self.reference_features[idx]
                }
                results.append(ref_info)
        
        return results
    
    def _numpy_retrieve(self, query_features: np.ndarray) -> List[Dict[str, Any]]:
        """使用numpy检索"""
        # 计算相似度
        similarities = np.dot(self.reference_features, query_features.T).flatten()
        
        # 获取top-k索引
        search_k = min(self.config.rerank_top_k if self.config.enable_reranking else self.config.reference_count,
                      len(similarities))
        top_indices = np.argpartition(similarities, -search_k)[-search_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # 构建结果
        results = []
        for idx in top_indices:
            ref_info = {
                'index': int(idx),
                'similarity': float(similarities[idx]),
                'metadata': self.reference_metadata[idx] if idx < len(self.reference_metadata) else {},
                'features': self.reference_features[idx]
            }
            results.append(ref_info)
        
        return results
    
    def _rerank_results(self, text: str, initial_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重排序检索结果"""
        # 这里可以实现更复杂的重排序逻辑
        # 例如：考虑文本长度匹配、语义多样性等
        
        # 简单实现：按相似度排序
        reranked = sorted(initial_results, key=lambda x: x['similarity'], reverse=True)
        return reranked
    
    def _update_retrieval_stats(self, retrieval_time: float, results: List[Dict[str, Any]]):
        """更新检索统计"""
        # 更新平均检索时间
        total_queries = self.retrieval_stats['total_queries']
        current_avg_time = self.retrieval_stats['average_retrieval_time']
        self.retrieval_stats['average_retrieval_time'] = (
            (current_avg_time * total_queries + retrieval_time) / (total_queries + 1)
        )
        
        # 更新平均相似度
        if results:
            avg_sim = np.mean([r['similarity'] for r in results])
            current_avg_sim = self.retrieval_stats['average_similarity']
            self.retrieval_stats['average_similarity'] = (
                (current_avg_sim * total_queries + avg_sim) / (total_queries + 1)
            )
    
    def batch_retrieve_references(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """批量检索参考图像
        
        Args:
            texts: 文本列表
            
        Returns:
            每个文本对应的参考图像列表
        """
        all_references = []
        
        for text in texts:
            references = self.retrieve_references(text)
            all_references.append(references)
        
        return all_references
    
    def load_reference_images(self, reference_info: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """加载参考图像
        
        Args:
            reference_info: 参考图像信息列表
            
        Returns:
            加载的图像张量列表
        """
        loaded_images = []
        
        for ref in reference_info:
            try:
                # 从元数据获取图像路径
                metadata = ref.get('metadata', {})
                image_path = metadata.get('image_path')
                
                if image_path and Path(image_path).exists():
                    # 加载图像
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = self.image_transform(image)
                    loaded_images.append(image_tensor)
                else:
                    logger.warning(f"图像路径不存在: {image_path}")
                    
            except Exception as e:
                logger.warning(f"加载参考图像失败: {e}")
                continue
        
        return loaded_images
    
    def add_reference_images(self, 
                           images: List[Union[str, Image.Image, torch.Tensor]], 
                           texts: List[str],
                           metadata_list: Optional[List[Dict[str, Any]]] = None) -> bool:
        """添加参考图像到数据库
        
        Args:
            images: 图像列表（路径、PIL图像或张量）
            texts: 对应的文本描述列表
            metadata_list: 元数据列表（可选）
            
        Returns:
            是否成功添加
        """
        try:
            if len(images) != len(texts):
                raise ValueError("图像和文本数量不匹配")
            
            new_features = []
            new_metadata = []
            
            for i, (image, text) in enumerate(zip(images, texts)):
                # 处理图像
                if isinstance(image, str):
                    # 图像路径
                    pil_image = Image.open(image).convert('RGB')
                    image_tensor = self.image_transform(pil_image)
                elif isinstance(image, Image.Image):
                    # PIL图像
                    image_tensor = self.image_transform(image)
                elif isinstance(image, torch.Tensor):
                    # 张量
                    image_tensor = image
                else:
                    raise ValueError(f"不支持的图像类型: {type(image)}")
                
                # 编码图像特征
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_tensor.unsqueeze(0))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    new_features.append(image_features.cpu().numpy().astype(np.float32))
                
                # 构建元数据
                metadata = {
                    'text': text,
                    'image_path': str(image) if isinstance(image, str) else None,
                    'index': len(self.reference_metadata) + i
                }
                
                if metadata_list and i < len(metadata_list):
                    metadata.update(metadata_list[i])
                
                new_metadata.append(metadata)
            
            # 更新数据库
            if self.reference_features.shape[0] == 0:
                self.reference_features = np.vstack(new_features)
            else:
                self.reference_features = np.vstack([self.reference_features] + new_features)
            
            self.reference_metadata.extend(new_metadata)
            
            # 重新初始化FAISS索引
            if self.config.use_faiss:
                self._initialize_faiss_index()
            
            # 保存数据库
            self._save_reference_database()
            
            logger.info(f"成功添加 {len(images)} 个参考图像")
            return True
            
        except Exception as e:
            logger.error(f"添加参考图像失败: {e}")
            return False
    
    def _save_reference_database(self):
        """保存参考数据库"""
        try:
            # 保存特征
            features_path = self.reference_db_path / "features.npy"
            np.save(features_path, self.reference_features)
            
            # 保存元数据
            metadata_path = self.reference_db_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.reference_metadata, f, ensure_ascii=False, indent=2)
            
            logger.debug("参考数据库已保存")
            
        except Exception as e:
            logger.error(f"保存参考数据库失败: {e}")
    
    def build_reference_database(self, 
                               dataset_loader,
                               max_samples: Optional[int] = None,
                               save_interval: int = 1000) -> bool:
        """从数据集构建参考数据库
        
        Args:
            dataset_loader: 数据集加载器
            max_samples: 最大样本数（可选）
            save_interval: 保存间隔
            
        Returns:
            是否成功构建
        """
        try:
            logger.info("开始构建参考数据库...")
            
            all_features = []
            all_metadata = []
            processed_count = 0
            
            for batch_idx, batch in enumerate(dataset_loader):
                images, texts = batch['images'], batch['texts']
                
                for image, text in zip(images, texts):
                    if max_samples and processed_count >= max_samples:
                        break
                    
                    try:
                        # 编码图像特征
                        with torch.no_grad():
                            if isinstance(image, torch.Tensor):
                                image_features = self.clip_model.encode_image(image.unsqueeze(0))
                            else:
                                image_tensor = self.image_transform(image)
                                image_features = self.clip_model.encode_image(image_tensor.unsqueeze(0))
                            
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            all_features.append(image_features.cpu().numpy().astype(np.float32))
                        
                        # 构建元数据
                        metadata = {
                            'text': text,
                            'index': processed_count,
                            'batch_idx': batch_idx
                        }
                        all_metadata.append(metadata)
                        
                        processed_count += 1
                        
                        # 定期保存
                        if processed_count % save_interval == 0:
                            logger.info(f"已处理 {processed_count} 个样本")
                    
                    except Exception as e:
                        logger.warning(f"处理样本失败: {e}")
                        continue
                
                if max_samples and processed_count >= max_samples:
                    break
            
            # 更新数据库
            if all_features:
                self.reference_features = np.vstack(all_features)
                self.reference_metadata = all_metadata
                
                # 初始化FAISS索引
                if self.config.use_faiss:
                    self._initialize_faiss_index()
                
                # 保存数据库
                self._save_reference_database()
                
                logger.info(f"参考数据库构建完成，共 {len(all_features)} 个样本")
                return True
            else:
                logger.warning("未能提取任何特征")
                return False
                
        except Exception as e:
            logger.error(f"构建参考数据库失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        stats = self.retrieval_stats.copy()
        
        # 添加数据库信息
        stats['database_info'] = {
            'total_references': self.reference_features.shape[0] if self.reference_features is not None else 0,
            'feature_dimension': self.reference_features.shape[1] if self.reference_features is not None else 0,
            'use_faiss': self.config.use_faiss,
            'faiss_index_type': self.config.faiss_index_type if self.config.use_faiss else None
        }
        
        # 计算成功率
        if stats['total_queries'] > 0:
            stats['success_rate'] = stats['successful_retrievals'] / stats['total_queries']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # 添加配置信息
        stats['config'] = {
            'reference_count': self.config.reference_count,
            'similarity_threshold': self.config.similarity_threshold,
            'cache_size': self.config.cache_size,
            'enable_reranking': self.config.enable_reranking
        }
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.retrieval_stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'cache_hits': 0,
            'average_retrieval_time': 0.0,
            'average_similarity': 0.0
        }
        logger.info("检索器统计信息已重置")
    
    def clear_cache(self):
        """清空缓存"""
        self.feature_cache.clear()
        logger.info("检索缓存已清空")
    
    def update_config(self, new_config: RetrievalConfig):
        """更新配置"""
        old_use_faiss = self.config.use_faiss
        self.config = new_config
        
        # 如果FAISS设置改变，重新初始化索引
        if old_use_faiss != new_config.use_faiss:
            if new_config.use_faiss:
                self._initialize_faiss_index()
            else:
                self.faiss_index = None
        
        logger.info(f"检索器配置已更新: {new_config}")

def create_retrieval_reference_generator(config: Dict[str, Any], 
                                       clip_model,
                                       reference_db_path: str) -> RetrievalReferenceGenerator:
    """创建检索参考生成器
    
    Args:
        config: 配置字典
        clip_model: CLIP模型
        reference_db_path: 参考数据库路径
        
    Returns:
        检索参考生成器实例
    """
    retrieval_config = RetrievalConfig(
        reference_count=config.get('reference_count', 5),
        similarity_threshold=config.get('similarity_threshold', 0.3),
        use_faiss=config.get('use_faiss', True),
        faiss_index_type=config.get('faiss_index_type', 'IVF'),
        nlist=config.get('nlist', 100),
        nprobe=config.get('nprobe', 10),
        device=config.get('device', 'cuda'),
        cache_size=config.get('cache_size', 1000),
        enable_reranking=config.get('enable_reranking', True),
        rerank_top_k=config.get('rerank_top_k', 20)
    )
    
    return RetrievalReferenceGenerator(
        clip_model=clip_model,
        reference_db_path=reference_db_path,
        config=retrieval_config
    )