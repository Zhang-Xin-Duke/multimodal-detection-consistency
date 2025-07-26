"""评估指标计算模块

提供各种评估指标的计算功能，包括检索、检测和分类指标。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.metrics.pairwise import cosine_similarity
import torch

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """指标结果"""
    name: str
    value: float
    description: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalMetrics:
    """检索指标结果"""
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    map_score: float
    mrr_score: float
    ndcg_at_k: Dict[int, float]


@dataclass
class DetectionMetrics:
    """检测指标结果"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    ap_score: float
    confusion_matrix: np.ndarray
    threshold: float


class RetrievalEvaluator:
    """检索评估器"""
    
    def __init__(self, k_values: List[int] = None):
        """
        初始化检索评估器
        
        Args:
            k_values: 评估的k值列表
        """
        self.k_values = k_values or [1, 5, 10, 20, 50]
    
    def compute_recall_at_k(self, 
                           retrieved_indices: np.ndarray,
                           relevant_indices: List[List[int]]) -> Dict[int, float]:
        """
        计算Recall@K
        
        Args:
            retrieved_indices: 检索结果索引 (N, max_k)
            relevant_indices: 相关项索引列表
            
        Returns:
            各k值的Recall@K
        """
        recall_at_k = {}
        
        for k in self.k_values:
            if k > retrieved_indices.shape[1]:
                continue
                
            total_recall = 0.0
            valid_queries = 0
            
            for i, relevant in enumerate(relevant_indices):
                if len(relevant) == 0:
                    continue
                    
                retrieved_k = retrieved_indices[i, :k]
                relevant_set = set(relevant)
                retrieved_set = set(retrieved_k)
                
                intersection = len(relevant_set & retrieved_set)
                recall = intersection / len(relevant_set)
                
                total_recall += recall
                valid_queries += 1
            
            if valid_queries > 0:
                recall_at_k[k] = total_recall / valid_queries
            else:
                recall_at_k[k] = 0.0
        
        return recall_at_k
    
    def compute_precision_at_k(self,
                              retrieved_indices: np.ndarray,
                              relevant_indices: List[List[int]]) -> Dict[int, float]:
        """
        计算Precision@K
        
        Args:
            retrieved_indices: 检索结果索引 (N, max_k)
            relevant_indices: 相关项索引列表
            
        Returns:
            各k值的Precision@K
        """
        precision_at_k = {}
        
        for k in self.k_values:
            if k > retrieved_indices.shape[1]:
                continue
                
            total_precision = 0.0
            valid_queries = 0
            
            for i, relevant in enumerate(relevant_indices):
                if len(relevant) == 0:
                    continue
                    
                retrieved_k = retrieved_indices[i, :k]
                relevant_set = set(relevant)
                retrieved_set = set(retrieved_k)
                
                intersection = len(relevant_set & retrieved_set)
                precision = intersection / k if k > 0 else 0.0
                
                total_precision += precision
                valid_queries += 1
            
            if valid_queries > 0:
                precision_at_k[k] = total_precision / valid_queries
            else:
                precision_at_k[k] = 0.0
        
        return precision_at_k
    
    def compute_map(self,
                   retrieved_indices: np.ndarray,
                   relevant_indices: List[List[int]]) -> float:
        """
        计算Mean Average Precision (MAP)
        
        Args:
            retrieved_indices: 检索结果索引 (N, max_k)
            relevant_indices: 相关项索引列表
            
        Returns:
            MAP分数
        """
        total_ap = 0.0
        valid_queries = 0
        
        for i, relevant in enumerate(relevant_indices):
            if len(relevant) == 0:
                continue
                
            retrieved = retrieved_indices[i]
            relevant_set = set(relevant)
            
            # 计算Average Precision
            ap = 0.0
            relevant_count = 0
            
            for j, item in enumerate(retrieved):
                if item in relevant_set:
                    relevant_count += 1
                    precision_at_j = relevant_count / (j + 1)
                    ap += precision_at_j
            
            if len(relevant) > 0:
                ap /= len(relevant)
                total_ap += ap
                valid_queries += 1
        
        return total_ap / valid_queries if valid_queries > 0 else 0.0
    
    def compute_mrr(self,
                   retrieved_indices: np.ndarray,
                   relevant_indices: List[List[int]]) -> float:
        """
        计算Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_indices: 检索结果索引 (N, max_k)
            relevant_indices: 相关项索引列表
            
        Returns:
            MRR分数
        """
        total_rr = 0.0
        valid_queries = 0
        
        for i, relevant in enumerate(relevant_indices):
            if len(relevant) == 0:
                continue
                
            retrieved = retrieved_indices[i]
            relevant_set = set(relevant)
            
            # 找到第一个相关项的位置
            for j, item in enumerate(retrieved):
                if item in relevant_set:
                    total_rr += 1.0 / (j + 1)
                    break
            
            valid_queries += 1
        
        return total_rr / valid_queries if valid_queries > 0 else 0.0
    
    def compute_ndcg_at_k(self,
                         retrieved_indices: np.ndarray,
                         relevant_indices: List[List[int]],
                         relevance_scores: Optional[List[List[float]]] = None) -> Dict[int, float]:
        """
        计算Normalized Discounted Cumulative Gain (NDCG@K)
        
        Args:
            retrieved_indices: 检索结果索引 (N, max_k)
            relevant_indices: 相关项索引列表
            relevance_scores: 相关性分数列表（可选）
            
        Returns:
            各k值的NDCG@K
        """
        ndcg_at_k = {}
        
        for k in self.k_values:
            if k > retrieved_indices.shape[1]:
                continue
                
            total_ndcg = 0.0
            valid_queries = 0
            
            for i, relevant in enumerate(relevant_indices):
                if len(relevant) == 0:
                    continue
                    
                retrieved_k = retrieved_indices[i, :k]
                relevant_set = set(relevant)
                
                # 计算DCG
                dcg = 0.0
                for j, item in enumerate(retrieved_k):
                    if item in relevant_set:
                        # 使用二元相关性（1或0）
                        relevance = 1.0
                        if relevance_scores and i < len(relevance_scores):
                            # 如果提供了相关性分数，使用实际分数
                            item_idx = relevant.index(item) if item in relevant else -1
                            if item_idx >= 0 and item_idx < len(relevance_scores[i]):
                                relevance = relevance_scores[i][item_idx]
                        
                        dcg += relevance / np.log2(j + 2)
                
                # 计算IDCG（理想DCG）
                if relevance_scores and i < len(relevance_scores):
                    ideal_scores = sorted(relevance_scores[i], reverse=True)[:k]
                else:
                    ideal_scores = [1.0] * min(len(relevant), k)
                
                idcg = sum(score / np.log2(j + 2) for j, score in enumerate(ideal_scores))
                
                # 计算NDCG
                ndcg = dcg / idcg if idcg > 0 else 0.0
                total_ndcg += ndcg
                valid_queries += 1
            
            if valid_queries > 0:
                ndcg_at_k[k] = total_ndcg / valid_queries
            else:
                ndcg_at_k[k] = 0.0
        
        return ndcg_at_k
    
    def evaluate(self,
                retrieved_indices: np.ndarray,
                relevant_indices: List[List[int]],
                relevance_scores: Optional[List[List[float]]] = None) -> RetrievalMetrics:
        """
        综合评估检索性能
        
        Args:
            retrieved_indices: 检索结果索引
            relevant_indices: 相关项索引列表
            relevance_scores: 相关性分数列表（可选）
            
        Returns:
            检索指标结果
        """
        recall_at_k = self.compute_recall_at_k(retrieved_indices, relevant_indices)
        precision_at_k = self.compute_precision_at_k(retrieved_indices, relevant_indices)
        map_score = self.compute_map(retrieved_indices, relevant_indices)
        mrr_score = self.compute_mrr(retrieved_indices, relevant_indices)
        ndcg_at_k = self.compute_ndcg_at_k(retrieved_indices, relevant_indices, relevance_scores)
        
        return RetrievalMetrics(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            map_score=map_score,
            mrr_score=mrr_score,
            ndcg_at_k=ndcg_at_k
        )


class DetectionEvaluator:
    """检测评估器"""
    
    def __init__(self, threshold: float = 0.5):
        """
        初始化检测评估器
        
        Args:
            threshold: 检测阈值
        """
        self.threshold = threshold
    
    def compute_binary_metrics(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_scores: Optional[np.ndarray] = None) -> DetectionMetrics:
        """
        计算二分类检测指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_scores: 预测分数（可选）
            
        Returns:
            检测指标结果
        """
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # AUC和AP（需要预测分数）
        auc_score = 0.0
        ap_score = 0.0
        
        if y_scores is not None:
            try:
                auc_score = roc_auc_score(y_true, y_scores)
                ap_score = average_precision_score(y_true, y_scores)
            except ValueError as e:
                logger.warning(f"计算AUC/AP失败: {e}")
        
        return DetectionMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc_score,
            ap_score=ap_score,
            confusion_matrix=cm,
            threshold=self.threshold
        )
    
    def find_optimal_threshold(self,
                              y_true: np.ndarray,
                              y_scores: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, float]:
        """
        寻找最优阈值
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            metric: 优化指标 ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            (最优阈值, 最优指标值)
        """
        thresholds = np.linspace(0, 1, 101)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            else:
                raise ValueError(f"不支持的指标: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def evaluate_with_threshold_search(self,
                                      y_true: np.ndarray,
                                      y_scores: np.ndarray,
                                      metric: str = 'f1') -> DetectionMetrics:
        """
        使用阈值搜索进行评估
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            metric: 优化指标
            
        Returns:
            检测指标结果
        """
        # 寻找最优阈值
        optimal_threshold, _ = self.find_optimal_threshold(y_true, y_scores, metric)
        
        # 使用最优阈值进行预测
        y_pred = (y_scores >= optimal_threshold).astype(int)
        
        # 更新阈值
        self.threshold = optimal_threshold
        
        return self.compute_binary_metrics(y_true, y_pred, y_scores)


class SimilarityMetrics:
    """相似度指标"""
    
    @staticmethod
    def compute_cosine_similarity(embedding1: np.ndarray, 
                                 embedding2: np.ndarray) -> float:
        """
        计算两个嵌入向量的余弦相似度
        
        Args:
            embedding1: 嵌入向量1
            embedding2: 嵌入向量2
            
        Returns:
            余弦相似度值
        """
        # 确保输入是二维数组
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
            
        similarity_matrix = cosine_similarity(embedding1, embedding2)
        return float(similarity_matrix[0, 0])
    
    @staticmethod
    def cosine_similarity_matrix(embeddings1: np.ndarray, 
                                embeddings2: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度矩阵
        
        Args:
            embeddings1: 嵌入矩阵1 (N, D)
            embeddings2: 嵌入矩阵2 (M, D)
            
        Returns:
            相似度矩阵 (N, M)
        """
        return cosine_similarity(embeddings1, embeddings2)
    
    @staticmethod
    def euclidean_distance_matrix(embeddings1: np.ndarray,
                                 embeddings2: np.ndarray) -> np.ndarray:
        """
        计算欧几里得距离矩阵
        
        Args:
            embeddings1: 嵌入矩阵1 (N, D)
            embeddings2: 嵌入矩阵2 (M, D)
            
        Returns:
            距离矩阵 (N, M)
        """
        # 使用广播计算距离
        diff = embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances
    
    @staticmethod
    def compute_consistency_score(similarities: np.ndarray,
                                 aggregation_method: str = 'mean') -> float:
        """
        计算一致性分数
        
        Args:
            similarities: 相似度数组
            aggregation_method: 聚合方法 ('mean', 'median', 'min', 'max')
            
        Returns:
            一致性分数
        """
        if len(similarities) == 0:
            return 0.0
        
        if aggregation_method == 'mean':
            return np.mean(similarities)
        elif aggregation_method == 'median':
            return np.median(similarities)
        elif aggregation_method == 'min':
            return np.min(similarities)
        elif aggregation_method == 'max':
            return np.max(similarities)
        else:
            raise ValueError(f"不支持的聚合方法: {aggregation_method}")


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        """
        初始化指标计算器
        """
        self.retrieval_evaluator = RetrievalEvaluator()
        self.detection_evaluator = DetectionEvaluator()
    
    def compute_all_metrics(self,
                           retrieved_indices: Optional[np.ndarray] = None,
                           relevant_indices: Optional[List[List[int]]] = None,
                           y_true: Optional[np.ndarray] = None,
                           y_pred: Optional[np.ndarray] = None,
                           y_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算所有指标
        
        Args:
            retrieved_indices: 检索结果索引
            relevant_indices: 相关项索引列表
            y_true: 真实标签
            y_pred: 预测标签
            y_scores: 预测分数
            
        Returns:
            所有指标的字典
        """
        results = {}
        
        # 检索指标
        if retrieved_indices is not None and relevant_indices is not None:
            retrieval_metrics = self.retrieval_evaluator.evaluate(
                retrieved_indices, relevant_indices
            )
            results['retrieval'] = retrieval_metrics
        
        # 检测指标
        if y_true is not None and y_pred is not None:
            detection_metrics = self.detection_evaluator.compute_binary_metrics(
                y_true, y_pred, y_scores
            )
            results['detection'] = detection_metrics
        
        return results
    
    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """
        格式化指标报告
        
        Args:
            metrics: 指标字典
            
        Returns:
            格式化的报告字符串
        """
        report = ["\n=== 评估指标报告 ==="]
        
        # 检索指标
        if 'retrieval' in metrics:
            retrieval = metrics['retrieval']
            report.append("\n--- 检索指标 ---")
            report.append(f"MAP: {retrieval.map_score:.4f}")
            report.append(f"MRR: {retrieval.mrr_score:.4f}")
            
            report.append("\nRecall@K:")
            for k, score in retrieval.recall_at_k.items():
                report.append(f"  R@{k}: {score:.4f}")
            
            report.append("\nPrecision@K:")
            for k, score in retrieval.precision_at_k.items():
                report.append(f"  P@{k}: {score:.4f}")
            
            report.append("\nNDCG@K:")
            for k, score in retrieval.ndcg_at_k.items():
                report.append(f"  NDCG@{k}: {score:.4f}")
        
        # 检测指标
        if 'detection' in metrics:
            detection = metrics['detection']
            report.append("\n--- 检测指标 ---")
            report.append(f"Accuracy: {detection.accuracy:.4f}")
            report.append(f"Precision: {detection.precision:.4f}")
            report.append(f"Recall: {detection.recall:.4f}")
            report.append(f"F1-Score: {detection.f1_score:.4f}")
            report.append(f"AUC: {detection.auc_score:.4f}")
            report.append(f"AP: {detection.ap_score:.4f}")
            report.append(f"Threshold: {detection.threshold:.4f}")
        
        return "\n".join(report)


def create_metrics_calculator() -> MetricsCalculator:
    """
    创建指标计算器
    
    Returns:
        指标计算器实例
    """
    return MetricsCalculator()