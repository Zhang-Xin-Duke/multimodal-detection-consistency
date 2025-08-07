"""评估指标模块

提供相似度、检索、检测指标和ROC曲线等评估功能。
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """通用指标结果"""
    name: str
    value: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'metadata': self.metadata
        }


@dataclass
class DetectionMetrics:
    """检测指标结果"""
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    fpr_at_95_tpr: float  # 95% TPR时的FPR
    threshold: float  # 最优阈值
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'auc': self.auc,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'fpr_at_95_tpr': self.fpr_at_95_tpr,
            'threshold': self.threshold,
            'confusion_matrix': self.confusion_matrix.tolist()
        }


@dataclass
class RetrievalMetrics:
    """检索指标结果"""
    recall_at_k: Dict[int, float]  # Recall@K
    precision_at_k: Dict[int, float]  # Precision@K
    map_score: float  # Mean Average Precision
    ndcg_at_k: Dict[int, float]  # NDCG@K
    mrr: float  # Mean Reciprocal Rank
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'recall_at_k': self.recall_at_k,
            'precision_at_k': self.precision_at_k,
            'map_score': self.map_score,
            'ndcg_at_k': self.ndcg_at_k,
            'mrr': self.mrr
        }


@dataclass
class SimilarityMetrics:
    """相似度指标结果"""
    cosine_similarity: float = 0.0
    euclidean_distance: float = 0.0
    manhattan_distance: float = 0.0
    pearson_correlation: float = 0.0
    spearman_correlation: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'cosine_similarity': self.cosine_similarity,
            'euclidean_distance': self.euclidean_distance,
            'manhattan_distance': self.manhattan_distance,
            'pearson_correlation': self.pearson_correlation,
            'spearman_correlation': self.spearman_correlation
        }


class SimilarityCalculator:
    """相似度计算器
    
    提供多种相似度和距离度量方法。
    """
    
    @staticmethod
    def cosine_similarity(x: Union[np.ndarray, torch.Tensor, list], 
                         y: Union[np.ndarray, torch.Tensor, list]) -> float:
        """计算余弦相似度
        
        Args:
            x: 第一个向量
            y: 第二个向量
            
        Returns:
            余弦相似度值
        """
        # 转换为numpy数组
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, list):
            x = np.array(x)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        elif isinstance(y, list):
            y = np.array(y)
        
        # 处理零向量
        if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
            return 0.0
        
        return 1 - cosine(x.flatten(), y.flatten())
    
    @staticmethod
    def batch_cosine_similarity(x: Union[np.ndarray, torch.Tensor], 
                               y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """批量计算余弦相似度
        
        Args:
            x: 第一组向量 (N, D)
            y: 第二组向量 (M, D)
            
        Returns:
            相似度矩阵 (N, M)
        """
        if isinstance(x, torch.Tensor):
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(y, p=2, dim=1)
            similarity = torch.mm(x, y.t())
            return similarity.detach().cpu().numpy()
        else:
            # NumPy实现
            x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
            y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
            return np.dot(x_norm, y_norm.T)
    
    @staticmethod
    def euclidean_distance(x: Union[np.ndarray, torch.Tensor, list], 
                          y: Union[np.ndarray, torch.Tensor, list]) -> float:
        """计算欧几里得距离
        
        Args:
            x: 第一个向量
            y: 第二个向量
            
        Returns:
            欧几里得距离
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, list):
            x = np.array(x)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        elif isinstance(y, list):
            y = np.array(y)
        
        return np.linalg.norm(x.flatten() - y.flatten())
    
    @staticmethod
    def manhattan_distance(x: Union[np.ndarray, torch.Tensor, list], 
                          y: Union[np.ndarray, torch.Tensor, list]) -> float:
        """计算曼哈顿距离
        
        Args:
            x: 第一个向量
            y: 第二个向量
            
        Returns:
            曼哈顿距离
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, list):
            x = np.array(x)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        elif isinstance(y, list):
            y = np.array(y)
        
        return np.sum(np.abs(x.flatten() - y.flatten()))
    
    @staticmethod
    def pearson_correlation(x: Union[np.ndarray, torch.Tensor], 
                           y: Union[np.ndarray, torch.Tensor]) -> float:
        """计算皮尔逊相关系数
        
        Args:
            x: 第一个向量
            y: 第二个向量
            
        Returns:
            皮尔逊相关系数
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        try:
            corr, _ = pearsonr(x.flatten(), y.flatten())
            return corr if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def spearman_correlation(x: Union[np.ndarray, torch.Tensor], 
                            y: Union[np.ndarray, torch.Tensor]) -> float:
        """计算斯皮尔曼相关系数
        
        Args:
            x: 第一个向量
            y: 第二个向量
            
        Returns:
            斯皮尔曼相关系数
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        try:
            corr, _ = spearmanr(x.flatten(), y.flatten())
            return corr if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
    
    @classmethod
    def compute_all_similarities(cls, x: Union[np.ndarray, torch.Tensor], 
                                y: Union[np.ndarray, torch.Tensor]) -> SimilarityMetrics:
        """计算所有相似度指标
        
        Args:
            x: 第一个向量
            y: 第二个向量
            
        Returns:
            相似度指标结果
        """
        return SimilarityMetrics(
            cosine_similarity=cls.cosine_similarity(x, y),
            euclidean_distance=cls.euclidean_distance(x, y),
            manhattan_distance=cls.manhattan_distance(x, y),
            pearson_correlation=cls.pearson_correlation(x, y),
            spearman_correlation=cls.spearman_correlation(x, y)
        )


class DetectionEvaluator:
    """检测评估器
    
    提供对抗样本检测的评估指标。
    """
    
    @staticmethod
    def compute_detection_metrics(scores: np.ndarray, labels: np.ndarray, 
                                 pos_label: int = 1) -> DetectionMetrics:
        """计算检测指标
        
        Args:
            scores: 检测分数（越高越可能是对抗样本）
            labels: 真实标签（1表示对抗样本，0表示正常样本）
            pos_label: 正类标签
            
        Returns:
            检测指标结果
        """
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
        auc = roc_auc_score(labels, scores)
        
        # 找到最优阈值（Youden's J statistic）
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # 根据最优阈值计算预测结果
        predictions = (scores >= optimal_threshold).astype(int)
        
        # 计算基本指标
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, pos_label=pos_label, zero_division=0)
        recall = recall_score(labels, predictions, pos_label=pos_label, zero_division=0)
        f1 = f1_score(labels, predictions, pos_label=pos_label, zero_division=0)
        cm = confusion_matrix(labels, predictions)
        
        # 计算95% TPR时的FPR
        fpr_at_95_tpr = DetectionEvaluator._compute_fpr_at_tpr(fpr, tpr, target_tpr=0.95)
        
        return DetectionMetrics(
            auc=auc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            fpr_at_95_tpr=fpr_at_95_tpr,
            threshold=optimal_threshold,
            confusion_matrix=cm
        )
    
    @staticmethod
    def _compute_fpr_at_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float = 0.95) -> float:
        """计算指定TPR下的FPR
        
        Args:
            fpr: False Positive Rate数组
            tpr: True Positive Rate数组
            target_tpr: 目标TPR
            
        Returns:
            对应的FPR值
        """
        # 找到最接近目标TPR的索引
        idx = np.argmin(np.abs(tpr - target_tpr))
        return fpr[idx]
    
    @staticmethod
    def compute_roc_curve(scores: np.ndarray, labels: np.ndarray, 
                         pos_label: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算ROC曲线
        
        Args:
            scores: 检测分数
            labels: 真实标签
            pos_label: 正类标签
            
        Returns:
            (fpr, tpr, thresholds)
        """
        return roc_curve(labels, scores, pos_label=pos_label)
    
    @staticmethod
    def compute_pr_curve(scores: np.ndarray, labels: np.ndarray, 
                        pos_label: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算Precision-Recall曲线
        
        Args:
            scores: 检测分数
            labels: 真实标签
            pos_label: 正类标签
            
        Returns:
            (precision, recall, thresholds)
        """
        precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=pos_label)
        return precision, recall, thresholds


class RetrievalEvaluator:
    """检索评估器
    
    提供信息检索的评估指标。
    """
    
    @staticmethod
    def compute_retrieval_metrics(similarities: np.ndarray, relevance: np.ndarray, 
                                 k_values: List[int] = [1, 5, 10, 20, 50]) -> RetrievalMetrics:
        """计算检索指标
        
        Args:
            similarities: 相似度矩阵 (N_queries, N_candidates)
            relevance: 相关性矩阵 (N_queries, N_candidates)，1表示相关，0表示不相关
            k_values: 计算Top-K指标的K值列表
            
        Returns:
            检索指标结果
        """
        n_queries = similarities.shape[0]
        
        # 获取排序后的索引
        sorted_indices = np.argsort(-similarities, axis=1)  # 降序排列
        
        # 计算各种指标
        recall_at_k = {}
        precision_at_k = {}
        ndcg_at_k = {}
        
        ap_scores = []
        rr_scores = []
        
        for i in range(n_queries):
            query_relevance = relevance[i]
            query_sorted_indices = sorted_indices[i]
            query_sorted_relevance = query_relevance[query_sorted_indices]
            
            # 计算AP (Average Precision)
            ap = RetrievalEvaluator._compute_average_precision(query_sorted_relevance)
            ap_scores.append(ap)
            
            # 计算RR (Reciprocal Rank)
            rr = RetrievalEvaluator._compute_reciprocal_rank(query_sorted_relevance)
            rr_scores.append(rr)
            
            # 计算各个K值的指标
            for k in k_values:
                if k not in recall_at_k:
                    recall_at_k[k] = []
                    precision_at_k[k] = []
                    ndcg_at_k[k] = []
                
                # Recall@K
                recall_k = RetrievalEvaluator._compute_recall_at_k(query_relevance, query_sorted_indices, k)
                recall_at_k[k].append(recall_k)
                
                # Precision@K
                precision_k = RetrievalEvaluator._compute_precision_at_k(query_sorted_relevance, k)
                precision_at_k[k].append(precision_k)
                
                # NDCG@K
                ndcg_k = RetrievalEvaluator._compute_ndcg_at_k(query_sorted_relevance, k)
                ndcg_at_k[k].append(ndcg_k)
        
        # 计算平均值
        recall_at_k = {k: np.mean(scores) for k, scores in recall_at_k.items()}
        precision_at_k = {k: np.mean(scores) for k, scores in precision_at_k.items()}
        ndcg_at_k = {k: np.mean(scores) for k, scores in ndcg_at_k.items()}
        
        map_score = np.mean(ap_scores)
        mrr = np.mean(rr_scores)
        
        return RetrievalMetrics(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            map_score=map_score,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr
        )
    
    @staticmethod
    def _compute_average_precision(relevance: np.ndarray) -> float:
        """计算平均精度
        
        Args:
            relevance: 排序后的相关性数组
            
        Returns:
            平均精度
        """
        if np.sum(relevance) == 0:
            return 0.0
        
        precision_scores = []
        num_relevant = 0
        
        for i, rel in enumerate(relevance):
            if rel == 1:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precision_scores.append(precision_at_i)
        
        return np.mean(precision_scores) if precision_scores else 0.0
    
    @staticmethod
    def _compute_reciprocal_rank(relevance: np.ndarray) -> float:
        """计算倒数排名
        
        Args:
            relevance: 排序后的相关性数组
            
        Returns:
            倒数排名
        """
        for i, rel in enumerate(relevance):
            if rel == 1:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def _compute_recall_at_k(relevance: np.ndarray, sorted_indices: np.ndarray, k: int) -> float:
        """计算Recall@K
        
        Args:
            relevance: 相关性数组
            sorted_indices: 排序后的索引
            k: K值
            
        Returns:
            Recall@K
        """
        total_relevant = np.sum(relevance)
        if total_relevant == 0:
            return 0.0
        
        top_k_indices = sorted_indices[:k]
        relevant_retrieved = np.sum(relevance[top_k_indices])
        
        return relevant_retrieved / total_relevant
    
    @staticmethod
    def _compute_precision_at_k(sorted_relevance: np.ndarray, k: int) -> float:
        """计算Precision@K
        
        Args:
            sorted_relevance: 排序后的相关性数组
            k: K值
            
        Returns:
            Precision@K
        """
        if k == 0:
            return 0.0
        
        top_k_relevance = sorted_relevance[:k]
        return np.sum(top_k_relevance) / k
    
    @staticmethod
    def _compute_ndcg_at_k(sorted_relevance: np.ndarray, k: int) -> float:
        """计算NDCG@K
        
        Args:
            sorted_relevance: 排序后的相关性数组
            k: K值
            
        Returns:
            NDCG@K
        """
        if k == 0:
            return 0.0
        
        # 计算DCG@K
        top_k_relevance = sorted_relevance[:k]
        dcg = RetrievalEvaluator._compute_dcg(top_k_relevance)
        
        # 计算IDCG@K
        ideal_relevance = np.sort(sorted_relevance)[::-1][:k]  # 理想排序
        idcg = RetrievalEvaluator._compute_dcg(ideal_relevance)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def _compute_dcg(relevance: np.ndarray) -> float:
        """计算DCG (Discounted Cumulative Gain)
        
        Args:
            relevance: 相关性数组
            
        Returns:
            DCG值
        """
        dcg = 0.0
        for i, rel in enumerate(relevance):
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg


class MetricsAggregator:
    """指标聚合器
    
    用于聚合和统计多次实验的指标结果。
    """
    
    def __init__(self):
        """初始化指标聚合器"""
        self.detection_metrics: List[DetectionMetrics] = []
        self.retrieval_metrics: List[RetrievalMetrics] = []
        self.similarity_metrics: List[SimilarityMetrics] = []
    
    def add_detection_metrics(self, metrics: DetectionMetrics):
        """添加检测指标
        
        Args:
            metrics: 检测指标
        """
        self.detection_metrics.append(metrics)
    
    def add_retrieval_metrics(self, metrics: RetrievalMetrics):
        """添加检索指标
        
        Args:
            metrics: 检索指标
        """
        self.retrieval_metrics.append(metrics)
    
    def add_similarity_metrics(self, metrics: SimilarityMetrics):
        """添加相似度指标
        
        Args:
            metrics: 相似度指标
        """
        self.similarity_metrics.append(metrics)
    
    def compute_detection_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算检测指标统计
        
        Returns:
            统计结果字典
        """
        if not self.detection_metrics:
            return {}
        
        stats = {}
        
        # 提取各项指标
        metrics_dict = {
            'auc': [m.auc for m in self.detection_metrics],
            'accuracy': [m.accuracy for m in self.detection_metrics],
            'precision': [m.precision for m in self.detection_metrics],
            'recall': [m.recall for m in self.detection_metrics],
            'f1_score': [m.f1_score for m in self.detection_metrics],
            'fpr_at_95_tpr': [m.fpr_at_95_tpr for m in self.detection_metrics]
        }
        
        # 计算统计量
        for metric_name, values in metrics_dict.items():
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return stats
    
    def compute_retrieval_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算检索指标统计
        
        Returns:
            统计结果字典
        """
        if not self.retrieval_metrics:
            return {}
        
        stats = {}
        
        # MAP和MRR
        map_scores = [m.map_score for m in self.retrieval_metrics]
        mrr_scores = [m.mrr for m in self.retrieval_metrics]
        
        stats['map_score'] = {
            'mean': np.mean(map_scores),
            'std': np.std(map_scores),
            'min': np.min(map_scores),
            'max': np.max(map_scores),
            'median': np.median(map_scores)
        }
        
        stats['mrr'] = {
            'mean': np.mean(mrr_scores),
            'std': np.std(mrr_scores),
            'min': np.min(mrr_scores),
            'max': np.max(mrr_scores),
            'median': np.median(mrr_scores)
        }
        
        # Recall@K和Precision@K
        if self.retrieval_metrics:
            k_values = list(self.retrieval_metrics[0].recall_at_k.keys())
            
            for k in k_values:
                recall_k_values = [m.recall_at_k[k] for m in self.retrieval_metrics]
                precision_k_values = [m.precision_at_k[k] for m in self.retrieval_metrics]
                ndcg_k_values = [m.ndcg_at_k[k] for m in self.retrieval_metrics]
                
                stats[f'recall_at_{k}'] = {
                    'mean': np.mean(recall_k_values),
                    'std': np.std(recall_k_values),
                    'min': np.min(recall_k_values),
                    'max': np.max(recall_k_values),
                    'median': np.median(recall_k_values)
                }
                
                stats[f'precision_at_{k}'] = {
                    'mean': np.mean(precision_k_values),
                    'std': np.std(precision_k_values),
                    'min': np.min(precision_k_values),
                    'max': np.max(precision_k_values),
                    'median': np.median(precision_k_values)
                }
                
                stats[f'ndcg_at_{k}'] = {
                    'mean': np.mean(ndcg_k_values),
                    'std': np.std(ndcg_k_values),
                    'min': np.min(ndcg_k_values),
                    'max': np.max(ndcg_k_values),
                    'median': np.median(ndcg_k_values)
                }
        
        return stats
    
    def compute_similarity_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算相似度指标统计
        
        Returns:
            统计结果字典
        """
        if not self.similarity_metrics:
            return {}
        
        stats = {}
        
        # 提取各项指标
        metrics_dict = {
            'cosine_similarity': [m.cosine_similarity for m in self.similarity_metrics],
            'euclidean_distance': [m.euclidean_distance for m in self.similarity_metrics],
            'manhattan_distance': [m.manhattan_distance for m in self.similarity_metrics],
            'pearson_correlation': [m.pearson_correlation for m in self.similarity_metrics],
            'spearman_correlation': [m.spearman_correlation for m in self.similarity_metrics]
        }
        
        # 计算统计量
        for metric_name, values in metrics_dict.items():
            # 过滤NaN值
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_values:
                stats[metric_name] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values),
                    'median': np.median(valid_values)
                }
            else:
                stats[metric_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0
                }
        
        return stats
    
    def print_summary(self):
        """打印指标摘要"""
        print("=== Metrics Summary ===")
        
        # 检测指标
        if self.detection_metrics:
            print("\nDetection Metrics:")
            detection_stats = self.compute_detection_statistics()
            for metric_name, stats in detection_stats.items():
                print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 检索指标
        if self.retrieval_metrics:
            print("\nRetrieval Metrics:")
            retrieval_stats = self.compute_retrieval_statistics()
            for metric_name, stats in retrieval_stats.items():
                print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 相似度指标
        if self.similarity_metrics:
            print("\nSimilarity Metrics:")
            similarity_stats = self.compute_similarity_statistics()
            for metric_name, stats in similarity_stats.items():
                print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print("======================")
    
    def clear(self):
        """清空所有指标"""
        self.detection_metrics.clear()
        self.retrieval_metrics.clear()
        self.similarity_metrics.clear()


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """计算置信区间
    
    Args:
        values: 数值列表
        confidence: 置信度
        
    Returns:
        (下界, 上界)
    """
    if not values:
        return 0.0, 0.0
    
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    
    # 使用t分布
    from scipy.stats import t
    alpha = 1 - confidence
    t_value = t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_value * std / np.sqrt(n)
    
    return mean - margin_error, mean + margin_error


def bootstrap_metric(metric_func, *args, n_bootstrap: int = 1000, confidence: float = 0.95, **kwargs) -> Dict[str, float]:
    """使用Bootstrap方法计算指标的置信区间
    
    Args:
        metric_func: 指标计算函数
        *args: 函数参数
        n_bootstrap: Bootstrap采样次数
        confidence: 置信度
        **kwargs: 函数关键字参数
        
    Returns:
        包含均值和置信区间的字典
    """
    # 获取数据长度
    data_length = len(args[0]) if args else 0
    if data_length == 0:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}
    
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        # Bootstrap采样
        indices = np.random.choice(data_length, size=data_length, replace=True)
        
        # 重采样数据
        bootstrap_args = []
        for arg in args:
            if hasattr(arg, '__getitem__'):
                bootstrap_args.append(arg[indices])
            else:
                bootstrap_args.append(arg)
        
        # 计算指标
        try:
            metric_value = metric_func(*bootstrap_args, **kwargs)
            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                bootstrap_values.append(metric_value)
        except Exception as e:
            logger.warning(f"Bootstrap sampling failed: {e}")
            continue
    
    if not bootstrap_values:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}
    
    # 计算统计量
    mean_value = np.mean(bootstrap_values)
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_values, lower_percentile)
    upper_bound = np.percentile(bootstrap_values, upper_percentile)
    
    return {
        'mean': mean_value,
        'lower': lower_bound,
        'upper': upper_bound,
        'std': np.std(bootstrap_values)
    }


class MetricsCalculator:
    """指标计算器
    
    提供统一的指标计算接口，整合各种评估指标。
    """
    
    def __init__(self):
        """初始化指标计算器"""
        self.similarity_calculator = SimilarityCalculator()
        self.detection_evaluator = DetectionEvaluator()
        
    def calculate_all_metrics(self, 
                            predictions: np.ndarray,
                            labels: np.ndarray,
                            embeddings1: Optional[np.ndarray] = None,
                            embeddings2: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """计算所有指标
        
        Args:
            predictions: 预测分数
            labels: 真实标签
            embeddings1: 第一组嵌入向量（可选）
            embeddings2: 第二组嵌入向量（可选）
            
        Returns:
            Dict[str, Any]: 所有指标结果
        """
        results = {}
        
        # 检测指标
        detection_metrics = self.detection_evaluator.evaluate_detection(
            predictions, labels
        )
        results['detection'] = detection_metrics.to_dict()
        
        # 相似度指标（如果提供了嵌入向量）
        if embeddings1 is not None and embeddings2 is not None:
            similarity_results = self.similarity_calculator.compute_all_similarities(
                embeddings1, embeddings2
            ).to_dict()
            results['similarity'] = similarity_results
            
        return results
    
    def calculate_detection_metrics(self, 
                                  predictions: np.ndarray,
                                  labels: np.ndarray) -> DetectionMetrics:
        """计算检测指标
        
        Args:
            predictions: 预测分数
            labels: 真实标签
            
        Returns:
            DetectionMetrics: 检测指标结果
        """
        return self.detection_evaluator.evaluate_detection(predictions, labels)
    
    def calculate_similarity_metrics(self, 
                                   embeddings1: np.ndarray,
                                   embeddings2: np.ndarray) -> Dict[str, float]:
        """计算相似度指标
        
        Args:
            embeddings1: 第一组嵌入向量
            embeddings2: 第二组嵌入向量
            
        Returns:
            Dict[str, float]: 相似度指标结果
        """
        return self.similarity_calculator.compute_all_similarities(
            embeddings1, embeddings2
        ).to_dict()
    
    def calculate_retrieval_metrics(self, 
                                  query_embeddings: np.ndarray,
                                  gallery_embeddings: np.ndarray,
                                  query_labels: np.ndarray,
                                  gallery_labels: np.ndarray,
                                  k_values: List[int] = [1, 5, 10]) -> RetrievalMetrics:
        """计算检索指标
        
        Args:
            query_embeddings: 查询嵌入向量
            gallery_embeddings: 画廊嵌入向量
            query_labels: 查询标签
            gallery_labels: 画廊标签
            k_values: Top-K值列表
            
        Returns:
            RetrievalMetrics: 检索指标结果
        """
        return self.detection_evaluator.evaluate_retrieval(
            query_embeddings, gallery_embeddings,
            query_labels, gallery_labels, k_values
        )
    
    def get_summary_report(self, 
                          predictions: np.ndarray,
                          labels: np.ndarray,
                          embeddings1: Optional[np.ndarray] = None,
                          embeddings2: Optional[np.ndarray] = None) -> str:
        """生成汇总报告
        
        Args:
            predictions: 预测分数
            labels: 真实标签
            embeddings1: 第一组嵌入向量（可选）
            embeddings2: 第二组嵌入向量（可选）
            
        Returns:
            str: 格式化的汇总报告
        """
        metrics = self.calculate_all_metrics(
            predictions, labels, embeddings1, embeddings2
        )
        
        report = ["\n=== 指标汇总报告 ==="]
        
        # 检测指标
        if 'detection' in metrics:
            det = metrics['detection']
            report.append("\n检测指标:")
            report.append(f"  AUC: {det['auc']:.4f}")
            report.append(f"  准确率: {det['accuracy']:.4f}")
            report.append(f"  精确率: {det['precision']:.4f}")
            report.append(f"  召回率: {det['recall']:.4f}")
            report.append(f"  F1分数: {det['f1_score']:.4f}")
            report.append(f"  FPR@95%TPR: {det['fpr_at_95_tpr']:.4f}")
        
        # 相似度指标
        if 'similarity' in metrics:
            sim = metrics['similarity']
            report.append("\n相似度指标:")
            for metric_name, value in sim.items():
                if isinstance(value, (int, float)):
                    report.append(f"  {metric_name}: {value:.4f}")
        
        report.append("\n====================")
        return "\n".join(report)