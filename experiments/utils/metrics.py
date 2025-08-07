"""指标计算模块

提供各种评估指标的计算功能，包括：
- 攻击成功率 (ASR)
- 检测准确率、精确率、召回率、F1分数
- 检索精度指标
- ROC AUC、PR AUC
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class AttackSuccessRate:
    """攻击成功率指标"""
    total_samples: int
    successful_attacks: int
    asr: float
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.total_samples > 0:
            self.asr = self.successful_attacks / self.total_samples
        else:
            self.asr = 0.0

@dataclass
class DetectionMetrics:
    """检测性能指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: np.ndarray
    
@dataclass
class RetrievalMetrics:
    """检索性能指标"""
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mean_reciprocal_rank: float
    mean_average_precision: float
    ndcg_at_5: float
    ndcg_at_10: float

def compute_asr(predictions: List[bool], 
               ground_truth: List[bool],
               confidence_level: float = 0.95) -> AttackSuccessRate:
    """计算攻击成功率
    
    Args:
        predictions: 攻击预测结果（True表示攻击成功）
        ground_truth: 真实标签（True表示确实是对抗样本）
        confidence_level: 置信区间水平
        
    Returns:
        攻击成功率指标
    """
    predictions = np.array(predictions, dtype=bool)
    ground_truth = np.array(ground_truth, dtype=bool)
    
    # 只考虑真实的对抗样本
    adversarial_mask = ground_truth
    adversarial_predictions = predictions[adversarial_mask]
    
    total_adversarial = np.sum(adversarial_mask)
    successful_attacks = np.sum(adversarial_predictions)
    
    asr = successful_attacks / total_adversarial if total_adversarial > 0 else 0.0
    
    # 计算置信区间（使用二项分布）
    confidence_interval = None
    if total_adversarial > 0:
        from scipy import stats
        try:
            ci_lower, ci_upper = stats.binom.interval(
                confidence_level, total_adversarial, asr
            )
            confidence_interval = (
                ci_lower / total_adversarial,
                ci_upper / total_adversarial
            )
        except ImportError:
            logger.warning("scipy未安装，无法计算置信区间")
    
    return AttackSuccessRate(
        total_samples=total_adversarial,
        successful_attacks=successful_attacks,
        asr=asr,
        confidence_interval=confidence_interval
    )

def compute_detection_metrics(y_true: Union[List, np.ndarray], 
                            y_pred: Union[List, np.ndarray],
                            y_scores: Optional[Union[List, np.ndarray]] = None) -> DetectionMetrics:
    """计算检测性能指标
    
    Args:
        y_true: 真实标签（0=正常，1=对抗）
        y_pred: 预测标签
        y_scores: 预测分数（可选，用于计算AUC）
        
    Returns:
        检测性能指标
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算特异性和假阳性率
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        specificity = 0.0
        false_positive_rate = 0.0
        false_negative_rate = 0.0
    
    # AUC指标
    roc_auc = 0.0
    pr_auc = 0.0
    
    if y_scores is not None:
        y_scores = np.array(y_scores)
        try:
            # 检查是否有足够的正负样本
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, y_scores)
                pr_auc = average_precision_score(y_true, y_scores)
        except ValueError as e:
            logger.warning(f"计算AUC时出错: {e}")
    
    return DetectionMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        specificity=specificity,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm
    )

def compute_retrieval_metrics(rankings: List[List[int]], 
                            relevant_items: List[List[int]],
                            k_values: List[int] = [1, 5, 10]) -> RetrievalMetrics:
    """计算检索性能指标
    
    Args:
        rankings: 每个查询的排序结果列表
        relevant_items: 每个查询的相关项目列表
        k_values: 计算recall@k的k值列表
        
    Returns:
        检索性能指标
    """
    if len(rankings) != len(relevant_items):
        raise ValueError("rankings和relevant_items长度不匹配")
    
    # 初始化指标
    recall_at_k = {k: [] for k in k_values}
    mrr_scores = []
    map_scores = []
    ndcg_scores = {k: [] for k in [5, 10] if k in k_values or k <= max(k_values)}
    
    for ranking, relevant in zip(rankings, relevant_items):
        if not relevant:  # 跳过没有相关项目的查询
            continue
        
        relevant_set = set(relevant)
        
        # 计算Recall@k
        for k in k_values:
            top_k = set(ranking[:k])
            recall_k = len(top_k & relevant_set) / len(relevant_set)
            recall_at_k[k].append(recall_k)
        
        # 计算MRR
        mrr = 0.0
        for i, item in enumerate(ranking):
            if item in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        mrr_scores.append(mrr)
        
        # 计算MAP
        ap = 0.0
        relevant_count = 0
        for i, item in enumerate(ranking):
            if item in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
        
        if len(relevant_set) > 0:
            ap /= len(relevant_set)
        map_scores.append(ap)
        
        # 计算NDCG@k
        for k in [5, 10]:
            if k in ndcg_scores:
                ndcg_k = compute_ndcg(ranking[:k], relevant_set)
                ndcg_scores[k].append(ndcg_k)
    
    # 计算平均值
    metrics = RetrievalMetrics(
        recall_at_1=np.mean(recall_at_k[1]) if 1 in recall_at_k and recall_at_k[1] else 0.0,
        recall_at_5=np.mean(recall_at_k[5]) if 5 in recall_at_k and recall_at_k[5] else 0.0,
        recall_at_10=np.mean(recall_at_k[10]) if 10 in recall_at_k and recall_at_k[10] else 0.0,
        mean_reciprocal_rank=np.mean(mrr_scores) if mrr_scores else 0.0,
        mean_average_precision=np.mean(map_scores) if map_scores else 0.0,
        ndcg_at_5=np.mean(ndcg_scores[5]) if 5 in ndcg_scores and ndcg_scores[5] else 0.0,
        ndcg_at_10=np.mean(ndcg_scores[10]) if 10 in ndcg_scores and ndcg_scores[10] else 0.0
    )
    
    return metrics

def compute_ndcg(ranking: List[int], relevant_set: set, k: Optional[int] = None) -> float:
    """计算NDCG@k
    
    Args:
        ranking: 排序结果
        relevant_set: 相关项目集合
        k: 截断位置（可选）
        
    Returns:
        NDCG分数
    """
    if k is not None:
        ranking = ranking[:k]
    
    # 计算DCG
    dcg = 0.0
    for i, item in enumerate(ranking):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2因为log2(1)=0
    
    # 计算IDCG
    idcg = 0.0
    for i in range(min(len(relevant_set), len(ranking))):
        idcg += 1.0 / np.log2(i + 2)
    
    # 计算NDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def compute_confidence_interval(values: List[float], 
                              confidence_level: float = 0.95) -> Tuple[float, float]:
    """计算置信区间
    
    Args:
        values: 数值列表
        confidence_level: 置信水平
        
    Returns:
        置信区间下界和上界
    """
    if not values:
        return 0.0, 0.0
    
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    
    # 使用t分布
    try:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        margin_error = t_value * std / np.sqrt(n)
        return mean - margin_error, mean + margin_error
    except ImportError:
        # 如果没有scipy，使用正态分布近似
        z_value = 1.96 if confidence_level == 0.95 else 2.576  # 99%置信区间
        margin_error = z_value * std / np.sqrt(n)
        return mean - margin_error, mean + margin_error

class MetricsAggregator:
    """指标聚合器
    
    用于收集和聚合多个实验的指标结果
    """
    
    def __init__(self):
        self.attack_results = []  # 攻击成功率结果
        self.detection_results = []  # 检测性能结果
        self.retrieval_results = []  # 检索性能结果
        self.experiment_metadata = []  # 实验元数据
    
    def add_attack_result(self, 
                         asr: AttackSuccessRate, 
                         metadata: Dict[str, Any] = None):
        """添加攻击结果"""
        self.attack_results.append(asr)
        self.experiment_metadata.append(metadata or {})
    
    def add_detection_result(self, 
                           metrics: DetectionMetrics, 
                           metadata: Dict[str, Any] = None):
        """添加检测结果"""
        self.detection_results.append(metrics)
        if len(self.experiment_metadata) <= len(self.detection_results):
            self.experiment_metadata.append(metadata or {})
    
    def add_retrieval_result(self, 
                           metrics: RetrievalMetrics, 
                           metadata: Dict[str, Any] = None):
        """添加检索结果"""
        self.retrieval_results.append(metrics)
        if len(self.experiment_metadata) <= len(self.retrieval_results):
            self.experiment_metadata.append(metadata or {})
    
    def compute_summary_statistics(self) -> Dict[str, Any]:
        """计算汇总统计"""
        summary = {}
        
        # 攻击成功率统计
        if self.attack_results:
            asr_values = [result.asr for result in self.attack_results]
            summary['attack_success_rate'] = {
                'mean': np.mean(asr_values),
                'std': np.std(asr_values),
                'min': np.min(asr_values),
                'max': np.max(asr_values),
                'confidence_interval': compute_confidence_interval(asr_values)
            }
        
        # 检测性能统计
        if self.detection_results:
            detection_stats = {}
            for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']:
                values = [getattr(result, metric_name) for result in self.detection_results]
                detection_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'confidence_interval': compute_confidence_interval(values)
                }
            summary['detection_metrics'] = detection_stats
        
        # 检索性能统计
        if self.retrieval_results:
            retrieval_stats = {}
            for metric_name in ['recall_at_1', 'recall_at_5', 'recall_at_10', 
                              'mean_reciprocal_rank', 'mean_average_precision']:
                values = [getattr(result, metric_name) for result in self.retrieval_results]
                retrieval_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'confidence_interval': compute_confidence_interval(values)
                }
            summary['retrieval_metrics'] = retrieval_stats
        
        return summary
    
    def get_results_by_condition(self, condition_key: str, condition_value: Any) -> Dict[str, List]:
        """根据条件筛选结果
        
        Args:
            condition_key: 条件键（在metadata中）
            condition_value: 条件值
            
        Returns:
            筛选后的结果
        """
        filtered_results = {
            'attack_results': [],
            'detection_results': [],
            'retrieval_results': [],
            'metadata': []
        }
        
        for i, metadata in enumerate(self.experiment_metadata):
            if metadata.get(condition_key) == condition_value:
                filtered_results['metadata'].append(metadata)
                
                if i < len(self.attack_results):
                    filtered_results['attack_results'].append(self.attack_results[i])
                if i < len(self.detection_results):
                    filtered_results['detection_results'].append(self.detection_results[i])
                if i < len(self.retrieval_results):
                    filtered_results['retrieval_results'].append(self.retrieval_results[i])
        
        return filtered_results
    
    def export_to_dict(self) -> Dict[str, Any]:
        """导出为字典格式"""
        return {
            'attack_results': [{
                'total_samples': result.total_samples,
                'successful_attacks': result.successful_attacks,
                'asr': result.asr,
                'confidence_interval': result.confidence_interval
            } for result in self.attack_results],
            
            'detection_results': [{
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'specificity': result.specificity,
                'false_positive_rate': result.false_positive_rate,
                'false_negative_rate': result.false_negative_rate,
                'roc_auc': result.roc_auc,
                'pr_auc': result.pr_auc,
                'confusion_matrix': result.confusion_matrix.tolist()
            } for result in self.detection_results],
            
            'retrieval_results': [{
                'recall_at_1': result.recall_at_1,
                'recall_at_5': result.recall_at_5,
                'recall_at_10': result.recall_at_10,
                'mean_reciprocal_rank': result.mean_reciprocal_rank,
                'mean_average_precision': result.mean_average_precision,
                'ndcg_at_5': result.ndcg_at_5,
                'ndcg_at_10': result.ndcg_at_10
            } for result in self.retrieval_results],
            
            'experiment_metadata': self.experiment_metadata,
            'summary_statistics': self.compute_summary_statistics()
        }
    
    def clear(self):
        """清空所有结果"""
        self.attack_results.clear()
        self.detection_results.clear()
        self.retrieval_results.clear()
        self.experiment_metadata.clear()
        logger.info("指标聚合器已清空")