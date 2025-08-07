"""一致性检测器

该模块实现跨模态一致性检测逻辑，通过分析图像-文本、图像-图像、文本-文本
之间的一致性来判断是否存在对抗攻击。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class ConsistencyMetrics:
    """一致性指标"""
    original_similarity: float
    text_variant_consistency: float
    text_variant_std: float
    retrieval_consistency: float
    retrieval_std: float
    generative_consistency: float
    generative_std: float
    cross_modal_variance: float
    overall_score: float
    is_adversarial: bool
    confidence: float

class ConsistencyChecker:
    """一致性检测器
    
    通过多种一致性指标来检测对抗样本：
    1. 原始图文相似度
    2. 文本变体一致性
    3. 检索参考一致性
    4. 生成参考一致性
    5. 跨模态方差分析
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 adaptive_threshold: bool = True,
                 voting_strategy: str = "weighted",
                 weights: Optional[Dict[str, float]] = None):
        """
        初始化一致性检测器
        
        Args:
            threshold: 基础检测阈值
            adaptive_threshold: 是否使用自适应阈值
            voting_strategy: 投票策略 ("simple", "weighted", "adaptive")
            weights: 各指标权重
        """
        self.base_threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.voting_strategy = voting_strategy
        
        # 默认权重
        self.weights = weights or {
            'original_similarity': 0.25,
            'text_variant_consistency': 0.25,
            'retrieval_consistency': 0.25,
            'generative_consistency': 0.25
        }
        
        # 统计信息
        self.detection_history = []
        self.threshold_history = []
        
        logger.info(f"一致性检测器初始化完成，阈值: {threshold}")
    
    def make_decision(self, 
                     consistency_scores: Dict[str, float],
                     return_details: bool = False) -> Dict[str, Any]:
        """做出检测决策
        
        Args:
            consistency_scores: 一致性分数字典
            return_details: 是否返回详细信息
            
        Returns:
            检测决策结果
        """
        # 1. 计算综合一致性分数
        overall_score = self._compute_overall_score(consistency_scores)
        
        # 2. 确定检测阈值
        threshold = self._get_adaptive_threshold(consistency_scores) if self.adaptive_threshold else self.base_threshold
        
        # 3. 做出决策
        is_adversarial = overall_score < threshold
        
        # 4. 计算置信度
        confidence = self._compute_confidence(overall_score, threshold, consistency_scores)
        
        # 5. 记录历史
        self.detection_history.append({
            'overall_score': overall_score,
            'threshold': threshold,
            'is_adversarial': is_adversarial,
            'confidence': confidence
        })
        self.threshold_history.append(threshold)
        
        result = {
            'is_adversarial': is_adversarial,
            'confidence': confidence,
            'overall_score': overall_score,
            'threshold': threshold
        }
        
        if return_details:
            result['details'] = self._get_detailed_analysis(consistency_scores, overall_score, threshold)
        
        return result
    
    def _compute_overall_score(self, scores: Dict[str, float]) -> float:
        """计算综合一致性分数"""
        if self.voting_strategy == "simple":
            return self._simple_voting(scores)
        elif self.voting_strategy == "weighted":
            return self._weighted_voting(scores)
        elif self.voting_strategy == "adaptive":
            return self._adaptive_voting(scores)
        else:
            raise ValueError(f"未知的投票策略: {self.voting_strategy}")
    
    def _simple_voting(self, scores: Dict[str, float]) -> float:
        """简单投票：平均所有一致性分数"""
        relevant_scores = [
            scores.get('original_similarity', 0),
            scores.get('text_variant_consistency', 0),
            scores.get('retrieval_consistency', 0),
            scores.get('generative_consistency', 0)
        ]
        
        # 过滤掉0值（表示该模块未启用）
        valid_scores = [s for s in relevant_scores if s > 0]
        
        if not valid_scores:
            return 0.0
        
        return float(np.mean(valid_scores))
    
    def _weighted_voting(self, scores: Dict[str, float]) -> float:
        """加权投票：根据预设权重计算"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score_name, weight in self.weights.items():
            if score_name in scores and scores[score_name] > 0:
                weighted_sum += scores[score_name] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _adaptive_voting(self, scores: Dict[str, float]) -> float:
        """自适应投票：根据分数质量动态调整权重"""
        # 计算各分数的可靠性
        reliability_weights = self._compute_reliability_weights(scores)
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        score_mapping = {
            'original_similarity': scores.get('original_similarity', 0),
            'text_variant_consistency': scores.get('text_variant_consistency', 0),
            'retrieval_consistency': scores.get('retrieval_consistency', 0),
            'generative_consistency': scores.get('generative_consistency', 0)
        }
        
        for score_name, score_value in score_mapping.items():
            if score_value > 0 and score_name in reliability_weights:
                weight = reliability_weights[score_name]
                weighted_sum += score_value * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _compute_reliability_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """计算各分数的可靠性权重"""
        weights = {}
        
        # 原始相似度：总是可靠的
        weights['original_similarity'] = 1.0
        
        # 文本变体一致性：标准差越小越可靠
        text_std = scores.get('text_variant_std', 1.0)
        weights['text_variant_consistency'] = 1.0 / (1.0 + text_std)
        
        # 检索一致性：标准差越小越可靠
        retrieval_std = scores.get('retrieval_std', 1.0)
        weights['retrieval_consistency'] = 1.0 / (1.0 + retrieval_std)
        
        # 生成一致性：标准差越小越可靠
        generative_std = scores.get('generative_std', 1.0)
        weights['generative_consistency'] = 1.0 / (1.0 + generative_std)
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _get_adaptive_threshold(self, scores: Dict[str, float]) -> float:
        """计算自适应阈值"""
        # 基于历史数据和当前分数特征调整阈值
        base_threshold = self.base_threshold
        
        # 1. 基于跨模态方差调整
        cross_modal_var = scores.get('cross_modal_variance', 0)
        if cross_modal_var > 0.1:  # 高方差表示可能的攻击
            base_threshold += 0.1
        
        # 2. 基于标准差调整
        avg_std = np.mean([
            scores.get('text_variant_std', 0),
            scores.get('retrieval_std', 0),
            scores.get('generative_std', 0)
        ])
        
        if avg_std > 0.2:  # 高标准差表示不一致
            base_threshold += 0.05
        
        # 3. 基于历史检测结果调整
        if len(self.threshold_history) > 10:
            recent_thresholds = self.threshold_history[-10:]
            threshold_trend = np.mean(recent_thresholds)
            # 平滑调整
            base_threshold = 0.7 * base_threshold + 0.3 * threshold_trend
        
        # 限制阈值范围
        return np.clip(base_threshold, 0.1, 0.9)
    
    def _compute_confidence(self, 
                          overall_score: float, 
                          threshold: float, 
                          scores: Dict[str, float]) -> float:
        """计算检测置信度"""
        # 1. 基于分数与阈值的距离
        distance_confidence = abs(overall_score - threshold) / threshold
        
        # 2. 基于分数一致性
        score_values = [
            scores.get('original_similarity', 0),
            scores.get('text_variant_consistency', 0),
            scores.get('retrieval_consistency', 0),
            scores.get('generative_consistency', 0)
        ]
        valid_scores = [s for s in score_values if s > 0]
        
        if len(valid_scores) > 1:
            consistency_confidence = 1.0 - np.std(valid_scores)
        else:
            consistency_confidence = 0.5
        
        # 3. 基于跨模态方差
        variance_confidence = 1.0 - min(scores.get('cross_modal_variance', 0), 1.0)
        
        # 综合置信度
        confidence = np.mean([distance_confidence, consistency_confidence, variance_confidence])
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _get_detailed_analysis(self, 
                             scores: Dict[str, float], 
                             overall_score: float, 
                             threshold: float) -> Dict[str, Any]:
        """获取详细分析结果"""
        analysis = {
            'individual_scores': scores,
            'overall_score': overall_score,
            'threshold': threshold,
            'score_analysis': {},
            'risk_factors': [],
            'confidence_factors': []
        }
        
        # 分析各个分数
        for score_name, score_value in scores.items():
            if score_value > 0:
                analysis['score_analysis'][score_name] = {
                    'value': score_value,
                    'normalized': score_value / max(scores.values()) if max(scores.values()) > 0 else 0,
                    'contribution': self.weights.get(score_name, 0) * score_value
                }
        
        # 识别风险因素
        if scores.get('cross_modal_variance', 0) > 0.1:
            analysis['risk_factors'].append('高跨模态方差')
        
        if scores.get('text_variant_std', 0) > 0.2:
            analysis['risk_factors'].append('文本变体不一致')
        
        if scores.get('original_similarity', 1) < 0.3:
            analysis['risk_factors'].append('低原始相似度')
        
        # 识别置信度因素
        if len([s for s in scores.values() if s > 0]) >= 3:
            analysis['confidence_factors'].append('多模块验证')
        
        if scores.get('retrieval_consistency', 0) > 0.7:
            analysis['confidence_factors'].append('高检索一致性')
        
        if scores.get('generative_consistency', 0) > 0.7:
            analysis['confidence_factors'].append('高生成一致性')
        
        return analysis
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检测器统计信息"""
        if not self.detection_history:
            return {'message': '暂无检测历史'}
        
        scores = [d['overall_score'] for d in self.detection_history]
        thresholds = [d['threshold'] for d in self.detection_history]
        confidences = [d['confidence'] for d in self.detection_history]
        adversarial_count = sum(1 for d in self.detection_history if d['is_adversarial'])
        
        stats = {
            'total_detections': len(self.detection_history),
            'adversarial_detections': adversarial_count,
            'adversarial_rate': adversarial_count / len(self.detection_history),
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            },
            'threshold_statistics': {
                'mean': float(np.mean(thresholds)),
                'std': float(np.std(thresholds)),
                'min': float(np.min(thresholds)),
                'max': float(np.max(thresholds))
            },
            'confidence_statistics': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
        }
        
        return stats
    
    def reset_history(self):
        """重置检测历史"""
        self.detection_history.clear()
        self.threshold_history.clear()
        logger.info("检测历史已重置")
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新权重"""
        self.weights.update(new_weights)
        logger.info(f"权重已更新: {self.weights}")
    
    def calibrate_threshold(self, 
                          validation_scores: List[Dict[str, float]], 
                          validation_labels: List[bool]) -> float:
        """基于验证数据校准阈值
        
        Args:
            validation_scores: 验证集一致性分数
            validation_labels: 验证集标签 (True表示对抗样本)
            
        Returns:
            最优阈值
        """
        if len(validation_scores) != len(validation_labels):
            raise ValueError("分数和标签数量不匹配")
        
        # 计算所有样本的综合分数
        overall_scores = [self._compute_overall_score(scores) for scores in validation_scores]
        
        # 寻找最优阈值（最大化F1分数）
        best_threshold = self.base_threshold
        best_f1 = 0.0
        
        thresholds = np.linspace(0.1, 0.9, 81)  # 0.1到0.9，步长0.01
        
        for threshold in thresholds:
            predictions = [score < threshold for score in overall_scores]
            
            # 计算F1分数
            tp = sum(1 for pred, label in zip(predictions, validation_labels) if pred and label)
            fp = sum(1 for pred, label in zip(predictions, validation_labels) if pred and not label)
            fn = sum(1 for pred, label in zip(predictions, validation_labels) if not pred and label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"阈值校准完成: {best_threshold:.3f} (F1: {best_f1:.3f})")
        self.base_threshold = best_threshold
        
        return best_threshold