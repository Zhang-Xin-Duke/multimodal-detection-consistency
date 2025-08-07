"""
对抗性检测器模块

实现基于一致性的对抗性检测器，用于检测对抗性样本。
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
from PIL import Image
from sklearn.metrics import roc_curve, auc
from .models.clip_model import CLIPModel, CLIPConfig

from .utils.metrics import SimilarityMetrics, DetectionEvaluator, SimilarityCalculator
import warnings
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyScore:
    """一致性分数数据类"""
    direct_score: float = 0.0
    text_variant_score: float = 0.0
    generative_ref_score: float = 0.0
    combined_score: float = 0.0
    is_adversarial: bool = False
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'direct_score': self.direct_score,
            'text_variant_score': self.text_variant_score,
            'generative_ref_score': self.generative_ref_score,
            'combined_score': self.combined_score,
            'is_adversarial': self.is_adversarial,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsistencyScore':
        """从字典创建"""
        return cls(**data)


class ThresholdManager:
    """阈值管理器"""
    
    def __init__(self, initial_threshold: float = 0.5):
        self.threshold = initial_threshold
        self.detection_history = []
        
    def update_threshold(self, new_threshold: float):
        """更新阈值"""
        self.threshold = new_threshold
        
    def is_adversarial(self, score: float) -> bool:
        """判断是否为对抗样本"""
        return score < self.threshold
        
    def batch_detection(self, scores: List[float]) -> List[bool]:
        """批量检测"""
        return [self.is_adversarial(score) for score in scores]
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'current_threshold': self.threshold,
            'detection_count': len(self.detection_history)
        }


class AggregationMethod(Enum):
    """聚合方法枚举"""
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    WEIGHTED_AVERAGE = "weighted_average"


class ThresholdMethod(Enum):
    """阈值方法枚举"""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    PERCENTILE = "percentile"


class AdaptiveThresholdManager:
    """自适应阈值管理器"""
    
    def __init__(self, initial_threshold: float = 0.5, adaptation_rate: float = 0.1):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.history = []
        self.performance_history = []
    
    def update_threshold(self, scores: List[float], labels: List[bool]):
        """根据性能更新阈值"""
        if len(scores) != len(labels):
            return
        
        # 计算当前阈值下的性能
        predictions = [score > self.threshold for score in scores]
        accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
        
        # 记录历史
        self.history.append(self.threshold)
        self.performance_history.append(accuracy)
        
        # 尝试调整阈值
        if len(self.performance_history) > 1:
            if accuracy < self.performance_history[-2]:
                # 性能下降，反向调整
                self.adaptation_rate *= -0.5
            
            # 调整阈值
            self.threshold += self.adaptation_rate
            self.threshold = max(0.0, min(1.0, self.threshold))
    
    def get_threshold(self) -> float:
        return self.threshold


class EnsembleDetector:
    """集成检测器"""
    
    def __init__(self, detectors: List['AdversarialDetector'], weights: Optional[List[float]] = None):
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)
        
        if len(self.weights) != len(self.detectors):
            raise ValueError("权重数量必须与检测器数量相同")
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def detect_adversarial(self, image: Union[Image.Image, torch.Tensor], 
                          text: str, 
                          methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """集成检测"""
        results = []
        
        for detector in self.detectors:
            result = detector.detect_adversarial(image, text, methods)
            results.append(result)
        
        # 加权平均聚合
        weighted_score = sum(r['aggregated_score'] * w for r, w in zip(results, self.weights))
        
        # 投票决定是否为对抗样本
        adversarial_votes = sum(r['is_adversarial'] * w for r, w in zip(results, self.weights))
        is_adversarial = adversarial_votes > 0.5
        
        return {
            'is_adversarial': is_adversarial,
            'aggregated_score': weighted_score,
            'individual_results': results,
            'weights': self.weights,
            'ensemble_confidence': max(weighted_score, 1 - weighted_score)
        }


@dataclass
class DetectorConfig:
    """检测器配置"""
    # 模型配置
    clip_model: str = "ViT-B/32"
    device: str = "cuda"
    
    # 检测方法
    detection_methods: List[str] = None
    
    # 文本变体检测
    use_text_variants: bool = True
    num_text_variants: int = 5
    text_similarity_threshold: float = 0.85
    
    # SD参考检测
    use_sd_reference: bool = True
    num_reference_images: int = 3
    reference_similarity_threshold: float = 0.75
    
    # 一致性检测
    consistency_threshold: float = 0.8
    consistency_weight: float = 0.5
    
    # 阈值设置
    detection_threshold: float = 0.5
    adaptive_threshold: bool = True
    threshold_percentile: float = 95.0
    
    # 聚合方法
    score_aggregation: str = "weighted_mean"  # mean, max, min, weighted_mean
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000
    
    # 批处理
    batch_size: int = 32
    
    def __post_init__(self):
        if self.detection_methods is None:
            self.detection_methods = ['text_variants', 'sd_reference', 'consistency']


from functools import lru_cache

class AdversarialDetector:
    """对抗性检测器"""
    
    def __init__(self, config: DetectorConfig):
        """
        初始化对抗性检测器
        
        Args:
            config: 检测器配置
        """
        self.config = config
        
        self.clip_model = None
        self.text_augmenter = None
        self.sd_generator = None
        
        # 相似性计算器
        self.similarity_calculator = SimilarityCalculator()
        
        # 检测评估器
        self.detection_evaluator = DetectionEvaluator()
        
        # 缓存
        self.detection_cache = {}
        self.threshold_cache = {}
        
        # 统计信息
        self.detection_stats = {
            'total_detections': 0,
            'cache_hits': 0,
            'detection_time': 0.0,
            'method_usage': {method: 0 for method in self.config.detection_methods}
        }
        
        logger.info("对抗性检测器初始化完成")
    
    def _get_clip_model(self) -> CLIPModel:
        if self.clip_model is None:
            self.clip_model = self._initialize_clip_model(self.config.clip_model, self.config.device)
        return self.clip_model

    def _initialize_clip_model(self, model_name: str, device: str) -> CLIPModel:
        """
        初始化CLIP模型
        
        Returns:
            CLIP模型实例
        """
        try:
            clip_config = CLIPConfig(
                model_name=model_name,
                device=device
            )
            
            clip_model = CLIPModel(clip_config)
            logger.info(f"CLIP模型初始化完成: {model_name}")
            
            return clip_model
            
        except Exception as e:
            logger.error(f"CLIP模型初始化失败: {e}")
            raise
    
    def _get_text_augmenter(self):
        if self.text_augmenter is None:
            if self.config.use_text_variants:
                self.text_augmenter = self._initialize_text_augmenter(
                    self.config.num_text_variants,
                    self.config.text_similarity_threshold
                )
        return self.text_augmenter

    def _initialize_text_augmenter(self, num_variants: int, similarity_threshold: float) -> Optional['TextAugmenter']:
        """
        初始化文本增强器
        
        Returns:
            文本增强器实例
        """
        try:
            from src.text_augment import TextAugmenter, TextAugmentConfig
            
            # 创建文本增强配置，使用正确的参数名
            text_augment_config = TextAugmentConfig(
                max_variants=num_variants, 
                min_similarity_threshold=similarity_threshold,
                paraphrase_model="Qwen/Qwen2-7B-Instruct",
                device=self.config.device
            )
            
            # 使用正确的参数初始化TextAugmenter
            text_augmenter = TextAugmenter(text_augment_config)
            logger.info("文本增强器初始化完成")
            
            return text_augmenter
            
        except Exception as e:
            logger.warning(f"文本增强器初始化失败: {e}")
            return None
    
    def _get_sd_generator(self):
        if self.sd_generator is None:
            if self.config.use_sd_reference:
                self.sd_generator = self._initialize_sd_generator(
                    self.config.num_reference_images,
                    self.config.device
                )
        return self.sd_generator

    def _initialize_sd_generator(self, num_images_per_prompt: int, device: str) -> Optional['SDReferenceGenerator']:
        """
        初始化SD参考生成器
        
        Returns:
            SD参考生成器实例
        """
        try:
            from src.sd_ref import SDReferenceGenerator, SDReferenceConfig
            sd_ref_config = SDReferenceConfig(num_images_per_prompt=num_images_per_prompt, device=device)
            sd_generator = SDReferenceGenerator(sd_ref_config)
            logger.info("SD参考生成器初始化完成")
            
            return sd_generator
            
        except Exception as e:
            logger.warning(f"SD参考生成器初始化失败: {e}")
            return None
    
    def detect_adversarial(self, image: Union[Image.Image, torch.Tensor], 
                          text: str, 
                          methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        检测对抗性样本
        
        Args:
            image: 输入图像
            text: 输入文本
            methods: 使用的检测方法
            
        Returns:
            检测结果字典
        """
        try:
            methods = methods or self.config.detection_methods
            
            # 检查缓存
            cache_key = self._get_cache_key(image, text, methods)
            if self.config.enable_cache and cache_key in self.detection_cache:
                self.detection_stats['cache_hits'] += 1
                return self.detection_cache[cache_key]
            
            start_time = time.time()
            
            # 执行各种检测方法
            detection_scores = {}
            detection_details = {}
            
            # 1. 文本变体检测
            if 'text_variants' in methods and self._get_text_augmenter() is not None:
                score, details = self._detect_by_text_variants(image, text)
                detection_scores['text_variants'] = score
                detection_details['text_variants'] = details
                self.detection_stats['method_usage']['text_variants'] += 1
            
            # 2. SD参考检测
            if 'sd_reference' in methods and self._get_sd_generator() is not None:
                score, details = self._detect_by_sd_reference(image, text)
                detection_scores['sd_reference'] = score
                detection_details['sd_reference'] = details
                self.detection_stats['method_usage']['sd_reference'] += 1
            
            # 3. 一致性检测
            if 'consistency' in methods:
                score, details = self._detect_by_consistency(image, text)
                detection_scores['consistency'] = score
                detection_details['consistency'] = details
                self.detection_stats['method_usage']['consistency'] += 1
            
            # 聚合检测分数
            aggregated_score = self._aggregate_scores(detection_scores)
            
            # 判断是否为对抗性样本
            is_adversarial = aggregated_score > self.config.detection_threshold
            
            # 构建结果
            result = {
                'is_adversarial': is_adversarial,
                'aggregated_score': aggregated_score,
                'detection_scores': detection_scores,
                'detection_details': detection_details,
                'detection_time': time.time() - start_time,
                'methods_used': methods,
                'threshold': self.config.detection_threshold
            }
            
            # 缓存结果
            if self.config.enable_cache:
                if len(self.detection_cache) >= self.config.cache_size:
                    # 清理最旧的缓存项
                    oldest_key = next(iter(self.detection_cache))
                    del self.detection_cache[oldest_key]
                
                self.detection_cache[cache_key] = result
            
            # 更新统计信息
            self.detection_stats['total_detections'] += 1
            self.detection_stats['detection_time'] += result['detection_time']
            
            logger.debug(f"对抗性检测完成: {is_adversarial} (分数: {aggregated_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"对抗性检测失败: {e}")
            return {
                'is_adversarial': False,
                'aggregated_score': 0.0,
                'detection_scores': {},
                'detection_details': {},
                'detection_time': 0.0,
                'methods_used': methods,
                'threshold': self.config.detection_threshold,
                'error': str(e)
            }
    
    def _detect_by_text_variants(self, image: Union[Image.Image, torch.Tensor], 
                                text: str) -> Tuple[float, Dict[str, Any]]:
        """
        基于文本变体的检测
        
        Args:
            image: 输入图像
            text: 输入文本
            
        Returns:
            (检测分数, 详细信息)
        """
        try:
            # 生成文本变体
            variants = self._get_text_augmenter().generate_variants(text)
            
            if not variants:
                return 0.0, {'error': '无法生成文本变体'}
            
            # 计算原始文本与图像的相似性
            original_similarity = self._get_clip_model().get_text_image_similarity(
                text, image
            ).item()
            
            # 计算变体与图像的相似性
            variant_similarities = []
            for variant in variants:
                similarity = self._get_clip_model().get_text_image_similarity(
                    variant, image
                ).item()
                variant_similarities.append(similarity)
            
            # 计算一致性分数
            variant_similarities = np.array(variant_similarities)
            mean_variant_similarity = variant_similarities.mean()
            std_variant_similarity = variant_similarities.std()
            
            # 检测分数：原始相似性与变体相似性的差异
            consistency_score = 1.0 - abs(original_similarity - mean_variant_similarity)
            
            # 变异性分数：变体之间的一致性
            variability_score = 1.0 - std_variant_similarity
            
            # 综合检测分数（低一致性 = 高对抗性概率）
            detection_score = 1.0 - (consistency_score * 0.7 + variability_score * 0.3)
            
            details = {
                'original_similarity': float(original_similarity),
                'variant_similarities': variant_similarities.tolist(),
                'mean_variant_similarity': float(mean_variant_similarity),
                'std_variant_similarity': float(std_variant_similarity),
                'consistency_score': float(consistency_score),
                'variability_score': float(variability_score),
                'num_variants': len(variants)
            }
            
            return float(detection_score), details
            
        except Exception as e:
            logger.error(f"文本变体检测失败: {e}")
            return 0.0, {'error': str(e)}
    
    def _detect_by_sd_reference(self, image: Union[Image.Image, torch.Tensor], 
                               text: str) -> Tuple[float, Dict[str, Any]]:
        """
        基于SD参考的检测
        
        Args:
            image: 输入图像
            text: 输入文本
            
        Returns:
            (检测分数, 详细信息)
        """
        try:
            # 生成参考图像
            ref_result = self._get_sd_generator().generate_reference_images(
                text, 
                num_images=self.config.num_reference_images
            )
            
            reference_images = ref_result['images']
            
            if not reference_images:
                return 0.0, {'error': '无法生成参考图像'}
            
            # 计算输入图像与参考图像的相似性
            similarities = []
            for ref_image in reference_images:
                similarity = self.similarity_calculator.cosine_similarity(
                    self._image_to_features(image),
                    self._image_to_features(ref_image)
                )
                similarities.append(similarity)
            
            similarities = np.array(similarities)
            mean_similarity = similarities.mean()
            max_similarity = similarities.max()
            std_similarity = similarities.std()
            
            # 检测分数：与参考图像的相似性越低，对抗性概率越高
            detection_score = 1.0 - mean_similarity
            
            details = {
                'reference_similarities': similarities.tolist(),
                'mean_similarity': float(mean_similarity),
                'max_similarity': float(max_similarity),
                'std_similarity': float(std_similarity),
                'num_references': len(reference_images),
                'generation_time': ref_result.get('generation_time', 0.0)
            }
            
            return float(detection_score), details
            
        except Exception as e:
            logger.error(f"SD参考检测失败: {e}")
            return 0.0, {'error': str(e)}
    
    def _detect_by_consistency(self, image: Union[Image.Image, torch.Tensor], 
                              text: str) -> Tuple[float, Dict[str, Any]]:
        """
        基于一致性的检测
        
        Args:
            image: 输入图像
            text: 输入文本
            
        Returns:
            (检测分数, 详细信息)
        """
        try:
            # 计算图像-文本相似性
            image_text_similarity = self._get_clip_model().get_text_image_similarity(text, image).item()
            
            # 计算一致性分数（简单使用相似度值作为一致性分数）
            consistency_score = float(image_text_similarity)
            
            # 检测分数：低一致性表示高对抗性概率
            detection_score = 1.0 - consistency_score
            
            details = {
                'image_text_similarity': float(image_text_similarity),
                'consistency_score': float(consistency_score)
            }
            
            return float(detection_score), details
            
        except Exception as e:
            logger.error(f"一致性检测失败: {e}")
            return 0.0, {'error': str(e)}
    
    def _image_to_features(self, image: Union[Image.Image, torch.Tensor]) -> np.ndarray:
        """
        将图像转换为特征向量
        
        Args:
            image: 输入图像
            
        Returns:
            特征向量
        """
        try:
            clip_model = self._get_clip_model()
            
            if isinstance(image, torch.Tensor):
                # 对于torch.Tensor，使用CLIP模型的encode_image_tensor方法
                # 确保张量格式正确 (C, H, W) 或 (B, C, H, W)
                if image.dim() == 3:
                    # 单张图像 (C, H, W) -> (1, C, H, W)
                    image = image.unsqueeze(0)
                elif image.dim() == 4 and image.size(0) == 1:
                    # 已经是批次格式 (1, C, H, W)
                    pass
                elif image.dim() == 4 and image.size(0) > 1:
                    # 多张图像，只取第一张
                    image = image[0:1]
                else:
                    logger.warning(f"意外的图像张量维度: {image.shape}")
                    # 尝试reshape为合理的图像格式
                    if image.numel() == 3 * 224 * 224:  # 假设是224x224的RGB图像
                        image = image.view(1, 3, 224, 224)
                    else:
                        raise ValueError(f"无法处理的图像张量形状: {image.shape}")
                
                # 使用CLIP编码
                features = clip_model.encode_image_tensor(image, requires_grad=False)
                # 确保返回单个特征向量
                if features.dim() > 1:
                    features = features.squeeze(0)  # 移除批次维度
                return features.cpu().numpy()
            else:
                # PIL图像，使用CLIP编码
                features = clip_model.encode_image([image])  # 传入列表
                # 确保返回单个特征向量
                if features.dim() > 1:
                    features = features.squeeze(0)  # 移除批次维度
                return features.cpu().numpy()
                
        except Exception as e:
            logger.error(f"图像特征提取失败: {e}")
            return np.array([])
    
    def _aggregate_scores(self, scores: Dict[str, float]) -> float:
        """
        聚合检测分数
        
        Args:
            scores: 各方法的检测分数
            
        Returns:
            聚合后的分数
        """
        if not scores:
            return 0.0
        
        score_values = list(scores.values())
        
        if self.config.score_aggregation == "mean":
            return np.mean(score_values)
        elif self.config.score_aggregation == "max":
            return np.max(score_values)
        elif self.config.score_aggregation == "min":
            return np.min(score_values)
        elif self.config.score_aggregation == "weighted_mean":
            # 权重设置
            weights = {
                'text_variants': 0.4,
                'sd_reference': 0.4,
                'consistency': 0.2
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for method, score in scores.items():
                weight = weights.get(method, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            return np.mean(score_values)
    
    def _get_cache_key(self, image: Union[Image.Image, torch.Tensor], 
                      text: str, methods: List[str]) -> str:
        """
        生成缓存键
        
        Args:
            image: 输入图像
            text: 输入文本
            methods: 检测方法
            
        Returns:
            缓存键
        """
        # 简化的缓存键生成（实际应用中可能需要更复杂的方法）
        text_hash = hash(text)
        methods_hash = hash(tuple(sorted(methods)))
        
        # 对于图像，使用简单的哈希（实际应用中可能需要更好的方法）
        if isinstance(image, torch.Tensor):
            image_hash = hash(image.cpu().numpy().tobytes())
        else:
            # PIL图像转换为numpy数组后哈希
            image_array = np.array(image)
            image_hash = hash(image_array.tobytes())
        
        return f"{text_hash}_{image_hash}_{methods_hash}"
    
    def batch_detect(self, images: List[Union[Image.Image, torch.Tensor]], 
                    texts: List[str], 
                    methods: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        批量检测对抗性样本
        
        Args:
            images: 图像列表
            texts: 文本列表
            methods: 检测方法
            
        Returns:
            检测结果列表
        """
        # This is a placeholder for a batch-capable detection.
        # The current implementation processes one by one.
        # For actual batch processing, you would need to adapt the methods below.
        results = []
        
        for image, text in zip(images, texts):
            result = self.detect_adversarial(image, text, methods)
            results.append(result)
        
        return results
    
    def compute_optimal_threshold(self, validation_data: List[Tuple[Any, Any, bool]], 
                                 methods: Optional[List[str]] = None) -> float:
        """
        计算最优检测阈值
        
        Args:
            validation_data: 验证数据 [(image, text, is_adversarial), ...]
            methods: 检测方法
            
        Returns:
            最优阈值
        """
        try:
            scores = []
            labels = []
            
            for image, text, is_adversarial in validation_data:
                result = self.detect_adversarial(image, text, methods)
                scores.append(result['aggregated_score'])
                labels.append(int(is_adversarial))
            
            # 使用ROC曲线找到最优阈值
            fpr, tpr, thresholds = roc_curve(labels, scores)
            
            # 找到最大化TPR-FPR的阈值
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            logger.info(f"计算得到最优阈值: {optimal_threshold:.3f}")
            return float(optimal_threshold)
            
        except Exception as e:
            logger.error(f"计算最优阈值失败: {e}")
            return self.config.detection_threshold
    
    def update_threshold(self, new_threshold: float):
        """
        更新检测阈值
        
        Args:
            new_threshold: 新阈值
        """
        self.config.detection_threshold = new_threshold
        logger.info(f"检测阈值已更新: {new_threshold}")
    
    def evaluate_detection_performance(self, test_data: List[Tuple[Any, Any, bool]], 
                                     methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        评估检测性能
        
        Args:
            test_data: 测试数据 [(image, text, is_adversarial), ...]
            methods: 检测方法
            
        Returns:
            性能评估结果
        """
        try:
            predictions = []
            scores = []
            labels = []
            
            for image, text, is_adversarial in test_data:
                result = self.detect_adversarial(image, text, methods)
                predictions.append(result['is_adversarial'])
                scores.append(result['aggregated_score'])
                labels.append(is_adversarial)
            
            # 计算性能指标
            performance = self.detection_evaluator.compute_metrics(
                np.array(labels), 
                np.array(predictions), 
                np.array(scores)
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"性能评估失败: {e}")
            return {}
    
    def clear_cache(self):
        """
        清理缓存
        """
        self.detection_cache.clear()
        self.threshold_cache.clear()
        logger.info("检测器缓存已清理")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'detection_stats': self.detection_stats.copy(),
            'cache_size': len(self.detection_cache),
            'config': {
                'detection_methods': self.config.detection_methods,
                'detection_threshold': self.config.detection_threshold,
                'score_aggregation': self.config.score_aggregation,
                'use_text_variants': self.config.use_text_variants,
                'use_sd_reference': self.config.use_sd_reference
            }
        }
    
    def save_model(self, save_path: str):
        """
        保存检测器模型
        
        Args:
            save_path: 保存路径
        """
        try:
            save_data = {
                'config': self.config.__dict__,
                'detection_stats': self.detection_stats,
                'threshold_cache': self.threshold_cache
            }
            
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"检测器模型已保存: {save_path}")
            
        except Exception as e:
            logger.error(f"保存检测器模型失败: {e}")
    
    def load_model(self, load_path: str):
        """
        加载检测器模型
        
        Args:
            load_path: 加载路径
        """
        try:
            with open(load_path, 'r') as f:
                save_data = json.load(f)
            
            # 更新配置
            for key, value in save_data['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # 恢复统计信息
            self.detection_stats.update(save_data.get('detection_stats', {}))
            self.threshold_cache.update(save_data.get('threshold_cache', {}))
            
            logger.info(f"检测器模型已加载: {load_path}")
            
        except Exception as e:
            logger.error(f"加载检测器模型失败: {e}")


def create_adversarial_detector(config: Optional[DetectorConfig] = None) -> AdversarialDetector:
    """
    创建对抗性检测器实例
    
    Args:
        config: 检测器配置
        
    Returns:
        对抗性检测器实例
    """
    if config is None:
        config = DetectorConfig()
    
    return AdversarialDetector(config)