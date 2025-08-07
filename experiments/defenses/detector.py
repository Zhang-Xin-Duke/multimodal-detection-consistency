"""多模态防御检测器

该模块实现了多模态对抗检测的主要防御逻辑，整合了文本变体生成、
检索参考、生成参考和一致性检测等多个组件。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .text_variants import TextVariantGenerator
from .retrieval_ref import RetrievalReferenceGenerator
from .generative_ref import GenerativeReferenceGenerator
from .consistency_checker import ConsistencyChecker

logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """检测配置"""
    # 文本变体配置
    use_text_variants: bool = True
    text_variant_count: int = 5
    
    # 检索参考配置
    use_retrieval_ref: bool = True
    retrieval_top_k: int = 10
    retrieval_weight: float = 0.3
    
    # 生成参考配置
    use_generative_ref: bool = True
    generation_count: int = 3
    generation_weight: float = 0.4
    
    # 一致性检测配置
    consistency_threshold: float = 0.5
    adaptive_threshold: bool = True
    voting_strategy: str = "weighted"  # "simple", "weighted", "adaptive"
    
    # 其他配置
    device: str = "cuda"
    debug_mode: bool = False

class MultiModalDefenseDetector:
    """多模态防御检测器
    
    整合多种防御机制来检测对抗样本：
    1. 文本变体生成：生成语义一致的文本变体
    2. 检索参考：从真实图像库中检索相似图像
    3. 生成参考：使用Stable Diffusion生成参考图像
    4. 一致性检测：计算跨模态一致性并做出检测决策
    """
    
    def __init__(self, 
                 clip_model,
                 qwen_model=None,
                 sd_model=None,
                 config: DetectionConfig = None):
        """
        初始化防御检测器
        
        Args:
            clip_model: CLIP模型实例
            qwen_model: Qwen模型实例（用于文本变体生成）
            sd_model: Stable Diffusion模型实例（用于图像生成）
            config: 检测配置
        """
        self.clip_model = clip_model
        self.config = config or DetectionConfig()
        self.device = torch.device(self.config.device)
        
        # 初始化各个组件
        self._initialize_components(qwen_model, sd_model)
        
        logger.info("多模态防御检测器初始化完成")
    
    def _initialize_components(self, qwen_model, sd_model):
        """初始化各个防御组件"""
        # 文本变体生成器
        if self.config.use_text_variants and qwen_model is not None:
            self.text_variant_generator = TextVariantGenerator(
                qwen_model=qwen_model,
                clip_model=self.clip_model,
                variant_count=self.config.text_variant_count
            )
        else:
            self.text_variant_generator = None
        
        # 检索参考生成器
        if self.config.use_retrieval_ref:
            self.retrieval_generator = RetrievalReferenceGenerator(
                clip_model=self.clip_model,
                top_k=self.config.retrieval_top_k
            )
        else:
            self.retrieval_generator = None
        
        # 生成参考生成器
        if self.config.use_generative_ref and sd_model is not None:
            self.generative_generator = GenerativeReferenceGenerator(
                sd_model=sd_model,
                clip_model=self.clip_model,
                generation_count=self.config.generation_count
            )
        else:
            self.generative_generator = None
        
        # 一致性检测器
        self.consistency_checker = ConsistencyChecker(
            threshold=self.config.consistency_threshold,
            adaptive_threshold=self.config.adaptive_threshold,
            voting_strategy=self.config.voting_strategy
        )
    
    def detect(self, 
               image: torch.Tensor, 
               text: str,
               return_details: bool = False) -> Dict[str, Any]:
        """检测对抗样本
        
        Args:
            image: 输入图像张量 [1, C, H, W]
            text: 输入文本查询
            return_details: 是否返回详细信息
            
        Returns:
            检测结果字典，包含：
            - is_adversarial: 是否为对抗样本
            - confidence: 检测置信度
            - details: 详细信息（如果requested）
        """
        with torch.no_grad():
            # 1. 生成文本变体
            text_variants = self._generate_text_variants(text)
            
            # 2. 生成检索参考
            retrieval_refs = self._generate_retrieval_references(text, text_variants)
            
            # 3. 生成生成参考
            generative_refs = self._generate_generative_references(text, text_variants)
            
            # 4. 计算一致性分数
            consistency_scores = self._compute_consistency_scores(
                image, text, text_variants, retrieval_refs, generative_refs
            )
            
            # 5. 做出检测决策
            detection_result = self.consistency_checker.make_decision(
                consistency_scores, return_details=return_details
            )
            
            # 6. 组装结果
            result = {
                'is_adversarial': detection_result['is_adversarial'],
                'confidence': detection_result['confidence'],
                'consistency_score': detection_result['overall_score']
            }
            
            if return_details:
                result['details'] = {
                    'text_variants': text_variants,
                    'retrieval_references': retrieval_refs,
                    'generative_references': generative_refs,
                    'consistency_scores': consistency_scores,
                    'detection_details': detection_result
                }
            
            return result
    
    def _generate_text_variants(self, text: str) -> List[str]:
        """生成文本变体"""
        if self.text_variant_generator is None:
            return [text]  # 如果没有文本变体生成器，返回原文本
        
        try:
            variants = self.text_variant_generator.generate_variants(text)
            return [text] + variants  # 包含原始文本
        except Exception as e:
            logger.warning(f"文本变体生成失败: {e}")
            return [text]
    
    def _generate_retrieval_references(self, 
                                     original_text: str, 
                                     text_variants: List[str]) -> List[torch.Tensor]:
        """生成检索参考图像"""
        if self.retrieval_generator is None:
            return []
        
        try:
            # 对所有文本变体进行检索
            all_references = []
            for text in text_variants:
                refs = self.retrieval_generator.retrieve_references(text)
                all_references.extend(refs)
            
            # 去重并限制数量
            unique_refs = self._deduplicate_references(all_references)
            return unique_refs[:self.config.retrieval_top_k]
            
        except Exception as e:
            logger.warning(f"检索参考生成失败: {e}")
            return []
    
    def _generate_generative_references(self, 
                                      original_text: str, 
                                      text_variants: List[str]) -> List[torch.Tensor]:
        """生成生成参考图像"""
        if self.generative_generator is None:
            return []
        
        try:
            # 选择最佳文本变体进行生成
            selected_texts = text_variants[:min(len(text_variants), 3)]  # 最多选择3个
            
            all_generated = []
            for text in selected_texts:
                generated = self.generative_generator.generate_references(text)
                all_generated.extend(generated)
            
            return all_generated[:self.config.generation_count]
            
        except Exception as e:
            logger.warning(f"生成参考生成失败: {e}")
            return []
    
    def _compute_consistency_scores(self, 
                                  image: torch.Tensor,
                                  original_text: str,
                                  text_variants: List[str],
                                  retrieval_refs: List[torch.Tensor],
                                  generative_refs: List[torch.Tensor]) -> Dict[str, float]:
        """计算一致性分数"""
        scores = {}
        
        # 1. 原始图文一致性
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text([original_text])
        original_similarity = torch.cosine_similarity(image_features, text_features, dim=-1).item()
        scores['original_similarity'] = original_similarity
        
        # 2. 文本变体一致性
        if len(text_variants) > 1:
            variant_similarities = []
            for variant in text_variants[1:]:  # 跳过原始文本
                variant_features = self.clip_model.encode_text([variant])
                sim = torch.cosine_similarity(image_features, variant_features, dim=-1).item()
                variant_similarities.append(sim)
            
            scores['text_variant_consistency'] = np.mean(variant_similarities)
            scores['text_variant_std'] = np.std(variant_similarities)
        else:
            scores['text_variant_consistency'] = original_similarity
            scores['text_variant_std'] = 0.0
        
        # 3. 检索参考一致性
        if retrieval_refs:
            retrieval_similarities = []
            for ref_image in retrieval_refs:
                ref_features = self.clip_model.encode_image(ref_image.unsqueeze(0))
                sim = torch.cosine_similarity(image_features, ref_features, dim=-1).item()
                retrieval_similarities.append(sim)
            
            scores['retrieval_consistency'] = np.mean(retrieval_similarities)
            scores['retrieval_std'] = np.std(retrieval_similarities)
        else:
            scores['retrieval_consistency'] = 0.0
            scores['retrieval_std'] = 0.0
        
        # 4. 生成参考一致性
        if generative_refs:
            generative_similarities = []
            for gen_image in generative_refs:
                gen_features = self.clip_model.encode_image(gen_image.unsqueeze(0))
                sim = torch.cosine_similarity(image_features, gen_features, dim=-1).item()
                generative_similarities.append(sim)
            
            scores['generative_consistency'] = np.mean(generative_similarities)
            scores['generative_std'] = np.std(generative_similarities)
        else:
            scores['generative_consistency'] = 0.0
            scores['generative_std'] = 0.0
        
        # 5. 跨模态一致性分析
        scores['cross_modal_variance'] = self._compute_cross_modal_variance(
            original_similarity,
            scores.get('text_variant_consistency', 0),
            scores.get('retrieval_consistency', 0),
            scores.get('generative_consistency', 0)
        )
        
        return scores
    
    def _compute_cross_modal_variance(self, *similarities) -> float:
        """计算跨模态方差"""
        valid_sims = [s for s in similarities if s > 0]
        if len(valid_sims) < 2:
            return 0.0
        return float(np.var(valid_sims))
    
    def _deduplicate_references(self, references: List[torch.Tensor]) -> List[torch.Tensor]:
        """去重参考图像"""
        if not references:
            return []
        
        # 简单的去重策略：计算特征相似度
        unique_refs = [references[0]]
        
        for ref in references[1:]:
            is_duplicate = False
            ref_features = self.clip_model.encode_image(ref.unsqueeze(0))
            
            for unique_ref in unique_refs:
                unique_features = self.clip_model.encode_image(unique_ref.unsqueeze(0))
                similarity = torch.cosine_similarity(ref_features, unique_features, dim=-1).item()
                
                if similarity > 0.95:  # 高相似度阈值
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_refs.append(ref)
        
        return unique_refs
    
    def batch_detect(self, 
                    images: torch.Tensor, 
                    texts: List[str],
                    return_details: bool = False) -> List[Dict[str, Any]]:
        """批量检测
        
        Args:
            images: 批量图像张量 [B, C, H, W]
            texts: 批量文本列表
            return_details: 是否返回详细信息
            
        Returns:
            检测结果列表
        """
        batch_size = images.shape[0]
        results = []
        
        for i in range(batch_size):
            image = images[i:i+1]  # 保持批次维度
            text = texts[i]
            
            result = self.detect(image, text, return_details)
            results.append(result)
        
        return results
    
    def update_config(self, new_config: DetectionConfig):
        """更新配置"""
        self.config = new_config
        # 重新初始化组件（如果需要）
        # 这里可以添加更智能的配置更新逻辑
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检测器统计信息"""
        stats = {
            'config': self.config.__dict__,
            'components': {
                'text_variant_generator': self.text_variant_generator is not None,
                'retrieval_generator': self.retrieval_generator is not None,
                'generative_generator': self.generative_generator is not None,
                'consistency_checker': self.consistency_checker is not None
            }
        }
        
        # 添加各组件的统计信息
        if hasattr(self.consistency_checker, 'get_statistics'):
            stats['consistency_checker_stats'] = self.consistency_checker.get_statistics()
        
        return stats