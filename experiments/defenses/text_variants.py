"""文本变体生成器

该模块负责生成语义一致的文本变体，用于增强对抗检测的鲁棒性。
通过生成多个语义相似但表达不同的文本查询，可以检测对抗攻击对文本的敏感性。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import re
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TextVariantConfig:
    """文本变体生成配置"""
    variant_count: int = 5
    max_length: int = 77  # CLIP文本编码器的最大长度
    temperature: float = 0.7
    diversity_threshold: float = 0.1  # 最小语义差异阈值
    similarity_threshold: float = 0.8  # 最大语义相似度阈值
    use_synonyms: bool = True
    use_paraphrasing: bool = True
    use_reordering: bool = True
    filter_quality: bool = True

class TextVariantGenerator:
    """文本变体生成器
    
    使用多种策略生成语义一致的文本变体：
    1. 同义词替换
    2. 句式改写
    3. 词序调整
    4. 语言模型生成
    """
    
    def __init__(self, 
                 qwen_model,
                 clip_model,
                 config: TextVariantConfig = None):
        """
        初始化文本变体生成器
        
        Args:
            qwen_model: Qwen语言模型实例
            clip_model: CLIP模型实例（用于语义相似度验证）
            config: 生成配置
        """
        self.qwen_model = qwen_model
        self.clip_model = clip_model
        self.config = config or TextVariantConfig()
        
        # 初始化同义词词典
        self.synonym_dict = self._load_synonym_dict()
        
        # 改写模板
        self.paraphrase_templates = self._load_paraphrase_templates()
        
        logger.info("文本变体生成器初始化完成")
    
    def generate_variants(self, text: str) -> List[str]:
        """生成文本变体
        
        Args:
            text: 原始文本
            
        Returns:
            文本变体列表
        """
        variants = []
        
        try:
            # 1. 同义词替换变体
            if self.config.use_synonyms:
                synonym_variants = self._generate_synonym_variants(text)
                variants.extend(synonym_variants)
            
            # 2. 改写变体
            if self.config.use_paraphrasing:
                paraphrase_variants = self._generate_paraphrase_variants(text)
                variants.extend(paraphrase_variants)
            
            # 3. 词序调整变体
            if self.config.use_reordering:
                reorder_variants = self._generate_reorder_variants(text)
                variants.extend(reorder_variants)
            
            # 4. 语言模型生成变体
            llm_variants = self._generate_llm_variants(text)
            variants.extend(llm_variants)
            
            # 5. 过滤和去重
            if self.config.filter_quality:
                variants = self._filter_variants(text, variants)
            
            # 6. 限制数量
            variants = variants[:self.config.variant_count]
            
            logger.debug(f"为文本 '{text[:50]}...' 生成了 {len(variants)} 个变体")
            
        except Exception as e:
            logger.error(f"生成文本变体时出错: {e}")
            variants = []
        
        return variants
    
    def _generate_synonym_variants(self, text: str) -> List[str]:
        """生成同义词替换变体"""
        variants = []
        words = text.split()
        
        # 为每个词尝试同义词替换
        for i in range(len(words)):
            word = words[i].lower().strip('.,!?;:')
            
            if word in self.synonym_dict:
                synonyms = self.synonym_dict[word]
                
                for synonym in synonyms[:3]:  # 最多3个同义词
                    new_words = words.copy()
                    # 保持原词的大小写格式
                    if words[i].isupper():
                        new_words[i] = synonym.upper()
                    elif words[i].istitle():
                        new_words[i] = synonym.capitalize()
                    else:
                        new_words[i] = synonym
                    
                    variant = ' '.join(new_words)
                    if variant != text and len(variant) <= self.config.max_length * 4:  # 粗略长度限制
                        variants.append(variant)
        
        return variants
    
    def _generate_paraphrase_variants(self, text: str) -> List[str]:
        """生成改写变体"""
        variants = []
        
        # 使用预定义模板进行改写
        for template in self.paraphrase_templates:
            try:
                variant = template.format(text=text)
                if variant != text and len(variant) <= self.config.max_length * 4:
                    variants.append(variant)
            except:
                continue
        
        # 简单的句式变换
        simple_variants = self._simple_paraphrases(text)
        variants.extend(simple_variants)
        
        return variants
    
    def _generate_reorder_variants(self, text: str) -> List[str]:
        """生成词序调整变体"""
        variants = []
        
        # 简单的词序调整（适用于短句）
        words = text.split()
        if len(words) <= 8:  # 只对短句进行词序调整
            # 随机交换相邻词语
            for _ in range(min(3, len(words) - 1)):
                new_words = words.copy()
                i = random.randint(0, len(words) - 2)
                new_words[i], new_words[i + 1] = new_words[i + 1], new_words[i]
                
                variant = ' '.join(new_words)
                if variant != text:
                    variants.append(variant)
        
        return variants
    
    def _generate_llm_variants(self, text: str) -> List[str]:
        """使用语言模型生成变体"""
        variants = []
        
        try:
            # 构造提示词
            prompts = [
                f"请改写以下句子，保持原意不变：{text}",
                f"用不同的表达方式重新描述：{text}",
                f"换一种说法表达同样的意思：{text}",
                f"请提供这句话的同义表达：{text}"
            ]
            
            for prompt in prompts[:2]:  # 限制调用次数
                response = self.qwen_model.generate(
                    prompt=prompt,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature
                )
                
                # 提取生成的文本
                generated_text = self._extract_generated_text(response, text)
                if generated_text and generated_text != text:
                    variants.append(generated_text)
                    
        except Exception as e:
            logger.warning(f"语言模型生成变体失败: {e}")
        
        return variants
    
    def _filter_variants(self, original_text: str, variants: List[str]) -> List[str]:
        """过滤和验证变体质量"""
        if not variants:
            return []
        
        filtered_variants = []
        
        # 编码原始文本
        original_features = self.clip_model.encode_text([original_text])
        
        for variant in variants:
            # 基本过滤
            if not self._basic_filter(variant, original_text):
                continue
            
            # 语义相似度过滤
            if not self._semantic_filter(variant, original_text, original_features):
                continue
            
            filtered_variants.append(variant)
        
        # 去重
        filtered_variants = list(set(filtered_variants))
        
        # 按语义相似度排序
        if len(filtered_variants) > 1:
            filtered_variants = self._rank_variants(original_text, filtered_variants, original_features)
        
        return filtered_variants
    
    def _basic_filter(self, variant: str, original: str) -> bool:
        """基本过滤条件"""
        # 长度检查
        if len(variant) > self.config.max_length * 4:  # 粗略字符长度限制
            return False
        
        # 空文本检查
        if not variant.strip():
            return False
        
        # 相同文本检查
        if variant.strip().lower() == original.strip().lower():
            return False
        
        # 基本语法检查（简单）
        if not re.search(r'[a-zA-Z]', variant):  # 必须包含字母
            return False
        
        return True
    
    def _semantic_filter(self, variant: str, original: str, original_features: torch.Tensor) -> bool:
        """语义相似度过滤"""
        try:
            variant_features = self.clip_model.encode_text([variant])
            similarity = torch.cosine_similarity(original_features, variant_features, dim=-1).item()
            
            # 相似度应该在合理范围内
            return (self.config.diversity_threshold < similarity < self.config.similarity_threshold)
            
        except Exception as e:
            logger.warning(f"语义过滤失败: {e}")
            return False
    
    def _rank_variants(self, original: str, variants: List[str], original_features: torch.Tensor) -> List[str]:
        """按语义相似度排序变体"""
        variant_scores = []
        
        for variant in variants:
            try:
                variant_features = self.clip_model.encode_text([variant])
                similarity = torch.cosine_similarity(original_features, variant_features, dim=-1).item()
                variant_scores.append((variant, similarity))
            except:
                variant_scores.append((variant, 0.0))
        
        # 按相似度降序排序
        variant_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [variant for variant, _ in variant_scores]
    
    def _extract_generated_text(self, response: str, original: str) -> Optional[str]:
        """从模型响应中提取生成的文本"""
        if not response:
            return None
        
        # 简单的文本提取逻辑
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and line != original and not line.startswith('请') and not line.startswith('用'):
                # 清理标点和格式
                cleaned = re.sub(r'^["""''']|["""''']$', '', line)

                cleaned = cleaned.strip('.,!?;: ')
                if cleaned:
                    return cleaned
        
        return None
    
    def _simple_paraphrases(self, text: str) -> List[str]:
        """简单的句式变换"""
        variants = []
        
        # 添加描述性词语
        descriptive_variants = [
            f"a photo of {text}",
            f"an image showing {text}",
            f"a picture of {text}",
            f"a scene with {text}"
        ]
        
        for variant in descriptive_variants:
            if variant != text and len(variant) <= self.config.max_length * 4:
                variants.append(variant)
        
        # 移除冗余词语（如果原文本已包含）
        if text.startswith(("a photo of", "an image", "a picture", "a scene")):
            # 尝试提取核心内容
            core_patterns = [
                r"a photo of (.+)",
                r"an image (?:showing|of) (.+)",
                r"a picture of (.+)",
                r"a scene with (.+)"
            ]
            
            for pattern in core_patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    core_text = match.group(1).strip()
                    if core_text:
                        variants.append(core_text)
                    break
        
        return variants
    
    def _load_synonym_dict(self) -> Dict[str, List[str]]:
        """加载同义词词典"""
        # 简化的同义词词典
        synonym_dict = {
            'cat': ['feline', 'kitten', 'kitty'],
            'dog': ['canine', 'puppy', 'hound'],
            'car': ['vehicle', 'automobile', 'auto'],
            'house': ['home', 'building', 'residence'],
            'person': ['individual', 'human', 'people'],
            'man': ['male', 'gentleman', 'guy'],
            'woman': ['female', 'lady', 'girl'],
            'child': ['kid', 'youngster', 'youth'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'mini'],
            'beautiful': ['pretty', 'lovely', 'gorgeous'],
            'old': ['elderly', 'aged', 'ancient'],
            'young': ['youthful', 'juvenile', 'new'],
            'red': ['crimson', 'scarlet', 'cherry'],
            'blue': ['azure', 'navy', 'cobalt'],
            'green': ['emerald', 'lime', 'forest'],
            'happy': ['joyful', 'cheerful', 'glad'],
            'sad': ['unhappy', 'sorrowful', 'melancholy'],
            'fast': ['quick', 'rapid', 'swift'],
            'slow': ['sluggish', 'gradual', 'leisurely']
        }
        
        return synonym_dict
    
    def _load_paraphrase_templates(self) -> List[str]:
        """加载改写模板"""
        templates = [
            "a view of {text}",
            "an image featuring {text}",
            "a photograph showing {text}",
            "a snapshot of {text}",
            "a depiction of {text}",
            "a representation of {text}"
        ]
        
        return templates
    
    def batch_generate_variants(self, texts: List[str]) -> List[List[str]]:
        """批量生成文本变体
        
        Args:
            texts: 文本列表
            
        Returns:
            变体列表的列表
        """
        all_variants = []
        
        for text in texts:
            variants = self.generate_variants(text)
            all_variants.append(variants)
        
        return all_variants
    
    def evaluate_variant_quality(self, original: str, variants: List[str]) -> Dict[str, Any]:
        """评估变体质量
        
        Args:
            original: 原始文本
            variants: 变体列表
            
        Returns:
            质量评估结果
        """
        if not variants:
            return {'message': '无变体可评估'}
        
        # 编码所有文本
        all_texts = [original] + variants
        all_features = self.clip_model.encode_text(all_texts)
        original_features = all_features[0:1]
        variant_features = all_features[1:]
        
        # 计算相似度
        similarities = torch.cosine_similarity(
            original_features.repeat(len(variants), 1),
            variant_features,
            dim=-1
        ).tolist()
        
        # 计算变体间多样性
        diversity_scores = []
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                sim = torch.cosine_similarity(
                    variant_features[i:i+1],
                    variant_features[j:j+1],
                    dim=-1
                ).item()
                diversity_scores.append(1 - sim)  # 多样性 = 1 - 相似度
        
        evaluation = {
            'variant_count': len(variants),
            'similarity_stats': {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities))
            },
            'diversity_stats': {
                'mean': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
                'std': float(np.std(diversity_scores)) if diversity_scores else 0.0
            },
            'quality_score': self._compute_quality_score(similarities, diversity_scores)
        }
        
        return evaluation
    
    def _compute_quality_score(self, similarities: List[float], diversity_scores: List[float]) -> float:
        """计算综合质量分数"""
        if not similarities:
            return 0.0
        
        # 相似度应该适中（不太高也不太低）
        ideal_similarity = 0.7
        similarity_penalty = np.mean([(abs(s - ideal_similarity)) for s in similarities])
        
        # 多样性应该尽可能高
        diversity_reward = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # 综合分数
        quality_score = diversity_reward - similarity_penalty
        
        return float(np.clip(quality_score, 0.0, 1.0))