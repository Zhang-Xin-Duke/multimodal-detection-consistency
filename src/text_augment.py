"""文本增强模块

提供多种文本增强方法，包括同义词替换、释义生成、句法变换和回译等。
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import random
import re
from abc import ABC, abstractmethod

# 导入相关模型
try:
    from .models.qwen_model import QwenModel, QwenConfig
except ImportError:
    QwenModel = None
    QwenConfig = None

# 导入第三方库
try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
except ImportError:
    nltk = None
    wordnet = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    pipeline = None

logger = logging.getLogger(__name__)


@dataclass
class TextAugmentConfig:
    """文本增强配置"""
    # 基础配置
    device: str = "cuda"
    batch_size: int = 8
    max_variants: int = 5
    min_similarity_threshold: float = 0.7
    max_similarity_threshold: float = 0.95
    
    # 增强方法配置
    enable_synonym_replacement: bool = True
    enable_paraphrase_generation: bool = True
    enable_syntax_transformation: bool = True
    enable_back_translation: bool = False
    
    # 同义词替换配置
    synonym_replacement_ratio: float = 0.3  # 替换词汇的比例
    min_word_length: int = 3  # 最小替换词长度
    
    # 释义生成配置
    paraphrase_model: str = "Qwen/Qwen2-1.5B-Instruct"
    paraphrase_temperature: float = 0.8
    paraphrase_max_length: int = 512
    
    # 句法变换配置
    syntax_transformation_ratio: float = 0.5
    
    # 回译配置
    back_translation_languages: List[str] = None
    translation_model: str = "Helsinki-NLP/opus-mt-en-de"
    
    # 质量控制
    enable_quality_filter: bool = True
    min_text_length: int = 5
    max_text_length: int = 1000
    filter_duplicates: bool = True
    
    def __post_init__(self):
        if self.back_translation_languages is None:
            self.back_translation_languages = ["de", "fr", "es"]


class BaseTextAugmenter(ABC):
    """文本增强器基类"""
    
    def __init__(self, config: TextAugmentConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    @abstractmethod
    def augment(self, text: str, num_variants: Optional[int] = None) -> List[str]:
        """生成文本变体"""
        pass
    
    def _filter_variants(self, original_text: str, variants: List[str]) -> List[str]:
        """过滤文本变体"""
        if not self.config.enable_quality_filter:
            return variants
        
        filtered = []
        seen = set()
        
        for variant in variants:
            # 基础过滤
            if not self._is_valid_text(variant):
                continue
            
            # 去重
            if self.config.filter_duplicates:
                variant_clean = self._normalize_text(variant)
                if variant_clean in seen or variant_clean == self._normalize_text(original_text):
                    continue
                seen.add(variant_clean)
            
            filtered.append(variant)
        
        return filtered
    
    def _is_valid_text(self, text: str) -> bool:
        """检查文本是否有效"""
        if not text or not text.strip():
            return False
        
        text_len = len(text.strip())
        if text_len < self.config.min_text_length or text_len > self.config.max_text_length:
            return False
        
        # 检查是否包含有意义的内容
        if len(re.findall(r'\w+', text)) < 2:
            return False
        
        return True
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本用于比较"""
        return re.sub(r'\s+', ' ', text.strip().lower())


class SynonymReplacer(BaseTextAugmenter):
    """同义词替换增强器"""
    
    def __init__(self, config: TextAugmentConfig):
        super().__init__(config)
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """确保NLTK数据可用"""
        if nltk is None:
            logger.warning("NLTK未安装，同义词替换功能不可用")
            return
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("下载NLTK punkt数据")
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("下载NLTK wordnet数据")
            nltk.download('wordnet')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            logger.info("下载NLTK POS标注器数据")
            nltk.download('averaged_perceptron_tagger')
    
    def augment(self, text: str, num_variants: Optional[int] = None) -> List[str]:
        """通过同义词替换生成文本变体"""
        if nltk is None or wordnet is None:
            logger.warning("NLTK不可用，跳过同义词替换")
            return []
        
        num_variants = num_variants or self.config.max_variants
        variants = []
        
        try:
            # 分词和词性标注
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # 生成多个变体
            for _ in range(num_variants * 2):  # 生成更多候选，然后过滤
                variant = self._replace_synonyms(tokens, pos_tags)
                if variant and variant != text:
                    variants.append(variant)
                
                if len(variants) >= num_variants:
                    break
            
            # 过滤和去重
            variants = self._filter_variants(text, variants)
            
        except Exception as e:
            logger.error(f"同义词替换失败: {e}")
        
        return variants[:num_variants]
    
    def _replace_synonyms(self, tokens: List[str], pos_tags: List[Tuple[str, str]]) -> str:
        """替换同义词"""
        new_tokens = tokens.copy()
        num_replacements = max(1, int(len(tokens) * self.config.synonym_replacement_ratio))
        
        # 随机选择要替换的词
        replacement_indices = random.sample(
            range(len(tokens)), 
            min(num_replacements, len(tokens))
        )
        
        for idx in replacement_indices:
            token = tokens[idx]
            pos = pos_tags[idx][1]
            
            # 只替换名词、动词、形容词和副词
            if (len(token) >= self.config.min_word_length and 
                pos.startswith(('NN', 'VB', 'JJ', 'RB'))):
                
                synonym = self._get_synonym(token, pos)
                if synonym:
                    new_tokens[idx] = synonym
        
        return ' '.join(new_tokens)
    
    def _get_synonym(self, word: str, pos: str) -> Optional[str]:
        """获取同义词"""
        try:
            # 转换POS标签到WordNet格式
            wn_pos = self._get_wordnet_pos(pos)
            if not wn_pos:
                return None
            
            # 获取同义词集合
            synsets = wordnet.synsets(word.lower(), pos=wn_pos)
            if not synsets:
                return None
            
            # 收集所有同义词
            synonyms = set()
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
            
            if synonyms:
                return random.choice(list(synonyms))
            
        except Exception as e:
            logger.debug(f"获取同义词失败 {word}: {e}")
        
        return None
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """转换TreeBank POS标签到WordNet格式"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


class ParaphraseGenerator(BaseTextAugmenter):
    """释义生成增强器"""
    
    def __init__(self, config: TextAugmentConfig):
        super().__init__(config)
        self.qwen_model = None
        self._load_model()
    
    def _load_model(self):
        """加载释义生成模型"""
        try:
            if QwenModel is not None and QwenConfig is not None:
                qwen_config = QwenConfig(
                    model_name=self.config.paraphrase_model,
                    device=self.config.device,
                    temperature=self.config.paraphrase_temperature,
                    max_length=self.config.paraphrase_max_length
                )
                self.qwen_model = QwenModel(qwen_config)
                logger.info("释义生成模型加载完成")
            else:
                logger.warning("Qwen模型不可用，释义生成功能受限")
        except Exception as e:
            logger.error(f"加载释义生成模型失败: {e}")
    
    def augment(self, text: str, num_variants: Optional[int] = None) -> List[str]:
        """生成释义变体"""
        if self.qwen_model is None:
            logger.warning("释义生成模型不可用")
            return []
        
        num_variants = num_variants or self.config.max_variants
        
        try:
            # 使用Qwen模型生成释义
            paraphrases = self.qwen_model.generate_paraphrases(
                text, 
                num_paraphrases=num_variants * 2,  # 生成更多候选
                temperature=self.config.paraphrase_temperature
            )
            
            # 过滤和去重
            variants = self._filter_variants(text, paraphrases)
            
            return variants[:num_variants]
            
        except Exception as e:
            logger.error(f"释义生成失败: {e}")
            return []


class SyntaxTransformer(BaseTextAugmenter):
    """句法变换增强器"""
    
    def augment(self, text: str, num_variants: Optional[int] = None) -> List[str]:
        """通过句法变换生成文本变体"""
        num_variants = num_variants or self.config.max_variants
        variants = []
        
        try:
            # 简单的句法变换
            if random.random() < self.config.syntax_transformation_ratio:
                # 主动语态转被动语态（简化实现）
                passive_variant = self._to_passive_voice(text)
                if passive_variant and passive_variant != text:
                    variants.append(passive_variant)
            
            # 句子重排（对于复合句）
            reordered_variant = self._reorder_clauses(text)
            if reordered_variant and reordered_variant != text:
                variants.append(reordered_variant)
            
            # 过滤变体
            variants = self._filter_variants(text, variants)
            
        except Exception as e:
            logger.error(f"句法变换失败: {e}")
        
        return variants[:num_variants]
    
    def _to_passive_voice(self, text: str) -> Optional[str]:
        """简单的主动转被动语态变换"""
        # 这是一个简化的实现，实际应用中需要更复杂的NLP处理
        patterns = [
            (r'(\w+)\s+(\w+ed|\w+s)\s+(\w+)', r'\3 is \2 by \1'),
            (r'(\w+)\s+makes?\s+(\w+)', r'\2 is made by \1'),
            (r'(\w+)\s+creates?\s+(\w+)', r'\2 is created by \1')
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return None
    
    def _reorder_clauses(self, text: str) -> Optional[str]:
        """重排句子中的从句"""
        # 简单的从句重排
        if ' and ' in text:
            parts = text.split(' and ')
            if len(parts) == 2:
                return f"{parts[1].strip()} and {parts[0].strip()}"
        
        if ' but ' in text:
            parts = text.split(' but ')
            if len(parts) == 2:
                return f"Although {parts[0].strip()}, {parts[1].strip()}"
        
        return None


class BackTranslator(BaseTextAugmenter):
    """回译增强器"""
    
    def __init__(self, config: TextAugmentConfig):
        super().__init__(config)
        self.translation_pipelines = {}
        self._load_models()
    
    def _load_models(self):
        """加载翻译模型"""
        if pipeline is None:
            logger.warning("transformers pipeline不可用，回译功能不可用")
            return
        
        try:
            for lang in self.config.back_translation_languages:
                # 加载英语到目标语言的翻译模型
                model_name = f"Helsinki-NLP/opus-mt-en-{lang}"
                self.translation_pipelines[f"en-{lang}"] = pipeline(
                    "translation", 
                    model=model_name,
                    device=0 if self.config.device == "cuda" else -1
                )
                
                # 加载目标语言到英语的翻译模型
                model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
                self.translation_pipelines[f"{lang}-en"] = pipeline(
                    "translation", 
                    model=model_name,
                    device=0 if self.config.device == "cuda" else -1
                )
            
            logger.info(f"回译模型加载完成: {self.config.back_translation_languages}")
            
        except Exception as e:
            logger.error(f"加载回译模型失败: {e}")
    
    def augment(self, text: str, num_variants: Optional[int] = None) -> List[str]:
        """通过回译生成文本变体"""
        if not self.translation_pipelines:
            logger.warning("回译模型不可用")
            return []
        
        num_variants = num_variants or self.config.max_variants
        variants = []
        
        try:
            for lang in self.config.back_translation_languages:
                if len(variants) >= num_variants:
                    break
                
                # 英语 -> 目标语言 -> 英语
                intermediate = self._translate(text, f"en-{lang}")
                if intermediate:
                    back_translated = self._translate(intermediate, f"{lang}-en")
                    if back_translated and back_translated != text:
                        variants.append(back_translated)
            
            # 过滤变体
            variants = self._filter_variants(text, variants)
            
        except Exception as e:
            logger.error(f"回译失败: {e}")
        
        return variants[:num_variants]
    
    def _translate(self, text: str, direction: str) -> Optional[str]:
        """翻译文本"""
        try:
            if direction not in self.translation_pipelines:
                return None
            
            result = self.translation_pipelines[direction](text)
            if result and len(result) > 0:
                return result[0]['translation_text']
            
        except Exception as e:
            logger.debug(f"翻译失败 {direction}: {e}")
        
        return None


class TextAugmenter:
    """文本增强器主类"""
    
    def __init__(self, config: Optional[TextAugmentConfig] = None):
        """
        初始化文本增强器
        
        Args:
            config: 文本增强配置
        """
        self.config = config or TextAugmentConfig()
        
        # 初始化各种增强器
        self.augmenters = []
        
        if self.config.enable_synonym_replacement:
            self.augmenters.append(SynonymReplacer(self.config))
        
        if self.config.enable_paraphrase_generation:
            self.augmenters.append(ParaphraseGenerator(self.config))
        
        if self.config.enable_syntax_transformation:
            self.augmenters.append(SyntaxTransformer(self.config))
        
        if self.config.enable_back_translation:
            self.augmenters.append(BackTranslator(self.config))
        
        logger.info(f"文本增强器初始化完成，启用{len(self.augmenters)}种增强方法")
    
    def augment(self, text: str, num_variants: Optional[int] = None) -> List[str]:
        """
        生成文本变体
        
        Args:
            text: 原始文本
            num_variants: 生成变体数量
            
        Returns:
            文本变体列表
        """
        if not text or not text.strip():
            return []
        
        num_variants = num_variants or self.config.max_variants
        all_variants = []
        
        # 使用所有启用的增强器
        for augmenter in self.augmenters:
            try:
                variants = augmenter.augment(text, num_variants)
                all_variants.extend(variants)
            except Exception as e:
                logger.error(f"增强器 {type(augmenter).__name__} 失败: {e}")
        
        # 去重和过滤
        unique_variants = []
        seen = set()
        original_normalized = self._normalize_text(text)
        
        for variant in all_variants:
            variant_normalized = self._normalize_text(variant)
            if (variant_normalized not in seen and 
                variant_normalized != original_normalized):
                unique_variants.append(variant)
                seen.add(variant_normalized)
        
        return unique_variants[:num_variants]
    
    def batch_augment(self, texts: List[str], 
                     num_variants: Optional[int] = None) -> List[List[str]]:
        """
        批量生成文本变体
        
        Args:
            texts: 文本列表
            num_variants: 每个文本的变体数量
            
        Returns:
            变体列表的列表
        """
        results = []
        
        for text in texts:
            try:
                variants = self.augment(text, num_variants)
                results.append(variants)
            except Exception as e:
                logger.error(f"批量增强失败: {e}")
                results.append([])
        
        return results
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本用于比较"""
        return re.sub(r'\s+', ' ', text.strip().lower())
    
    def get_augmenter_info(self) -> Dict[str, Any]:
        """
        获取增强器信息
        
        Returns:
            增强器信息字典
        """
        return {
            "num_augmenters": len(self.augmenters),
            "augmenter_types": [type(aug).__name__ for aug in self.augmenters],
            "config": self.config
        }


def create_text_augmenter(config: Optional[TextAugmentConfig] = None) -> TextAugmenter:
    """
    创建文本增强器实例
    
    Args:
        config: 文本增强配置
        
    Returns:
        文本增强器实例
    """
    return TextAugmenter(config)