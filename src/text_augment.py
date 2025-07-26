"""
文本增强模块

提供多种文本变体生成方法，用于增强文本查询的鲁棒性。
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import random
import re
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
from .models import QwenModel, QwenConfig, CLIPModel, CLIPConfig
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)


@dataclass
class TextAugmentConfig:
    """文本增强配置"""
    # 基础配置
    num_variants: int = 5
    similarity_threshold: float = 0.8
    max_attempts: int = 10
    
    # 同义词替换
    synonym_prob: float = 0.3
    max_synonyms_per_word: int = 3
    
    # 释义生成
    paraphrase_model: str = "Qwen/Qwen2-7B-Instruct"
    paraphrase_temperature: float = 0.8
    paraphrase_max_length: int = 512
    use_flash_attention: bool = False
    
    # 回译
    back_translation_languages: List[str] = None
    
    # 句法变换
    syntax_transform_prob: float = 0.2
    
    # 词汇替换
    lexical_substitution_prob: float = 0.25
    
    # 相似度计算
    similarity_model: str = "ViT-B/32"
    
    # 缓存
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.back_translation_languages is None:
            self.back_translation_languages = ['zh', 'fr', 'de', 'es']


from functools import lru_cache

class TextAugmenter:
    """文本增强器"""
    
    def __init__(self, config: TextAugmentConfig):
        """
        初始化文本增强器
        
        Args:
            config: 文本增强配置
        """
        self.config = config
        
        # 缓存
        self.variant_cache = {}
        self.similarity_cache = {}
        
        # 同义词词典
        self.synonym_dict = {}
        
        # 初始化组件
        self._initialize_models()
        self._load_synonym_dict()
        
        logger.info("文本增强器初始化完成")
    
    def _initialize_models(self):
        """
        初始化所需模型
        """
        try:
            # 初始化Qwen模型用于释义生成
            from .utils.config import get_qwen_cache_dir
            qwen_config = QwenConfig(
                model_name=self.config.paraphrase_model,
                temperature=self.config.paraphrase_temperature,
                max_length=self.config.paraphrase_max_length,
                use_flash_attention=self.config.use_flash_attention,
                cache_dir=get_qwen_cache_dir()  # 使用统一的缓存目录配置
            )
            self.qwen_model = QwenModel(qwen_config)
            
            # 初始化CLIP模型用于相似度计算
            clip_config = CLIPConfig(
                model_name=self.config.similarity_model
            )
            self.clip_model = CLIPModel(clip_config)
            
            logger.info("模型初始化完成")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def _load_synonym_dict(self):
        """
        加载同义词词典
        """
        try:
            # 这里可以加载预构建的同义词词典
            # 暂时使用WordNet作为同义词来源
            self.synonym_dict = {}
            logger.info("同义词词典加载完成")
            
        except Exception as e:
            logger.warning(f"加载同义词词典失败: {e}")
    
    def generate_variants(self, text: str, 
                         methods: Optional[List[str]] = None) -> List[str]:
        """
        生成文本变体
        
        Args:
            text: 原始文本
            methods: 使用的增强方法列表
            
        Returns:
            文本变体列表
        """
        if methods is None:
            methods = ['synonym', 'paraphrase', 'syntax', 'lexical']
        
        # 检查缓存
        cache_key = f"{text}_{'-'.join(sorted(methods))}"
        if self.config.enable_cache and cache_key in self.variant_cache:
            return self.variant_cache[cache_key]
        
        variants = []
        
        try:
            # 同义词替换
            if 'synonym' in methods:
                synonym_variants = self._generate_synonym_variants(text)
                variants.extend(synonym_variants)
            
            # 释义生成
            if 'paraphrase' in methods:
                paraphrase_variants = self._generate_paraphrase_variants(text)
                variants.extend(paraphrase_variants)
            
            # 句法变换
            if 'syntax' in methods:
                syntax_variants = self._generate_syntax_variants(text)
                variants.extend(syntax_variants)
            
            # 词汇替换
            if 'lexical' in methods:
                lexical_variants = self._generate_lexical_variants(text)
                variants.extend(lexical_variants)
            
            # 回译
            if 'back_translation' in methods:
                bt_variants = self._generate_back_translation_variants(text)
                variants.extend(bt_variants)
            
            # 去重并过滤
            variants = self._filter_variants(text, variants)
            
            # 限制数量
            if len(variants) > self.config.num_variants:
                variants = variants[:self.config.num_variants]
            
            # 缓存结果
            if self.config.enable_cache:
                self.variant_cache[cache_key] = variants
            
            logger.debug(f"生成文本变体: {text} -> {len(variants)}个变体")
            return variants
            
        except Exception as e:
            logger.error(f"生成文本变体失败: {e}")
            return []
    
    def _generate_synonym_variants(self, text: str) -> List[str]:
        """
        生成同义词替换变体
        
        Args:
            text: 原始文本
            
        Returns:
            同义词变体列表
        """
        variants = []
        
        try:
            # 分词和词性标注
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # 为每个词找同义词
            for _ in range(self.config.num_variants):
                variant_tokens = tokens.copy()
                
                for i, (word, pos) in enumerate(pos_tags):
                    if random.random() < self.config.synonym_prob:
                        synonyms = self._get_synonyms(word, pos)
                        if synonyms:
                            variant_tokens[i] = random.choice(synonyms)
                
                variant = ' '.join(variant_tokens)
                if variant != text.lower():
                    variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.warning(f"同义词替换失败: {e}")
            return []
    
    def _get_synonyms(self, word: str, pos: str) -> List[str]:
        """
        获取词的同义词
        
        Args:
            word: 单词
            pos: 词性
            
        Returns:
            同义词列表
        """
        synonyms = []
        
        try:
            # 转换词性标签
            wordnet_pos = self._get_wordnet_pos(pos)
            if wordnet_pos is None:
                return synonyms
            
            # 获取同义词集
            synsets = wordnet.synsets(word, pos=wordnet_pos)
            
            for synset in synsets[:self.config.max_synonyms_per_word]:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and synonym not in synonyms:
                        synonyms.append(synonym)
            
            return synonyms[:self.config.max_synonyms_per_word]
            
        except Exception as e:
            logger.debug(f"获取同义词失败 {word}: {e}")
            return []
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """
        将TreeBank词性标签转换为WordNet词性标签
        
        Args:
            treebank_tag: TreeBank标签
            
        Returns:
            WordNet标签
        """
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
    
    def _generate_paraphrase_variants(self, text: str) -> List[str]:
        """
        生成释义变体
        
        Args:
            text: 原始文本
            
        Returns:
            释义变体列表
        """
        # This is a placeholder for a batch-capable paraphrase generation.
        # The current implementation processes one by one.
        # For actual batch processing, you would need to adapt the model interaction.
        try:
            if self.qwen_model is None:
                return []

            prompt = f"请为以下文本生成{self.config.num_variants}个不同的表达方式，保持原意不变：\n\n{text}\n\n请用不同的词汇和句式重新表达，每个变体占一行，并以'- '开头："

            # This part should be batched.
            generated_text = self.qwen_model.generate_text([prompt])[0]

            variants = [v.strip() for v in generated_text.split('- ') if v.strip()] 

            return variants

        except Exception as e:
            logger.warning(f"释义生成失败: {e}")
            return []
    
    def _parse_generated_variants(self, generated_text: str, original_text: str) -> List[str]:
        """
        解析生成的变体文本
        
        Args:
            generated_text: 生成的文本
            original_text: 原始文本
            
        Returns:
            解析后的变体列表
        """
        variants = []
        
        try:
            # 按行分割
            lines = generated_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # 过滤无效行
                if not line or line == original_text:
                    continue
                
                # 移除编号和标点
                line = re.sub(r'^\d+[.)\s]*', '', line)
                line = re.sub(r'^[-*•]\s*', '', line)
                line = line.strip(string.punctuation + ' ')
                
                if line and len(line) > 5:  # 最小长度过滤
                    variants.append(line)
            
            return variants[:self.config.num_variants]
            
        except Exception as e:
            logger.warning(f"解析变体失败: {e}")
            return []
    
    def _generate_syntax_variants(self, text: str) -> List[str]:
        """
        生成句法变换变体
        
        Args:
            text: 原始文本
            
        Returns:
            句法变体列表
        """
        variants = []
        
        try:
            # 简单的句法变换
            # 1. 主被动语态转换（简化实现）
            if ' is ' in text or ' are ' in text:
                variant = text.replace(' is ', ' was ').replace(' are ', ' were ')
                variants.append(variant)
            
            # 2. 词序调整
            words = text.split()
            if len(words) > 3:
                # 随机调整词序
                for _ in range(2):
                    shuffled_words = words.copy()
                    # 保持前两个和后两个词的相对位置
                    middle = shuffled_words[2:-2]
                    if len(middle) > 1:
                        random.shuffle(middle)
                        variant = ' '.join(shuffled_words[:2] + middle + shuffled_words[-2:])
                        variants.append(variant)
            
            # 3. 添加修饰词
            modifiers = ['beautiful', 'amazing', 'wonderful', 'great', 'nice']
            if random.random() < self.config.syntax_transform_prob:
                modifier = random.choice(modifiers)
                variant = f"{modifier} {text}"
                variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.warning(f"句法变换失败: {e}")
            return []
    
    def _generate_lexical_variants(self, text: str) -> List[str]:
        """
        生成词汇替换变体
        
        Args:
            text: 原始文本
            
        Returns:
            词汇变体列表
        """
        variants = []
        
        try:
            # 词汇替换映射
            lexical_substitutions = {
                'cat': ['kitten', 'feline', 'kitty'],
                'dog': ['puppy', 'canine', 'hound'],
                'car': ['vehicle', 'automobile', 'auto'],
                'house': ['home', 'residence', 'dwelling'],
                'big': ['large', 'huge', 'enormous'],
                'small': ['tiny', 'little', 'mini'],
                'good': ['great', 'excellent', 'wonderful'],
                'bad': ['poor', 'terrible', 'awful'],
                'fast': ['quick', 'rapid', 'speedy'],
                'slow': ['sluggish', 'gradual', 'leisurely']
            }
            
            words = text.lower().split()
            
            for _ in range(self.config.num_variants):
                variant_words = words.copy()
                
                for i, word in enumerate(words):
                    if word in lexical_substitutions and random.random() < self.config.lexical_substitution_prob:
                        substitutes = lexical_substitutions[word]
                        variant_words[i] = random.choice(substitutes)
                
                variant = ' '.join(variant_words)
                if variant != text.lower():
                    variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.warning(f"词汇替换失败: {e}")
            return []
    
    def _generate_back_translation_variants(self, text: str) -> List[str]:
        """
        生成回译变体
        
        Args:
            text: 原始文本
            
        Returns:
            回译变体列表
        """
        # 这里需要集成翻译API，暂时返回空列表
        # 实际实现中可以使用Google Translate API或其他翻译服务
        logger.info("回译功能需要集成翻译API")
        return []
    
    def _filter_variants(self, original_text: str, variants: List[str]) -> List[str]:
        """
        过滤和排序变体
        
        Args:
            original_text: 原始文本
            variants: 变体列表
            
        Returns:
            过滤后的变体列表
        """
        try:
            if not variants:
                return []
            
            # 去重
            unique_variants = list(dict.fromkeys(variants))
            
            # 移除与原文相同的变体
            unique_variants = [v for v in unique_variants if v.lower() != original_text.lower()]
            
            # 计算相似度并过滤
            filtered_variants = []
            
            for variant in unique_variants:
                similarity = self.compute_similarity(original_text, variant)
                
                # 保留相似度在合理范围内的变体
                if 0.3 <= similarity <= self.config.similarity_threshold:
                    filtered_variants.append((variant, similarity))
            
            # 按相似度排序（降序）
            filtered_variants.sort(key=lambda x: x[1], reverse=True)
            
            # 返回变体文本
            return [variant for variant, _ in filtered_variants]
            
        except Exception as e:
            logger.warning(f"过滤变体失败: {e}")
            return variants
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 检查缓存
            cache_key = f"{text1}||{text2}"
            if self.config.enable_cache and cache_key in self.similarity_cache:
                return self.similarity_cache[cache_key]
            
            if self.clip_model is None:
                # 使用简单的字符串相似度
                return self._simple_similarity(text1, text2)
            
            # 使用CLIP计算语义相似度
            features1 = self.clip_model.encode_text([text1])
            features2 = self.clip_model.encode_text([text2])
            
            similarity = cosine_similarity(features1, features2)[0, 0]
            similarity = float(similarity)
            
            # 缓存结果
            if self.config.enable_cache:
                self.similarity_cache[cache_key] = similarity
            
            return similarity
            
        except Exception as e:
            logger.warning(f"计算相似度失败: {e}")
            return self._simple_similarity(text1, text2)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        简单的字符串相似度计算
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数
        """
        # 使用Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def batch_generate_variants(self, texts: List[str], 
                               methods: Optional[List[str]] = None) -> List[List[str]]:
        """
        批量生成文本变体
        
        Args:
            texts: 文本列表
            methods: 使用的增强方法列表
            
        Returns:
            变体列表的列表
        """
        # TODO: 实现真正的批处理，特别是对于模型调用部分
        all_variants = []
        
        for text in texts:
            variants = self.generate_variants(text, methods)
            all_variants.append(variants)
        
        return all_variants
    
    def filter_by_similarity(self, original_text: str, 
                           variants: List[str],
                           min_similarity: float = 0.3,
                           max_similarity: float = 0.9) -> List[str]:
        """
        根据相似度过滤变体
        
        Args:
            original_text: 原始文本
            variants: 变体列表
            min_similarity: 最小相似度
            max_similarity: 最大相似度
            
        Returns:
            过滤后的变体列表
        """
        filtered_variants = []
        
        for variant in variants:
            similarity = self.compute_similarity(original_text, variant)
            if min_similarity <= similarity <= max_similarity:
                filtered_variants.append(variant)
        
        return filtered_variants
    
    def save_cache(self, cache_path: str):
        """
        保存缓存
        
        Args:
            cache_path: 缓存文件路径
        """
        try:
            cache_data = {
                'variant_cache': self.variant_cache,
                'similarity_cache': self.similarity_cache
            }
            
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"缓存已保存: {cache_path}")
            
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def load_cache(self, cache_path: str):
        """
        加载缓存
        
        Args:
            cache_path: 缓存文件路径
        """
        try:
            cache_path = Path(cache_path)
            if not cache_path.exists():
                logger.warning(f"缓存文件不存在: {cache_path}")
                return
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.variant_cache = cache_data.get('variant_cache', {})
            self.similarity_cache = cache_data.get('similarity_cache', {})
            
            logger.info(f"缓存已加载: {cache_path}")
            
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
    
    def clear_cache(self):
        """
        清理缓存
        """
        self.variant_cache.clear()
        self.similarity_cache.clear()
        logger.info("缓存已清理")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'variant_cache_size': len(self.variant_cache),
            'similarity_cache_size': len(self.similarity_cache),
            'config': {
                'num_variants': self.config.num_variants,
                'similarity_threshold': self.config.similarity_threshold,
                'synonym_prob': self.config.synonym_prob,
                'paraphrase_model': self.config.paraphrase_model,
                'similarity_model': self.config.similarity_model
            }
        }


def create_text_augmenter(config: Optional[TextAugmentConfig] = None) -> TextAugmenter:
    """
    创建文本增强器实例
    
    Args:
        config: 文本增强配置
        
    Returns:
        文本增强器实例
    """
    if config is None:
        config = TextAugmentConfig()
    
    return TextAugmenter(config)