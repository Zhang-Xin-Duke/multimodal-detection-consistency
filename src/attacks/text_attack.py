"""文本攻击模块

实现基于文本的对抗性攻击方法，包括TextFooler、BERT-Attack等。
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
import logging
import re
import random
from collections import defaultdict
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string

# 确保NLTK数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

@dataclass
class TextAttackConfig:
    """文本攻击配置"""
    # 基础设置
    random_seed: int = 42
    device: str = 'cuda'
    
    # 攻击方法
    attack_method: str = 'textfooler'  # 'textfooler', 'bert_attack', 'synonym_replacement'
    
    # TextFooler参数
    max_candidates: int = 50          # 最大候选词数量
    max_replace_ratio: float = 0.2    # 最大替换比例
    min_similarity: float = 0.8       # 最小语义相似度
    
    # 同义词替换参数
    synonym_prob: float = 0.1         # 同义词替换概率
    use_wordnet: bool = True          # 是否使用WordNet
    
    # 约束参数
    preserve_stopwords: bool = True   # 保留停用词
    preserve_named_entities: bool = True  # 保留命名实体
    max_iterations: int = 20          # 最大迭代次数
    
    # 评估参数
    similarity_threshold: float = 0.7  # 攻击成功阈值
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000

class TextAttacker:
    """文本攻击器"""
    
    def __init__(self, clip_model, config: TextAttackConfig):
        """
        初始化文本攻击器
        
        Args:
            clip_model: CLIP模型实例
            config: 攻击配置
        """
        self.clip_model = clip_model
        self.config = config
        self.device = torch.device(config.device)
        
        # 设置随机种子
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        
        # 初始化停用词
        self.stop_words = set(stopwords.words('english')) if config.preserve_stopwords else set()
        
        # 初始化统计信息
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'average_replacements': 0.0,
            'average_iterations': 0.0
        }
        
        # 缓存
        self.cache = {} if config.enable_cache else None
        self.synonym_cache = {}  # 同义词缓存
        
        logging.info(f"文本攻击器初始化完成，方法: {config.attack_method}")
    
    def attack(self, text: str, 
               image: Optional[torch.Tensor] = None,
               target_text: Optional[str] = None) -> Dict[str, Any]:
        """
        执行文本攻击
        
        Args:
            text: 原始文本
            image: 对应的图像（可选）
            target_text: 目标文本（目标攻击时使用）
        
        Returns:
            攻击结果字典
        """
        # 检查缓存
        cache_key = self._get_cache_key(text, target_text)
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 根据攻击方法选择具体实现
        if self.config.attack_method == 'textfooler':
            result = self._textfooler_attack(text, image, target_text)
        elif self.config.attack_method == 'synonym_replacement':
            result = self._synonym_replacement_attack(text, image, target_text)
        else:
            raise ValueError(f"不支持的攻击方法: {self.config.attack_method}")
        
        # 缓存结果
        if self.cache and len(self.cache) < self.config.cache_size:
            self.cache[cache_key] = result
        
        return result
    
    def _textfooler_attack(self, text: str, 
                          image: Optional[torch.Tensor] = None,
                          target_text: Optional[str] = None) -> Dict[str, Any]:
        """
        TextFooler攻击实现
        
        Args:
            text: 原始文本
            image: 对应的图像
            target_text: 目标文本
        
        Returns:
            攻击结果
        """
        # 分词和词性标注
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # 获取原始文本的特征
        with torch.no_grad():
            original_features = self.clip_model.encode_text([text])
            if target_text:
                target_features = self.clip_model.encode_text([target_text])
            else:
                target_features = None
        
        # 计算词重要性
        word_importance = self._calculate_word_importance(text, tokens, image)
        
        # 按重要性排序
        sorted_indices = sorted(range(len(tokens)), 
                              key=lambda i: word_importance[i], reverse=True)
        
        adversarial_text = text
        replaced_words = []
        attack_info = {
            'iterations': 0,
            'replacements': [],
            'similarity_history': [],
            'success': False
        }
        
        max_replacements = int(len(tokens) * self.config.max_replace_ratio)
        
        for iteration in range(self.config.max_iterations):
            attack_info['iterations'] = iteration + 1
            best_replacement = None
            best_score = float('-inf')
            
            # 尝试替换每个重要词
            for idx in sorted_indices[:max_replacements]:
                if idx in [r['index'] for r in replaced_words]:
                    continue  # 跳过已替换的词
                
                word = tokens[idx]
                pos = pos_tags[idx][1]
                
                # 跳过停用词和标点符号
                if (word in self.stop_words or 
                    word in string.punctuation or 
                    len(word) < 2):
                    continue
                
                # 获取候选同义词
                candidates = self._get_synonyms(word, pos)
                
                # 评估每个候选词
                for candidate in candidates[:self.config.max_candidates]:
                    if candidate == word:
                        continue
                    
                    # 生成候选文本
                    candidate_text = self._replace_word_in_text(
                        adversarial_text, word, candidate
                    )
                    
                    # 计算攻击效果
                    score = self._evaluate_candidate(
                        candidate_text, original_features, target_features
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_replacement = {
                            'index': idx,
                            'original': word,
                            'replacement': candidate,
                            'text': candidate_text,
                            'score': score
                        }
            
            # 应用最佳替换
            if best_replacement:
                adversarial_text = best_replacement['text']
                replaced_words.append(best_replacement)
                attack_info['replacements'].append(best_replacement)
                
                # 检查攻击是否成功
                success = self._check_attack_success(
                    adversarial_text, original_features, target_features
                )
                
                attack_info['similarity_history'].append(best_replacement['score'])
                
                if success:
                    attack_info['success'] = True
                    break
            else:
                break  # 没有找到有效替换
        
        # 更新统计信息
        self._update_stats(attack_info)
        
        return {
            'adversarial_text': adversarial_text,
            'original_text': text,
            'success': attack_info['success'],
            'attack_info': attack_info,
            'config': self.config
        }
    
    def _synonym_replacement_attack(self, text: str,
                                  image: Optional[torch.Tensor] = None,
                                  target_text: Optional[str] = None) -> Dict[str, Any]:
        """
        简单的同义词替换攻击
        
        Args:
            text: 原始文本
            image: 对应的图像
            target_text: 目标文本
        
        Returns:
            攻击结果
        """
        tokens = word_tokenize(text.lower())
        adversarial_tokens = tokens.copy()
        replaced_count = 0
        
        attack_info = {
            'iterations': 1,
            'replacements': [],
            'success': False
        }
        
        for i, token in enumerate(tokens):
            if (token in self.stop_words or 
                token in string.punctuation or 
                len(token) < 2):
                continue
            
            if random.random() < self.config.synonym_prob:
                synonyms = self._get_synonyms(token)
                if synonyms:
                    replacement = random.choice(synonyms)
                    adversarial_tokens[i] = replacement
                    replaced_count += 1
                    
                    attack_info['replacements'].append({
                        'index': i,
                        'original': token,
                        'replacement': replacement
                    })
        
        adversarial_text = ' '.join(adversarial_tokens)
        
        # 评估攻击成功性
        with torch.no_grad():
            original_features = self.clip_model.encode_text([text])
            adv_features = self.clip_model.encode_text([adversarial_text])
            
            similarity = torch.cosine_similarity(original_features, adv_features).item()
            attack_info['success'] = similarity < self.config.similarity_threshold
        
        # 更新统计信息
        self._update_stats(attack_info)
        
        return {
            'adversarial_text': adversarial_text,
            'original_text': text,
            'success': attack_info['success'],
            'attack_info': attack_info,
            'config': self.config
        }
    
    def _calculate_word_importance(self, text: str, tokens: List[str],
                                 image: Optional[torch.Tensor] = None) -> List[float]:
        """
        计算词的重要性分数
        
        Args:
            text: 原始文本
            tokens: 分词结果
            image: 对应的图像
        
        Returns:
            词重要性分数列表
        """
        importance_scores = []
        
        with torch.no_grad():
            original_features = self.clip_model.encode_text([text])
        
        for i, token in enumerate(tokens):
            # 创建删除当前词的文本
            modified_tokens = tokens[:i] + tokens[i+1:]
            modified_text = ' '.join(modified_tokens)
            
            if not modified_text.strip():
                importance_scores.append(0.0)
                continue
            
            # 计算删除该词后的特征变化
            with torch.no_grad():
                modified_features = self.clip_model.encode_text([modified_text])
                similarity = torch.cosine_similarity(
                    original_features, modified_features
                ).item()
                
                # 重要性 = 1 - 相似度（删除重要词会导致相似度下降）
                importance = 1.0 - similarity
                importance_scores.append(max(0.0, importance))
        
        return importance_scores
    
    def _get_synonyms(self, word: str, pos: Optional[str] = None) -> List[str]:
        """
        获取词的同义词
        
        Args:
            word: 目标词
            pos: 词性标签
        
        Returns:
            同义词列表
        """
        # 检查缓存
        cache_key = f"{word}_{pos}"
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]
        
        synonyms = set()
        
        if self.config.use_wordnet:
            # 使用WordNet获取同义词
            synsets = wordnet.synsets(word)
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != word and len(synonym) > 1:
                        synonyms.add(synonym)
        
        synonym_list = list(synonyms)
        
        # 缓存结果
        self.synonym_cache[cache_key] = synonym_list
        
        return synonym_list
    
    def _replace_word_in_text(self, text: str, old_word: str, new_word: str) -> str:
        """
        在文本中替换词
        
        Args:
            text: 原始文本
            old_word: 要替换的词
            new_word: 新词
        
        Returns:
            替换后的文本
        """
        # 使用正则表达式进行词边界匹配
        pattern = r'\b' + re.escape(old_word) + r'\b'
        return re.sub(pattern, new_word, text, flags=re.IGNORECASE)
    
    def _evaluate_candidate(self, candidate_text: str,
                          original_features: torch.Tensor,
                          target_features: Optional[torch.Tensor] = None) -> float:
        """
        评估候选文本的攻击效果
        
        Args:
            candidate_text: 候选文本
            original_features: 原始文本特征
            target_features: 目标文本特征
        
        Returns:
            攻击效果分数
        """
        with torch.no_grad():
            candidate_features = self.clip_model.encode_text([candidate_text])
            
            if target_features is not None:
                # 目标攻击：最大化与目标的相似度
                target_sim = torch.cosine_similarity(
                    candidate_features, target_features
                ).item()
                original_sim = torch.cosine_similarity(
                    candidate_features, original_features
                ).item()
                return target_sim - original_sim
            else:
                # 非目标攻击：最小化与原始的相似度
                similarity = torch.cosine_similarity(
                    candidate_features, original_features
                ).item()
                return -similarity
    
    def _check_attack_success(self, adversarial_text: str,
                            original_features: torch.Tensor,
                            target_features: Optional[torch.Tensor] = None) -> bool:
        """
        检查攻击是否成功
        
        Args:
            adversarial_text: 对抗文本
            original_features: 原始文本特征
            target_features: 目标文本特征
        
        Returns:
            攻击是否成功
        """
        with torch.no_grad():
            adv_features = self.clip_model.encode_text([adversarial_text])
            
            if target_features is not None:
                # 目标攻击：检查是否与目标更相似
                target_sim = torch.cosine_similarity(adv_features, target_features).item()
                original_sim = torch.cosine_similarity(adv_features, original_features).item()
                return target_sim > original_sim
            else:
                # 非目标攻击：检查相似度是否低于阈值
                similarity = torch.cosine_similarity(adv_features, original_features).item()
                return similarity < self.config.similarity_threshold
    
    def batch_attack(self, texts: List[str],
                    images: Optional[List[torch.Tensor]] = None,
                    target_texts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        批量文本攻击
        
        Args:
            texts: 文本列表
            images: 图像列表（可选）
            target_texts: 目标文本列表（可选）
        
        Returns:
            攻击结果列表
        """
        results = []
        
        for i, text in enumerate(texts):
            image = images[i] if images else None
            target_text = target_texts[i] if target_texts else None
            
            result = self.attack(text, image, target_text)
            results.append(result)
        
        return results
    
    def _get_cache_key(self, text: str, target_text: Optional[str] = None) -> str:
        """
        生成缓存键
        
        Args:
            text: 原始文本
            target_text: 目标文本
        
        Returns:
            缓存键
        """
        text_hash = hash(text)
        target_hash = hash(target_text) if target_text else 0
        method_hash = hash(self.config.attack_method)
        
        return f"text_{text_hash}_{target_hash}_{method_hash}"
    
    def _update_stats(self, attack_info: Dict):
        """
        更新攻击统计信息
        
        Args:
            attack_info: 攻击信息
        """
        self.attack_stats['total_attacks'] += 1
        if attack_info['success']:
            self.attack_stats['successful_attacks'] += 1
        
        # 更新平均值
        total = self.attack_stats['total_attacks']
        num_replacements = len(attack_info.get('replacements', []))
        
        self.attack_stats['average_replacements'] = (
            (self.attack_stats['average_replacements'] * (total - 1) + 
             num_replacements) / total
        )
        
        self.attack_stats['average_iterations'] = (
            (self.attack_stats['average_iterations'] * (total - 1) + 
             attack_info['iterations']) / total
        )
    
    def get_attack_stats(self) -> Dict[str, Any]:
        """
        获取攻击统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.attack_stats.copy()
        if stats['total_attacks'] > 0:
            stats['success_rate'] = stats['successful_attacks'] / stats['total_attacks']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'average_replacements': 0.0,
            'average_iterations': 0.0
        }
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
        self.synonym_cache.clear()

def create_text_attacker(clip_model, config: Optional[TextAttackConfig] = None) -> TextAttacker:
    """
    创建文本攻击器的工厂函数
    
    Args:
        clip_model: CLIP模型实例
        config: 攻击配置，如果为None则使用默认配置
    
    Returns:
        文本攻击器实例
    """
    if config is None:
        config = TextAttackConfig()
    
    return TextAttacker(clip_model, config)