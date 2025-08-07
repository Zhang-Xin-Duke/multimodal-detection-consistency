"""文本攻击模块

实现基于文本的对抗性攻击方法，包括TextFooler、BERT-Attack等。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
import logging
import time
import re
import random
from collections import defaultdict
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string
from tqdm import tqdm

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
    
    # 批处理和多GPU优化
    enable_multi_gpu: bool = False    # 是否启用多GPU并行
    gpu_ids: Optional[List[int]] = None  # GPU设备ID列表，None表示使用所有可用GPU
    batch_size: int = 32              # 批处理大小
    batch_size_per_gpu: int = 8       # 每个GPU的批处理大小
    num_workers: int = 4              # 数据加载器工作进程数
    
    # 内存优化
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    mixed_precision: bool = True      # 是否使用混合精度训练
    pin_memory: bool = True           # 是否使用固定内存
    gradient_clip_value: float = 1.0  # 梯度裁剪阈值
    
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
        self.config = config
        
        # 设置设备和多GPU
        self._setup_devices()
        
        # 初始化CLIP模型
        self._initialize_clip_model(clip_model)
        
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
        
        logging.info(f"文本攻击器初始化完成，方法: {config.attack_method}，设备: {self.device}")
    
    def _setup_devices(self):
        """设置设备和多GPU配置"""
        if self.config.enable_multi_gpu and torch.cuda.device_count() > 1:
            if self.config.gpu_ids is None:
                self.config.gpu_ids = list(range(torch.cuda.device_count()))
            
            self.device = torch.device(f'cuda:{self.config.gpu_ids[0]}')
            self.gpu_ids = self.config.gpu_ids
            logging.info(f"启用多GPU模式，使用GPU: {self.gpu_ids}")
        else:
            self.device = torch.device(self.config.device)
            self.gpu_ids = None
            logging.info(f"使用单GPU模式，设备: {self.device}")
    
    def _initialize_clip_model(self, clip_model):
        """初始化CLIP模型"""
        self.clip_model = clip_model.to(self.device)
        
        # 多GPU封装
        if self.config.enable_multi_gpu and self.gpu_ids and len(self.gpu_ids) > 1:
            self.clip_model = nn.DataParallel(self.clip_model, device_ids=self.gpu_ids)
            logging.info(f"CLIP模型已封装为DataParallel，使用GPU: {self.gpu_ids}")
    
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
        真正的批量文本攻击实现
        
        Args:
            texts: 文本列表
            images: 图像列表（可选）
            target_texts: 目标文本列表（可选）
        
        Returns:
            攻击结果列表
        """
        logging.info(f"开始批量文本攻击 {len(texts)} 个样本")
        start_time = time.time()
        
        # 分批处理以适应GPU内存
        all_results = []
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size
        
        with tqdm(total=total_batches, desc="文本攻击批量处理进度") as pbar:
            for batch_idx in range(0, len(texts), self.config.batch_size):
                batch_end = min(batch_idx + self.config.batch_size, len(texts))
                
                batch_texts = texts[batch_idx:batch_end]
                batch_images = images[batch_idx:batch_end] if images else None
                batch_targets = target_texts[batch_idx:batch_end] if target_texts else None
                
                # 执行批量攻击
                batch_results = self._batch_text_attack(
                    batch_texts, batch_images, batch_targets
                )
                all_results.extend(batch_results)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '已处理': f'{batch_end}/{len(texts)}',
                    '成功率': f'{sum(r["success"] for r in all_results) / len(all_results):.2%}'
                })
        
        # 更新攻击统计
        successful_attacks = sum(1 for r in all_results if r['success'])
        self.attack_stats['total_attacks'] += len(texts)
        self.attack_stats['successful_attacks'] += successful_attacks
        
        total_time = time.time() - start_time
        logging.info(f"批量文本攻击完成，总时间: {total_time:.2f}s，成功率: {successful_attacks/len(texts):.2%}")
        
        return all_results
    
    def _batch_text_attack(self, batch_texts: List[str],
                          batch_images: Optional[List[torch.Tensor]] = None,
                          batch_targets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        执行批量文本攻击的核心逻辑
        
        Args:
            batch_texts: 批量文本列表
            batch_images: 批量图像列表（可选）
            batch_targets: 批量目标文本列表（可选）
        
        Returns:
            批量攻击结果列表
        """
        batch_size = len(batch_texts)
        results = []
        
        # 编码原始文本特征
        with torch.no_grad():
            if hasattr(self.clip_model, 'tokenize'):
                text_tokens = torch.cat([self.clip_model.tokenize(text).to(self.device) 
                                       for text in batch_texts])
                original_text_features = self.clip_model.encode_text(text_tokens)
            else:
                # 如果没有tokenize方法，使用encode_text直接编码
                original_text_features = self.clip_model.encode_text(batch_texts)
            
            original_text_features = original_text_features / original_text_features.norm(dim=-1, keepdim=True)
            
            # 编码目标文本特征（如果有）
            if batch_targets:
                if hasattr(self.clip_model, 'tokenize'):
                    target_tokens = torch.cat([self.clip_model.tokenize(target).to(self.device) 
                                             for target in batch_targets])
                    target_features = self.clip_model.encode_text(target_tokens)
                else:
                    target_features = self.clip_model.encode_text(batch_targets)
                target_features = target_features / target_features.norm(dim=-1, keepdim=True)
            else:
                target_features = None
        
        # 编码图像特征（如果有）
        if batch_images:
            with torch.no_grad():
                image_tensors = torch.stack([img.to(self.device) for img in batch_images])
                image_features = self.clip_model.encode_image(image_tensors)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            image_features = None
        
        # 使用混合精度训练
        if self.config.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        
        # 对每个文本执行攻击
        for i in range(batch_size):
            text = batch_texts[i]
            original_feature = original_text_features[i:i+1]
            target_feature = target_features[i:i+1] if target_features is not None else None
            image_feature = image_features[i:i+1] if image_features is not None else None
            
            # 执行具体的攻击方法
            if self.config.attack_method == 'textfooler':
                result = self._batch_textfooler_attack(
                    text, original_feature, target_feature, image_feature
                )
            elif self.config.attack_method == 'synonym_replacement':
                result = self._batch_synonym_replacement_attack(
                    text, original_feature, target_feature, image_feature
                )
            else:
                # 回退到单个攻击
                image = batch_images[i] if batch_images else None
                target = batch_targets[i] if batch_targets else None
                result = self.attack(text, image, target)
            
            results.append(result)
        
        return results
    
    def _batch_textfooler_attack(self, text: str, 
                                original_feature: torch.Tensor,
                                target_feature: Optional[torch.Tensor] = None,
                                image_feature: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        批量TextFooler攻击实现
        
        Args:
            text: 原始文本
            original_feature: 原始文本特征
            target_feature: 目标文本特征（可选）
            image_feature: 图像特征（可选）
        
        Returns:
            攻击结果
        """
        # 分词和词性标注
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # 计算词重要性
        word_importance = self._calculate_word_importance_batch(
            text, tokens, original_feature, target_feature, image_feature
        )
        
        # 按重要性排序
        sorted_indices = sorted(range(len(tokens)), 
                              key=lambda i: word_importance[i], reverse=True)
        
        # 初始化攻击信息
        attack_info = {
            'original_text': text,
            'adversarial_text': text,
            'success': False,
            'replacements': [],
            'iterations': 0,
            'similarity_drop': 0.0,
            'perturbation_rate': 0.0
        }
        
        adversarial_text = text
        replaced_words = []
        max_replacements = min(len(tokens), self.config.max_word_replacements)
        
        # 迭代攻击
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
                    score = self._evaluate_candidate_batch(
                        candidate_text, original_feature, target_feature, image_feature
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_replacement = {
                            'index': idx,
                            'original': word,
                            'replacement': candidate,
                            'score': score,
                            'text': candidate_text
                        }
            
            # 应用最佳替换
            if best_replacement and best_score > self.config.attack_threshold:
                adversarial_text = best_replacement['text']
                replaced_words.append(best_replacement)
                attack_info['replacements'] = replaced_words
                attack_info['adversarial_text'] = adversarial_text
                
                # 检查攻击成功
                if self._check_attack_success_batch(
                    adversarial_text, original_feature, target_feature, image_feature
                ):
                    attack_info['success'] = True
                    break
            else:
                break  # 没有找到有效替换
        
        # 计算最终指标
        attack_info['similarity_drop'] = self._calculate_similarity_drop_batch(
            text, adversarial_text, original_feature
        )
        attack_info['perturbation_rate'] = len(replaced_words) / len(tokens)
        
        return attack_info
    
    def _batch_synonym_replacement_attack(self, text: str,
                                         original_feature: torch.Tensor,
                                         target_feature: Optional[torch.Tensor] = None,
                                         image_feature: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        批量同义词替换攻击实现
        
        Args:
            text: 原始文本
            original_feature: 原始文本特征
            target_feature: 目标文本特征（可选）
            image_feature: 图像特征（可选）
        
        Returns:
            攻击结果
        """
        # 分词
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # 随机选择要替换的词
        replaceable_indices = []
        for i, (word, pos) in enumerate(pos_tags):
            if (word not in self.stop_words and 
                word not in string.punctuation and 
                len(word) >= 2):
                replaceable_indices.append(i)
        
        if not replaceable_indices:
            return {
                'original_text': text,
                'adversarial_text': text,
                'success': False,
                'replacements': [],
                'iterations': 1,
                'similarity_drop': 0.0,
                'perturbation_rate': 0.0
            }
        
        # 随机选择替换数量
        num_replacements = min(
            len(replaceable_indices),
            random.randint(1, self.config.max_word_replacements)
        )
        
        selected_indices = random.sample(replaceable_indices, num_replacements)
        
        # 执行替换
        adversarial_text = text
        replaced_words = []
        
        for idx in selected_indices:
            word = tokens[idx]
            pos = pos_tags[idx][1]
            
            # 获取同义词
            synonyms = self._get_synonyms(word, pos)
            if synonyms:
                replacement = random.choice(synonyms[:self.config.max_candidates])
                adversarial_text = self._replace_word_in_text(
                    adversarial_text, word, replacement
                )
                replaced_words.append({
                    'index': idx,
                    'original': word,
                    'replacement': replacement
                })
        
        # 检查攻击成功
        success = self._check_attack_success_batch(
            adversarial_text, original_feature, target_feature, image_feature
        )
        
        # 计算指标
        similarity_drop = self._calculate_similarity_drop_batch(
            text, adversarial_text, original_feature
        )
        
        return {
            'original_text': text,
            'adversarial_text': adversarial_text,
            'success': success,
            'replacements': replaced_words,
            'iterations': 1,
            'similarity_drop': similarity_drop,
            'perturbation_rate': len(replaced_words) / len(tokens)
        }
    
    def _calculate_word_importance_batch(self, text: str, tokens: List[str],
                                        original_feature: torch.Tensor,
                                        target_feature: Optional[torch.Tensor] = None,
                                        image_feature: Optional[torch.Tensor] = None) -> List[float]:
        """
        批量计算词重要性
        
        Args:
            text: 原始文本
            tokens: 分词结果
            original_feature: 原始文本特征
            target_feature: 目标文本特征（可选）
            image_feature: 图像特征（可选）
        
        Returns:
            词重要性列表
        """
        importance_scores = []
        
        for i, token in enumerate(tokens):
            # 创建删除当前词的文本
            masked_tokens = tokens[:i] + ['[MASK]'] + tokens[i+1:]
            masked_text = ' '.join(masked_tokens)
            
            # 编码掩码文本
            with torch.no_grad():
                if hasattr(self.clip_model, 'tokenize'):
                    masked_tokens_tensor = self.clip_model.tokenize(masked_text).to(self.device)
                    masked_feature = self.clip_model.encode_text(masked_tokens_tensor)
                else:
                    masked_feature = self.clip_model.encode_text([masked_text])
                masked_feature = masked_feature / masked_feature.norm(dim=-1, keepdim=True)
            
            # 计算特征差异
            if target_feature is not None:
                # 目标攻击：计算与目标的相似度变化
                original_sim = torch.cosine_similarity(original_feature, target_feature)
                masked_sim = torch.cosine_similarity(masked_feature, target_feature)
                importance = (masked_sim - original_sim).item()
            elif image_feature is not None:
                # 图像-文本攻击：计算与图像的相似度变化
                original_sim = torch.cosine_similarity(original_feature, image_feature)
                masked_sim = torch.cosine_similarity(masked_feature, image_feature)
                importance = (original_sim - masked_sim).item()
            else:
                # 无目标攻击：计算特征变化幅度
                importance = torch.norm(original_feature - masked_feature).item()
            
            importance_scores.append(importance)
        
        return importance_scores
    
    def _evaluate_candidate_batch(self, candidate_text: str,
                                 original_feature: torch.Tensor,
                                 target_feature: Optional[torch.Tensor] = None,
                                 image_feature: Optional[torch.Tensor] = None) -> float:
        """
        批量评估候选文本的攻击效果
        
        Args:
            candidate_text: 候选文本
            original_feature: 原始文本特征
            target_feature: 目标文本特征（可选）
            image_feature: 图像特征（可选）
        
        Returns:
            攻击效果分数
        """
        # 编码候选文本
        with torch.no_grad():
            if hasattr(self.clip_model, 'tokenize'):
                candidate_tokens = self.clip_model.tokenize(candidate_text).to(self.device)
                candidate_feature = self.clip_model.encode_text(candidate_tokens)
            else:
                candidate_feature = self.clip_model.encode_text([candidate_text])
            candidate_feature = candidate_feature / candidate_feature.norm(dim=-1, keepdim=True)
        
        if target_feature is not None:
            # 目标攻击：最大化与目标的相似度
            score = torch.cosine_similarity(candidate_feature, target_feature).item()
        elif image_feature is not None:
            # 图像-文本攻击：最小化与图像的相似度
            score = -torch.cosine_similarity(candidate_feature, image_feature).item()
        else:
            # 无目标攻击：最大化与原始文本的差异
            score = -torch.cosine_similarity(candidate_feature, original_feature).item()
        
        return score
    
    def _check_attack_success_batch(self, adversarial_text: str,
                                   original_feature: torch.Tensor,
                                   target_feature: Optional[torch.Tensor] = None,
                                   image_feature: Optional[torch.Tensor] = None) -> bool:
        """
        批量检查攻击是否成功
        
        Args:
            adversarial_text: 对抗文本
            original_feature: 原始文本特征
            target_feature: 目标文本特征（可选）
            image_feature: 图像特征（可选）
        
        Returns:
            是否攻击成功
        """
        # 编码对抗文本
        with torch.no_grad():
            if hasattr(self.clip_model, 'tokenize'):
                adv_tokens = self.clip_model.tokenize(adversarial_text).to(self.device)
                adv_feature = self.clip_model.encode_text(adv_tokens)
            else:
                adv_feature = self.clip_model.encode_text([adversarial_text])
            adv_feature = adv_feature / adv_feature.norm(dim=-1, keepdim=True)
        
        if target_feature is not None:
            # 目标攻击：检查与目标的相似度是否超过阈值
            target_sim = torch.cosine_similarity(adv_feature, target_feature).item()
            return target_sim > self.config.success_threshold
        elif image_feature is not None:
            # 图像-文本攻击：检查与图像的相似度是否下降
            original_sim = torch.cosine_similarity(original_feature, image_feature).item()
            adv_sim = torch.cosine_similarity(adv_feature, image_feature).item()
            return (original_sim - adv_sim) > self.config.success_threshold
        else:
            # 无目标攻击：检查与原始文本的相似度是否下降
            similarity = torch.cosine_similarity(adv_feature, original_feature).item()
            return similarity < (1.0 - self.config.success_threshold)
    
    def _calculate_similarity_drop_batch(self, original_text: str, 
                                        adversarial_text: str,
                                        original_feature: torch.Tensor) -> float:
        """
        批量计算相似度下降
        
        Args:
            original_text: 原始文本
            adversarial_text: 对抗文本
            original_feature: 原始文本特征
        
        Returns:
            相似度下降值
        """
        if original_text == adversarial_text:
            return 0.0
        
        # 编码对抗文本
        with torch.no_grad():
            if hasattr(self.clip_model, 'tokenize'):
                adv_tokens = self.clip_model.tokenize(adversarial_text).to(self.device)
                adv_feature = self.clip_model.encode_text(adv_tokens)
            else:
                adv_feature = self.clip_model.encode_text([adversarial_text])
            adv_feature = adv_feature / adv_feature.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        similarity = torch.cosine_similarity(original_feature, adv_feature).item()
        return 1.0 - similarity
    
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