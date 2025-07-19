"""数据验证模块

实现数据集验证和数据泄漏检测功能。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import hashlib
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import torch
from models import CLIPModel, CLIPConfig
from utils.metrics import SimilarityMetrics
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DataValidationConfig:
    """数据验证配置"""
    # 相似性阈值
    image_similarity_threshold: float = 0.95
    text_similarity_threshold: float = 0.9
    cross_modal_similarity_threshold: float = 0.85
    
    # 聚类参数
    clustering_eps: float = 0.1
    clustering_min_samples: int = 2
    
    # 特征提取
    use_clip_features: bool = True
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"
    
    # 文本特征
    use_tfidf: bool = True
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    
    # 验证规则
    check_duplicates: bool = True
    check_near_duplicates: bool = True
    check_data_leakage: bool = True
    check_distribution: bool = True
    check_quality: bool = True
    
    # 质量检查
    min_image_size: Tuple[int, int] = (32, 32)
    max_image_size: Tuple[int, int] = (2048, 2048)
    min_text_length: int = 5
    max_text_length: int = 1000
    
    # 输出配置
    save_reports: bool = True
    report_dir: Optional[str] = None
    verbose: bool = True


@dataclass
class ValidationResult:
    """验证结果"""
    # 基本统计
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    
    # 重复检测
    exact_duplicates: List[Tuple[int, int]] = None
    near_duplicates: List[Tuple[int, int, float]] = None
    
    # 数据泄漏
    potential_leakage: List[Dict[str, Any]] = None
    
    # 质量问题
    quality_issues: List[Dict[str, Any]] = None
    
    # 分布分析
    distribution_stats: Dict[str, Any] = None
    
    # 聚类结果
    clusters: Dict[str, Any] = None
    
    # 总体评估
    overall_score: float = 0.0
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.exact_duplicates is None:
            self.exact_duplicates = []
        if self.near_duplicates is None:
            self.near_duplicates = []
        if self.potential_leakage is None:
            self.potential_leakage = []
        if self.quality_issues is None:
            self.quality_issues = []
        if self.distribution_stats is None:
            self.distribution_stats = {}
        if self.clusters is None:
            self.clusters = {}
        if self.recommendations is None:
            self.recommendations = []


class DataValidator:
    """数据验证器"""
    
    def __init__(self, config: DataValidationConfig):
        """
        初始化数据验证器
        
        Args:
            config: 验证配置
        """
        self.config = config
        
        # 初始化CLIP模型
        self.clip_model = None
        if self.config.use_clip_features:
            self.clip_model = self._initialize_clip_model()
        
        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = None
        if self.config.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                stop_words='english'
            )
        
        # 相似性计算器
        self.similarity_metrics = SimilarityMetrics()
        
        # 缓存
        self.feature_cache = {}
        
        logger.info("数据验证器初始化完成")
    
    def _initialize_clip_model(self) -> Optional[CLIPModel]:
        """
        初始化CLIP模型
        
        Returns:
            CLIP模型实例
        """
        try:
            clip_config = CLIPConfig(
                model_name=self.config.clip_model,
                device=self.config.device
            )
            
            clip_model = CLIPModel(clip_config)
            logger.info(f"CLIP模型初始化完成: {self.config.clip_model}")
            
            return clip_model
            
        except Exception as e:
            logger.warning(f"CLIP模型初始化失败: {e}")
            return None
    
    def validate_dataset(self, images: List[Union[Image.Image, str]], 
                        texts: List[str], 
                        labels: Optional[List[Any]] = None,
                        split_info: Optional[Dict[str, List[int]]] = None) -> ValidationResult:
        """
        验证数据集
        
        Args:
            images: 图像列表（PIL图像或路径）
            texts: 文本列表
            labels: 标签列表
            split_info: 数据集分割信息 {'train': [indices], 'val': [indices], 'test': [indices]}
            
        Returns:
            验证结果
        """
        try:
            logger.info(f"开始验证数据集: {len(images)} 个样本")
            
            result = ValidationResult(
                total_samples=len(images)
            )
            
            # 基本质量检查
            if self.config.check_quality:
                self._check_data_quality(images, texts, result)
            
            # 重复检测
            if self.config.check_duplicates:
                self._detect_duplicates(images, texts, result)
            
            # 近似重复检测
            if self.config.check_near_duplicates:
                self._detect_near_duplicates(images, texts, result)
            
            # 数据泄漏检测
            if self.config.check_data_leakage and split_info:
                self._detect_data_leakage(images, texts, split_info, result)
            
            # 分布分析
            if self.config.check_distribution:
                self._analyze_distribution(images, texts, labels, result)
            
            # 计算总体评分
            self._compute_overall_score(result)
            
            # 生成建议
            self._generate_recommendations(result)
            
            # 保存报告
            if self.config.save_reports and self.config.report_dir:
                self._save_validation_report(result)
            
            logger.info(f"数据集验证完成，总体评分: {result.overall_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"数据集验证失败: {e}")
            return ValidationResult(total_samples=len(images))
    
    def _check_data_quality(self, images: List[Union[Image.Image, str]], 
                           texts: List[str], 
                           result: ValidationResult):
        """
        检查数据质量
        
        Args:
            images: 图像列表
            texts: 文本列表
            result: 验证结果
        """
        try:
            logger.info("检查数据质量...")
            
            valid_count = 0
            
            for i, (image, text) in enumerate(zip(images, texts)):
                issues = []
                
                # 检查图像
                if isinstance(image, str):
                    # 图像路径
                    if not Path(image).exists():
                        issues.append(f"图像文件不存在: {image}")
                    else:
                        try:
                            img = Image.open(image)
                            width, height = img.size
                            
                            if (width < self.config.min_image_size[0] or 
                                height < self.config.min_image_size[1]):
                                issues.append(f"图像尺寸过小: {width}x{height}")
                            
                            if (width > self.config.max_image_size[0] or 
                                height > self.config.max_image_size[1]):
                                issues.append(f"图像尺寸过大: {width}x{height}")
                                
                        except Exception as e:
                            issues.append(f"图像加载失败: {e}")
                elif isinstance(image, Image.Image):
                    width, height = image.size
                    
                    if (width < self.config.min_image_size[0] or 
                        height < self.config.min_image_size[1]):
                        issues.append(f"图像尺寸过小: {width}x{height}")
                    
                    if (width > self.config.max_image_size[0] or 
                        height > self.config.max_image_size[1]):
                        issues.append(f"图像尺寸过大: {width}x{height}")
                
                # 检查文本
                if not isinstance(text, str):
                    issues.append("文本不是字符串类型")
                elif len(text.strip()) < self.config.min_text_length:
                    issues.append(f"文本长度过短: {len(text.strip())}")
                elif len(text) > self.config.max_text_length:
                    issues.append(f"文本长度过长: {len(text)}")
                
                # 记录问题
                if issues:
                    result.quality_issues.append({
                        'index': i,
                        'issues': issues
                    })
                else:
                    valid_count += 1
            
            result.valid_samples = valid_count
            result.invalid_samples = len(images) - valid_count
            
            logger.info(f"质量检查完成: {valid_count}/{len(images)} 个有效样本")
            
        except Exception as e:
            logger.error(f"数据质量检查失败: {e}")
    
    def _detect_duplicates(self, images: List[Union[Image.Image, str]], 
                          texts: List[str], 
                          result: ValidationResult):
        """
        检测完全重复
        
        Args:
            images: 图像列表
            texts: 文本列表
            result: 验证结果
        """
        try:
            logger.info("检测完全重复...")
            
            # 文本重复检测
            text_hashes = {}
            for i, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in text_hashes:
                    result.exact_duplicates.append((text_hashes[text_hash], i))
                else:
                    text_hashes[text_hash] = i
            
            # 图像重复检测（基于哈希）
            image_hashes = {}
            for i, image in enumerate(images):
                try:
                    if isinstance(image, str):
                        img = Image.open(image)
                    else:
                        img = image
                    
                    # 简单的图像哈希
                    img_array = np.array(img.resize((8, 8)).convert('L'))
                    img_hash = hashlib.md5(img_array.tobytes()).hexdigest()
                    
                    if img_hash in image_hashes:
                        result.exact_duplicates.append((image_hashes[img_hash], i))
                    else:
                        image_hashes[img_hash] = i
                        
                except Exception as e:
                    logger.warning(f"图像哈希计算失败 (索引 {i}): {e}")
            
            logger.info(f"完全重复检测完成: 发现 {len(result.exact_duplicates)} 对重复")
            
        except Exception as e:
            logger.error(f"完全重复检测失败: {e}")
    
    def _detect_near_duplicates(self, images: List[Union[Image.Image, str]], 
                               texts: List[str], 
                               result: ValidationResult):
        """
        检测近似重复
        
        Args:
            images: 图像列表
            texts: 文本列表
            result: 验证结果
        """
        try:
            logger.info("检测近似重复...")
            
            # 文本近似重复检测
            if self.tfidf_vectorizer and len(texts) > 1:
                try:
                    text_features = self.tfidf_vectorizer.fit_transform(texts)
                    text_similarities = cosine_similarity(text_features)
                    
                    for i in range(len(texts)):
                        for j in range(i + 1, len(texts)):
                            similarity = text_similarities[i, j]
                            if similarity >= self.config.text_similarity_threshold:
                                result.near_duplicates.append((i, j, similarity))
                                
                except Exception as e:
                    logger.warning(f"文本相似性计算失败: {e}")
            
            # 图像近似重复检测
            if self.clip_model and len(images) > 1:
                try:
                    # 提取图像特征
                    image_features = []
                    valid_indices = []
                    
                    for i, image in enumerate(images):
                        try:
                            if isinstance(image, str):
                                img = Image.open(image)
                            else:
                                img = image
                            
                            features = self.clip_model.encode_image(img)
                            image_features.append(features)
                            valid_indices.append(i)
                            
                        except Exception as e:
                            logger.warning(f"图像特征提取失败 (索引 {i}): {e}")
                    
                    if len(image_features) > 1:
                        image_features = np.array(image_features)
                        image_similarities = cosine_similarity(image_features)
                        
                        for i in range(len(valid_indices)):
                            for j in range(i + 1, len(valid_indices)):
                                similarity = image_similarities[i, j]
                                if similarity >= self.config.image_similarity_threshold:
                                    result.near_duplicates.append((
                                        valid_indices[i], 
                                        valid_indices[j], 
                                        similarity
                                    ))
                                    
                except Exception as e:
                    logger.warning(f"图像相似性计算失败: {e}")
            
            logger.info(f"近似重复检测完成: 发现 {len(result.near_duplicates)} 对近似重复")
            
        except Exception as e:
            logger.error(f"近似重复检测失败: {e}")
    
    def _detect_data_leakage(self, images: List[Union[Image.Image, str]], 
                            texts: List[str], 
                            split_info: Dict[str, List[int]], 
                            result: ValidationResult):
        """
        检测数据泄漏
        
        Args:
            images: 图像列表
            texts: 文本列表
            split_info: 数据集分割信息
            result: 验证结果
        """
        try:
            logger.info("检测数据泄漏...")
            
            # 检查训练集和测试集之间的泄漏
            train_indices = set(split_info.get('train', []))
            test_indices = set(split_info.get('test', []))
            val_indices = set(split_info.get('val', []))
            
            # 文本泄漏检测
            if self.tfidf_vectorizer:
                try:
                    all_texts = [texts[i] for i in range(len(texts)) 
                                if i in train_indices or i in test_indices or i in val_indices]
                    all_indices = [i for i in range(len(texts)) 
                                  if i in train_indices or i in test_indices or i in val_indices]
                    
                    if len(all_texts) > 1:
                        text_features = self.tfidf_vectorizer.fit_transform(all_texts)
                        text_similarities = cosine_similarity(text_features)
                        
                        for i, idx1 in enumerate(all_indices):
                            for j, idx2 in enumerate(all_indices):
                                if i >= j:
                                    continue
                                
                                # 检查是否跨数据集
                                in_train1 = idx1 in train_indices
                                in_test1 = idx1 in test_indices
                                in_val1 = idx1 in val_indices
                                
                                in_train2 = idx2 in train_indices
                                in_test2 = idx2 in test_indices
                                in_val2 = idx2 in val_indices
                                
                                cross_split = (
                                    (in_train1 and (in_test2 or in_val2)) or
                                    (in_test1 and (in_train2 or in_val2)) or
                                    (in_val1 and (in_train2 or in_test2))
                                )
                                
                                if cross_split and text_similarities[i, j] >= self.config.text_similarity_threshold:
                                    split1 = 'train' if in_train1 else ('test' if in_test1 else 'val')
                                    split2 = 'train' if in_train2 else ('test' if in_test2 else 'val')
                                    
                                    result.potential_leakage.append({
                                        'type': 'text',
                                        'index1': idx1,
                                        'index2': idx2,
                                        'split1': split1,
                                        'split2': split2,
                                        'similarity': text_similarities[i, j]
                                    })
                                    
                except Exception as e:
                    logger.warning(f"文本泄漏检测失败: {e}")
            
            # 图像泄漏检测
            if self.clip_model:
                try:
                    # 提取跨分割的图像特征
                    cross_features = []
                    cross_indices = []
                    cross_splits = []
                    
                    for split_name, indices in split_info.items():
                        for idx in indices:
                            if idx < len(images):
                                try:
                                    if isinstance(images[idx], str):
                                        img = Image.open(images[idx])
                                    else:
                                        img = images[idx]
                                    
                                    features = self.clip_model.encode_image(img)
                                    cross_features.append(features)
                                    cross_indices.append(idx)
                                    cross_splits.append(split_name)
                                    
                                except Exception as e:
                                    logger.warning(f"图像特征提取失败 (索引 {idx}): {e}")
                    
                    if len(cross_features) > 1:
                        cross_features = np.array(cross_features)
                        image_similarities = cosine_similarity(cross_features)
                        
                        for i in range(len(cross_indices)):
                            for j in range(i + 1, len(cross_indices)):
                                if (cross_splits[i] != cross_splits[j] and 
                                    image_similarities[i, j] >= self.config.image_similarity_threshold):
                                    
                                    result.potential_leakage.append({
                                        'type': 'image',
                                        'index1': cross_indices[i],
                                        'index2': cross_indices[j],
                                        'split1': cross_splits[i],
                                        'split2': cross_splits[j],
                                        'similarity': image_similarities[i, j]
                                    })
                                    
                except Exception as e:
                    logger.warning(f"图像泄漏检测失败: {e}")
            
            logger.info(f"数据泄漏检测完成: 发现 {len(result.potential_leakage)} 个潜在泄漏")
            
        except Exception as e:
            logger.error(f"数据泄漏检测失败: {e}")
    
    def _analyze_distribution(self, images: List[Union[Image.Image, str]], 
                             texts: List[str], 
                             labels: Optional[List[Any]], 
                             result: ValidationResult):
        """
        分析数据分布
        
        Args:
            images: 图像列表
            texts: 文本列表
            labels: 标签列表
            result: 验证结果
        """
        try:
            logger.info("分析数据分布...")
            
            # 文本长度分布
            text_lengths = [len(text) for text in texts]
            result.distribution_stats['text_length'] = {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths),
                'median': np.median(text_lengths)
            }
            
            # 图像尺寸分布
            image_sizes = []
            for image in images:
                try:
                    if isinstance(image, str):
                        img = Image.open(image)
                    else:
                        img = image
                    image_sizes.append(img.size)
                except:
                    continue
            
            if image_sizes:
                widths = [size[0] for size in image_sizes]
                heights = [size[1] for size in image_sizes]
                
                result.distribution_stats['image_size'] = {
                    'width': {
                        'mean': np.mean(widths),
                        'std': np.std(widths),
                        'min': np.min(widths),
                        'max': np.max(widths)
                    },
                    'height': {
                        'mean': np.mean(heights),
                        'std': np.std(heights),
                        'min': np.min(heights),
                        'max': np.max(heights)
                    }
                }
            
            # 标签分布
            if labels:
                from collections import Counter
                label_counts = Counter(labels)
                result.distribution_stats['labels'] = {
                    'unique_labels': len(label_counts),
                    'label_distribution': dict(label_counts),
                    'most_common': label_counts.most_common(5)
                }
            
            logger.info("数据分布分析完成")
            
        except Exception as e:
            logger.error(f"数据分布分析失败: {e}")
    
    def _compute_overall_score(self, result: ValidationResult):
        """
        计算总体评分
        
        Args:
            result: 验证结果
        """
        try:
            score = 100.0
            
            # 质量分数
            if result.total_samples > 0:
                quality_score = (result.valid_samples / result.total_samples) * 100
                score = min(score, quality_score)
            
            # 重复惩罚
            duplicate_penalty = len(result.exact_duplicates) * 5
            near_duplicate_penalty = len(result.near_duplicates) * 2
            score -= (duplicate_penalty + near_duplicate_penalty)
            
            # 数据泄漏惩罚
            leakage_penalty = len(result.potential_leakage) * 10
            score -= leakage_penalty
            
            # 确保分数在0-100范围内
            result.overall_score = max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"计算总体评分失败: {e}")
            result.overall_score = 0.0
    
    def _generate_recommendations(self, result: ValidationResult):
        """
        生成改进建议
        
        Args:
            result: 验证结果
        """
        try:
            recommendations = []
            
            # 质量问题建议
            if result.invalid_samples > 0:
                recommendations.append(
                    f"发现 {result.invalid_samples} 个无效样本，建议检查并修复数据质量问题"
                )
            
            # 重复问题建议
            if result.exact_duplicates:
                recommendations.append(
                    f"发现 {len(result.exact_duplicates)} 对完全重复，建议删除重复样本"
                )
            
            if result.near_duplicates:
                recommendations.append(
                    f"发现 {len(result.near_duplicates)} 对近似重复，建议检查并考虑删除"
                )
            
            # 数据泄漏建议
            if result.potential_leakage:
                recommendations.append(
                    f"发现 {len(result.potential_leakage)} 个潜在数据泄漏，建议重新划分数据集"
                )
            
            # 分布建议
            if 'text_length' in result.distribution_stats:
                text_stats = result.distribution_stats['text_length']
                if text_stats['std'] > text_stats['mean']:
                    recommendations.append("文本长度分布不均匀，建议进行文本预处理或过滤")
            
            # 总体建议
            if result.overall_score < 70:
                recommendations.append("数据集质量较低，建议进行全面的数据清理")
            elif result.overall_score < 85:
                recommendations.append("数据集质量中等，建议针对性改进")
            else:
                recommendations.append("数据集质量良好")
            
            result.recommendations = recommendations
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
    
    def _save_validation_report(self, result: ValidationResult):
        """
        保存验证报告
        
        Args:
            result: 验证结果
        """
        try:
            report_dir = Path(self.config.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成报告文件名
            timestamp = int(time.time())
            report_file = report_dir / f"validation_report_{timestamp}.json"
            
            # 准备报告数据
            report_data = {
                'timestamp': timestamp,
                'config': self.config.__dict__,
                'results': {
                    'total_samples': result.total_samples,
                    'valid_samples': result.valid_samples,
                    'invalid_samples': result.invalid_samples,
                    'exact_duplicates_count': len(result.exact_duplicates),
                    'near_duplicates_count': len(result.near_duplicates),
                    'potential_leakage_count': len(result.potential_leakage),
                    'quality_issues_count': len(result.quality_issues),
                    'overall_score': result.overall_score,
                    'distribution_stats': result.distribution_stats,
                    'recommendations': result.recommendations
                },
                'details': {
                    'exact_duplicates': result.exact_duplicates,
                    'near_duplicates': [(i, j, float(sim)) for i, j, sim in result.near_duplicates],
                    'potential_leakage': result.potential_leakage,
                    'quality_issues': result.quality_issues
                }
            }
            
            # 保存报告
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"验证报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"保存验证报告失败: {e}")
    
    def validate_splits(self, train_data: Tuple[List, List], 
                       val_data: Tuple[List, List], 
                       test_data: Tuple[List, List]) -> Dict[str, Any]:
        """
        验证数据集分割
        
        Args:
            train_data: 训练数据 (images, texts)
            val_data: 验证数据 (images, texts)
            test_data: 测试数据 (images, texts)
            
        Returns:
            分割验证结果
        """
        try:
            # 合并所有数据
            all_images = train_data[0] + val_data[0] + test_data[0]
            all_texts = train_data[1] + val_data[1] + test_data[1]
            
            # 构建分割信息
            split_info = {
                'train': list(range(len(train_data[0]))),
                'val': list(range(len(train_data[0]), len(train_data[0]) + len(val_data[0]))),
                'test': list(range(len(train_data[0]) + len(val_data[0]), len(all_images)))
            }
            
            # 执行验证
            result = self.validate_dataset(all_images, all_texts, split_info=split_info)
            
            # 添加分割特定的统计
            split_stats = {
                'train_size': len(train_data[0]),
                'val_size': len(val_data[0]),
                'test_size': len(test_data[0]),
                'train_ratio': len(train_data[0]) / len(all_images),
                'val_ratio': len(val_data[0]) / len(all_images),
                'test_ratio': len(test_data[0]) / len(all_images)
            }
            
            return {
                'validation_result': result,
                'split_stats': split_stats
            }
            
        except Exception as e:
            logger.error(f"数据集分割验证失败: {e}")
            return {}


def create_data_validator(config: Optional[DataValidationConfig] = None) -> DataValidator:
    """
    创建数据验证器实例
    
    Args:
        config: 验证配置
        
    Returns:
        数据验证器实例
    """
    if config is None:
        config = DataValidationConfig()
    
    return DataValidator(config)