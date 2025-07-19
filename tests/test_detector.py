"""检测器模块测试

测试AdversarialDetector类的各种功能。
"""

import unittest
import numpy as np
import torch
import tempfile
import os
from PIL import Image

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from detector import (
    AdversarialDetector, DetectorConfig, AggregationMethod, ThresholdMethod,
    AdaptiveThresholdManager, EnsembleDetector
)
from models import CLIPModel
from text_augment import TextAugmenter
from sd_ref import SDReferenceGenerator


class TestDetectorConfig(unittest.TestCase):
    """测试DetectorConfig配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = DetectorConfig()
        
        self.assertEqual(config.clip_model, "ViT-B/32")
        self.assertEqual(config.device, "cuda")
        self.assertTrue(config.use_text_variants)
        self.assertTrue(config.use_sd_reference)
        self.assertEqual(config.text_similarity_threshold, 0.85)
        self.assertEqual(config.reference_similarity_threshold, 0.75)
        self.assertEqual(config.consistency_threshold, 0.8)
        self.assertEqual(config.score_aggregation, "weighted_mean")
        self.assertTrue(config.adaptive_threshold)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = DetectorConfig(
            clip_model="ViT-L/14",
            device="cpu",
            text_similarity_threshold=0.9,
            reference_similarity_threshold=0.8,
            consistency_threshold=0.7,
            score_aggregation="max",
            adaptive_threshold=True
        )
        
        self.assertEqual(config.clip_model, "ViT-L/14")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.text_similarity_threshold, 0.9)
        self.assertEqual(config.reference_similarity_threshold, 0.8)
        self.assertEqual(config.consistency_threshold, 0.7)
        self.assertEqual(config.score_aggregation, "max")
        self.assertTrue(config.adaptive_threshold)


class TestAdversarialDetector(unittest.TestCase):
    """测试AdversarialDetector类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = DetectorConfig(
            device="cuda",
            batch_size=4
        )
        
        # 创建detector实例（使用真实模型）
        self.detector = AdversarialDetector(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector.config)
        self.assertIsNotNone(self.detector.clip_model)
        self.assertIsNotNone(self.detector.text_augmenter)
        self.assertIsNotNone(self.detector.sd_generator)
        self.assertEqual(len(self.detector.detection_cache), 0)
        self.assertEqual(self.detector.detection_stats['total_detections'], 0)
    
    def test_compute_consistency_score(self):
        """测试一致性分数计算"""
        # 准备测试数据
        candidate_vector = np.random.rand(512).astype(np.float32)
        reference_vectors = np.random.rand(5, 512).astype(np.float32)
        
        # 测试不同聚合方法
        score_mean = self.detector._compute_consistency_score(
            candidate_vector, reference_vectors, AggregationMethod.MEAN
        )
        score_max = self.detector._compute_consistency_score(
            candidate_vector, reference_vectors, AggregationMethod.MAX
        )
        score_min = self.detector._compute_consistency_score(
            candidate_vector, reference_vectors, AggregationMethod.MIN
        )
        score_weighted = self.detector._compute_consistency_score(
            candidate_vector, reference_vectors, AggregationMethod.WEIGHTED_AVERAGE
        )
        
        # 验证分数范围
        for score in [score_mean, score_max, score_min, score_weighted]:
            self.assertIsInstance(score, (float, np.floating))
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)
        
        # 验证聚合关系
        self.assertLessEqual(score_min, score_mean)
        self.assertLessEqual(score_mean, score_max)
    
    def test_compute_similarity_scores(self):
        """测试相似度分数计算"""
        # 准备测试数据
        candidate_vector = np.random.rand(512).astype(np.float32)
        reference_vectors = np.random.rand(3, 512).astype(np.float32)
        
        similarities = self.detector._compute_similarity_scores(
            candidate_vector, reference_vectors
        )
        
        # 验证结果
        self.assertEqual(len(similarities), 3)
        self.assertTrue(all(isinstance(sim, (float, np.floating)) for sim in similarities))
        self.assertTrue(all(-1.0 <= sim <= 1.0 for sim in similarities))
    
    def test_detect_adversarial_sample(self):
        """测试对抗样本检测"""
        # 创建测试图像
        img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        
        # 准备测试数据
        text = "a cat sitting on a chair"
        
        # 执行检测
        result = self.detector.detect_adversarial(img, text)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn('is_adversarial', result)
        self.assertIn('aggregated_score', result)
        
        self.assertIsInstance(result['is_adversarial'], bool)
        self.assertIsInstance(result['aggregated_score'], (float, np.floating))
        self.assertGreaterEqual(result['aggregated_score'], 0.0)
        self.assertLessEqual(result['aggregated_score'], 1.0)
    
    def test_batch_detect_adversarial_samples(self):
        """测试批量对抗样本检测"""
        # 创建临时测试图像
        temp_dir = tempfile.mkdtemp()
        images = []
        
        try:
            # 创建2个测试图像
            for i in range(2):
                img = Image.new('RGB', (224, 224), color=(i*100, i*100, i*100))
                images.append(img)
            
            # 准备测试数据
            texts = ["a cat sitting", "a dog running"]
            
            # 执行批量检测
            results = self.detector.batch_detect(
                images, texts
            )
            
            # 验证结果
            self.assertEqual(len(results), 2)
            
            for result in results:
                self.assertIsInstance(result, dict)
                self.assertIn('is_adversarial', result)
                self.assertIn('confidence', result)
                self.assertIn('scores', result)
                self.assertIn('details', result)
                
        finally:
            # 清理临时目录
            os.rmdir(temp_dir)
    
    def test_calibrate_threshold(self):
        """测试阈值校准"""
        # 准备校准数据
        clean_scores = np.array([0.9, 0.85, 0.8, 0.88, 0.92])
        adversarial_scores = np.array([0.3, 0.4, 0.35, 0.25, 0.45])
        
        # 执行阈值校准
        threshold = self.detector.calibrate_threshold(
            clean_scores, adversarial_scores, method="optimal"
        )
        
        # 验证阈值
        self.assertIsInstance(threshold, (float, np.floating))
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        
        # 测试其他校准方法
        threshold_percentile = self.detector.calibrate_threshold(
            clean_scores, adversarial_scores, method="percentile", percentile=95
        )
        
        self.assertIsInstance(threshold_percentile, (float, np.floating))
    
    def test_update_adaptive_threshold(self):
        """测试自适应阈值更新"""
        # 准备测试数据
        recent_scores = np.array([0.8, 0.75, 0.9, 0.85, 0.7])
        labels = np.array([1, 1, 0, 0, 1])  # 1表示干净样本，0表示对抗样本
        
        # 执行自适应更新
        old_threshold = self.detector.config.consistency_threshold
        self.detector.update_adaptive_threshold(recent_scores, labels)
        
        # 验证阈值可能发生变化（取决于具体实现）
        new_threshold = self.detector.config.consistency_threshold
        self.assertIsInstance(new_threshold, (float, np.floating))
        self.assertGreater(new_threshold, 0.0)
        self.assertLess(new_threshold, 1.0)
    
    def test_get_detection_statistics(self):
        """测试检测统计信息"""
        # 模拟一些检测历史
        self.detector.detection_stats['total_detections'] = 100
        self.detector.detection_stats['adversarial_detected'] = 15
        self.detector.detection_stats['clean_detected'] = 85
        
        stats = self.detector.get_detection_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_detections', stats)
        self.assertIn('adversarial_detected', stats)
        self.assertIn('clean_detected', stats)
        self.assertIn('detection_rate', stats)
        
        self.assertEqual(stats['total_detections'], 100)
        self.assertEqual(stats['adversarial_detected'], 15)
        self.assertEqual(stats['clean_detected'], 85)
        self.assertAlmostEqual(stats['detection_rate'], 0.15, places=2)
    
    def test_clear_cache(self):
        """测试缓存清理"""
        # 添加一些缓存数据
        self.detector.detection_cache['test_key'] = 'test_value'
        self.assertEqual(len(self.detector.detection_cache), 1)
        
        # 清理缓存
        self.detector.clear_cache()
        self.assertEqual(len(self.detector.detection_cache), 0)


class TestAdaptiveThresholdManager(unittest.TestCase):
    """测试AdaptiveThresholdManager类"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = AdaptiveThresholdManager(
            initial_threshold=0.7,
            adaptation_rate=0.1,
            min_threshold=0.3,
            max_threshold=0.9
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.manager.current_threshold, 0.7)
        self.assertEqual(self.manager.adaptation_rate, 0.1)
        self.assertEqual(self.manager.min_threshold, 0.3)
        self.assertEqual(self.manager.max_threshold, 0.9)
        self.assertEqual(len(self.manager.score_history), 0)
    
    def test_update_threshold(self):
        """测试阈值更新"""
        # 准备测试数据
        scores = np.array([0.8, 0.75, 0.9, 0.85])
        labels = np.array([1, 1, 0, 0])  # 1表示干净，0表示对抗
        
        old_threshold = self.manager.current_threshold
        self.manager.update_threshold(scores, labels)
        
        # 验证阈值在合理范围内
        self.assertGreaterEqual(self.manager.current_threshold, self.manager.min_threshold)
        self.assertLessEqual(self.manager.current_threshold, self.manager.max_threshold)
    
    def test_detect_drift(self):
        """测试漂移检测"""
        # 添加一些历史分数
        for _ in range(50):
            self.manager.score_history.append(np.random.normal(0.7, 0.1))
        
        # 添加一些漂移分数
        drift_scores = [0.3, 0.35, 0.4, 0.32, 0.38]
        
        has_drift = self.manager.detect_drift(drift_scores)
        
        # 验证漂移检测结果
        self.assertIsInstance(has_drift, bool)
    
    def test_get_statistics(self):
        """测试统计信息获取"""
        # 添加一些更新历史
        self.manager.update_count = 10
        self.manager.drift_count = 2
        
        stats = self.manager.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('current_threshold', stats)
        self.assertIn('update_count', stats)
        self.assertIn('drift_count', stats)
        self.assertEqual(stats['update_count'], 10)
        self.assertEqual(stats['drift_count'], 2)


class TestEnsembleDetector(unittest.TestCase):
    """测试EnsembleDetector类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建多个真实检测器实例
        config1 = DetectorConfig(device="cpu", batch_size=2)
        config2 = DetectorConfig(device="cpu", batch_size=2)
        config3 = DetectorConfig(device="cpu", batch_size=2)
        
        self.detector1 = AdversarialDetector(config1)
        self.detector2 = AdversarialDetector(config2)
        self.detector3 = AdversarialDetector(config3)
        
        self.detectors = [self.detector1, self.detector2, self.detector3]
        self.weights = [0.4, 0.3, 0.3]
        
        self.ensemble = EnsembleDetector(self.detectors, self.weights)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(len(self.ensemble.detectors), 3)
        self.assertEqual(len(self.ensemble.weights), 3)
        self.assertAlmostEqual(sum(self.ensemble.weights), 1.0, places=5)
    
    def test_detect_adversarial_sample(self):
        """测试集成检测"""
        # 创建临时测试图像
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "test_image.jpg")
        
        try:
            # 创建测试图像
            img = Image.new('RGB', (224, 224), color=(100, 150, 200))
            img.save(image_path)
            
            # 执行集成检测
            result = self.ensemble.detect_adversarial(
                img, "test text"
            )
            
            # 验证结果
            self.assertIsInstance(result, dict)
            self.assertIn('is_adversarial', result)
            self.assertIn('confidence', result)
            self.assertIn('ensemble_score', result)
            self.assertIn('individual_results', result)
            
            # 验证individual_results包含所有检测器的结果
            individual_results = result['individual_results']
            self.assertEqual(len(individual_results), 3)
            
            for individual_result in individual_results:
                self.assertIsInstance(individual_result, dict)
                self.assertIn('is_adversarial', individual_result)
                self.assertIn('confidence', individual_result)
                
        finally:
            # 清理临时文件
            if os.path.exists(image_path):
                os.unlink(image_path)
            os.rmdir(temp_dir)
    
    def test_weighted_voting(self):
        """测试加权投票"""
        # 准备测试数据
        scores = [0.2, 0.7, 0.1]  # 对应检测器1、2、3的分数
        confidences = [0.8, 0.7, 0.9]
        
        ensemble_score = self.ensemble._weighted_voting(scores, confidences)
        
        # 验证集成分数
        self.assertIsInstance(ensemble_score, (float, np.floating))
        self.assertGreaterEqual(ensemble_score, 0.0)
        self.assertLessEqual(ensemble_score, 1.0)
        
        # 手动计算期望值进行验证
        expected_score = (0.2 * 0.4 + 0.7 * 0.3 + 0.1 * 0.3)
        self.assertAlmostEqual(ensemble_score, expected_score, places=5)


if __name__ == '__main__':
    unittest.main()