#!/usr/bin/env python3
"""
简单的检测器测试脚本
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock
import numpy as np
from PIL import Image

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.detector import DetectorConfig, AdversarialDetector

class TestDetectorSimple(unittest.TestCase):
    """简单的检测器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建配置
        self.config = DetectorConfig(
            clip_model="ViT-B/32",
            device="cpu",  # 使用CPU避免GPU问题
            detection_methods=["consistency"],  # 只使用一致性检测
            use_text_variants=False,  # 禁用文本变体
            use_sd_reference=False,  # 禁用SD参考
            enable_cache=False  # 禁用缓存
        )
        
        # Mock CLIP模型
        self.mock_clip = Mock()
        self.mock_clip.get_text_image_similarity.return_value = Mock()
        self.mock_clip.get_text_image_similarity.return_value.item.return_value = 0.8
        
        # Mock相似性计算器
        self.mock_similarity_metrics = Mock()
        self.mock_similarity_metrics.compute_consistency_score.return_value = 0.7
        
        # 创建检测器实例（使用Mock）
        self.detector = AdversarialDetector(self.config)
        self.detector.clip_model = self.mock_clip
        self.detector.similarity_metrics = self.mock_similarity_metrics
        self.detector.text_augmenter = None
        self.detector.sd_generator = None
    
    def test_config_creation(self):
        """测试配置创建"""
        self.assertEqual(self.config.clip_model, "ViT-B/32")
        self.assertEqual(self.config.device, "cpu")
        self.assertFalse(self.config.use_text_variants)
        self.assertFalse(self.config.use_sd_reference)
    
    def test_detect_adversarial_basic(self):
        """测试基本对抗检测功能"""
        # 创建测试图像
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        text = "a cat sitting on a chair"
        
        # 执行检测
        result = self.detector.detect_adversarial(image, text)
        
        # 验证结果结构
        self.assertIsInstance(result, dict)
        self.assertIn('is_adversarial', result)
        self.assertIn('aggregated_score', result)
        self.assertIn('detection_scores', result)
        self.assertIn('detection_details', result)
        
        # 验证结果类型
        self.assertIsInstance(result['is_adversarial'], bool)
        self.assertIsInstance(result['aggregated_score'], (int, float))
        self.assertIsInstance(result['detection_scores'], dict)
        self.assertIsInstance(result['detection_details'], dict)
        
        # 验证分数范围
        self.assertGreaterEqual(result['aggregated_score'], 0.0)
        self.assertLessEqual(result['aggregated_score'], 1.0)
    
    def test_batch_detect(self):
        """测试批量检测"""
        # 创建测试数据
        images = [Image.new('RGB', (224, 224), color=(i*50, i*50, i*50)) for i in range(3)]
        texts = ["a cat", "a dog", "a bird"]
        
        # 执行批量检测
        results = self.detector.batch_detect(images, texts)
        
        # 验证结果
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('is_adversarial', result)
            self.assertIn('aggregated_score', result)

if __name__ == '__main__':
    print("运行简单检测器测试...")
    unittest.main(verbosity=2)