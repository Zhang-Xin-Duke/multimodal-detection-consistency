#!/usr/bin/env python3
"""快速配置测试脚本

只测试配置类，不加载任何模型，避免模型下载和初始化延迟。
"""

import sys
import os
import unittest

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入配置类
from text_augment import TextAugmentConfig
from retrieval import RetrievalConfig
from sd_ref import SDReferenceConfig
from detector import DetectorConfig
from pipeline import PipelineConfig

class TestConfigClasses(unittest.TestCase):
    """测试所有配置类"""
    
    def test_text_augment_config(self):
        """测试TextAugmentConfig"""
        # 默认配置
        config = TextAugmentConfig()
        self.assertEqual(config.num_variants, 5)
        self.assertEqual(config.similarity_threshold, 0.8)
        
        # 自定义配置
        custom_config = TextAugmentConfig(
            num_variants=10,
            similarity_threshold=0.9
        )
        self.assertEqual(custom_config.num_variants, 10)
        self.assertEqual(custom_config.similarity_threshold, 0.9)
    
    def test_retrieval_config(self):
        """测试RetrievalConfig"""
        config = RetrievalConfig()
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.similarity_metric, 'cosine')
    
    def test_sd_reference_config(self):
        """测试SDReferenceConfig"""
        config = SDReferenceConfig()
        self.assertEqual(config.num_images_per_prompt, 3)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.num_inference_steps, 50)
    
    def test_detector_config(self):
        """测试DetectorConfig"""
        config = DetectorConfig()
        self.assertEqual(config.score_aggregation, 'weighted_mean')
        self.assertEqual(config.detection_threshold, 0.5)
        self.assertEqual(config.consistency_threshold, 0.8)
    
    def test_pipeline_config(self):
        """测试PipelineConfig"""
        config = PipelineConfig()
        self.assertTrue(config.enable_text_augment)
        self.assertTrue(config.enable_retrieval)
        self.assertTrue(config.enable_sd_reference)
        self.assertTrue(config.enable_detection)

if __name__ == '__main__':
    print("运行快速配置测试（不加载模型）...")
    unittest.main(verbosity=2)