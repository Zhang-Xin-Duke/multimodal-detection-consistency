"""文本增强模块测试

测试TextAugmenter类的各种功能。
使用真实模型进行严谨的实验测试。
"""

import unittest
import numpy as np
from PIL import Image
import torch
import os
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text_augment import TextAugmenter, TextAugmentConfig
from models import QwenModel, CLIPModel


class TestTextAugmentConfig(unittest.TestCase):
    """测试TextAugmentConfig配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TextAugmentConfig()
        
        self.assertEqual(config.num_variants, 5)
        self.assertEqual(config.similarity_threshold, 0.8)
        self.assertEqual(config.max_attempts, 10)
        self.assertEqual(config.synonym_prob, 0.3)
        self.assertIsNotNone(config.back_translation_languages)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = TextAugmentConfig(
            num_variants=10,
            similarity_threshold=0.9,
            synonym_prob=0.5
        )
        
        self.assertEqual(config.num_variants, 10)
        self.assertEqual(config.similarity_threshold, 0.9)
        self.assertEqual(config.synonym_prob, 0.5)


class TestTextAugmenter(unittest.TestCase):
    """测试TextAugmenter类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = TextAugmentConfig(
            num_variants=3,
            similarity_threshold=0.8
        )
        
        # 使用真实模型进行测试
        self.augmenter = TextAugmenter(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.augmenter.config)
        self.assertEqual(self.augmenter.config.num_variants, 3)
        self.assertIsInstance(self.augmenter.variant_cache, dict)
        self.assertIsInstance(self.augmenter.similarity_cache, dict)
    
    def test_generate_synonym_variants(self):
        """测试同义词变体生成"""
        test_text = "a cat sitting"
        variants = self.augmenter._generate_synonym_variants(test_text)
        
        self.assertIsInstance(variants, list)
        self.assertTrue(len(variants) <= self.config.num_variants)
        
        # 验证变体不为空且与原文本不同
        for variant in variants:
            self.assertIsInstance(variant, str)
            self.assertNotEqual(variant.strip(), test_text)
    
    def test_generate_paraphrase_variants(self):
        """测试释义变体生成"""
        test_text = "a cat sitting"
        variants = self.augmenter._generate_paraphrase_variants(test_text)
        
        self.assertIsInstance(variants, list)
        self.assertTrue(len(variants) > 0)
        
        # 验证变体质量
        for variant in variants:
            self.assertIsInstance(variant, str)
            self.assertTrue(len(variant.strip()) > 0)
            self.assertNotEqual(variant.strip(), test_text)
    
    def test_parse_generated_variants(self):
        """测试生成变体解析"""
        generated_text = "1. A feline resting\n2. A cat positioned\n3. A kitty sitting"
        original_text = "a cat sitting"
        
        variants = self.augmenter._parse_generated_variants(generated_text, original_text)
        
        self.assertIsInstance(variants, list)
        self.assertEqual(len(variants), 3)
        self.assertIn("A feline resting", variants)
        self.assertIn("A cat positioned", variants)
        self.assertIn("A kitty sitting", variants)
    
    def test_filter_variants(self):
        """测试变体过滤"""
        original_text = "a cat sitting"
        variants = [
            "a cat sitting",  # 重复，应被过滤
            "a feline resting",
            "a kitty perching",
            "a dog running",  # 相似度低，可能被过滤
            "a cat positioned"
        ]
        
        filtered_variants = self.augmenter._filter_variants(original_text, variants)
        
        self.assertIsInstance(filtered_variants, list)
        self.assertNotIn("a cat sitting", filtered_variants)  # 重复被过滤
        
        # 验证过滤后的变体都满足相似度阈值
        for variant in filtered_variants:
            similarity = self.augmenter.compute_text_similarity(original_text, variant)
            self.assertGreaterEqual(similarity, self.config.similarity_threshold)
    
    def test_compute_text_similarity(self):
        """测试文本相似度计算"""
        text1 = "a cat sitting"
        text2 = "a feline resting"
        
        similarity = self.augmenter.compute_text_similarity(text1, text2)
        
        self.assertIsInstance(similarity, float)
        self.assertTrue(0.0 <= similarity <= 1.0)
        
        # 测试相同文本的相似度应该接近1
        same_similarity = self.augmenter.compute_text_similarity(text1, text1)
        self.assertGreater(same_similarity, 0.95)
        
        # 测试完全不同文本的相似度应该较低
        diff_similarity = self.augmenter.compute_text_similarity(text1, "completely different content")
        self.assertLess(diff_similarity, 0.5)
    
    def test_generate_variants_with_cache(self):
        """测试带缓存的变体生成"""
        text = "a cat sitting"
        methods = ['synonym', 'paraphrase']
        
        # 清空缓存
        self.augmenter.variant_cache.clear()
        
        # 第一次调用
        variants1 = self.augmenter.generate_variants(text, methods)
        
        # 验证缓存中有数据
        cache_key = f"{text}_{','.join(sorted(methods))}"
        self.assertIn(cache_key, self.augmenter.variant_cache)
        
        # 第二次调用（应该使用缓存）
        variants2 = self.augmenter.generate_variants(text, methods)
        
        # 验证两次调用结果相同
        self.assertEqual(variants1, variants2)
        
        # 验证返回的是变体列表
        self.assertIsInstance(variants1, list)
        for variant in variants1:
            self.assertIsInstance(variant, str)
    
    def test_batch_generate_variants(self):
        """测试批量变体生成"""
        texts = ["a cat sitting", "a dog running", "a bird flying"]
        
        batch_variants = self.augmenter.batch_generate_variants(texts)
        
        self.assertIsInstance(batch_variants, list)
        self.assertEqual(len(batch_variants), len(texts))
        
        # 验证每个文本都生成了变体
        for i, variants in enumerate(batch_variants):
            self.assertIsInstance(variants, list)
            # 验证变体不为空且与原文本不同
            for variant in variants:
                self.assertIsInstance(variant, str)
                self.assertNotEqual(variant.strip(), texts[i])


class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""
    
    def test_create_augmentation_prompts(self):
        """测试增强提示词创建"""
        from text_augment import create_augmentation_prompts
        
        query = "a cat sitting on a table"
        prompts = create_augmentation_prompts(query)
        
        self.assertIsInstance(prompts, list)
        self.assertTrue(len(prompts) > 0)
        
        for prompt in prompts:
            self.assertIsInstance(prompt, str)
            self.assertIn(query, prompt)


if __name__ == '__main__':
    unittest.main()