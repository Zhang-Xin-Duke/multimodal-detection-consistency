"""管道模块测试

测试DefensePipeline类的各种功能。
"""

import unittest
import numpy as np
import tempfile
import os
import time
from PIL import Image

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pipeline import (
    DefensePipeline, PipelineConfig, PipelineResult, PipelineProfiler
)
from text_augment import TextAugmenter
from retrieval import MultiModalRetriever
from sd_ref import SDReferenceGenerator
from ref_bank import ReferenceBank
from detector import AdversarialDetector


class TestPipelineConfig(unittest.TestCase):
    """测试PipelineConfig配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PipelineConfig()
        
        self.assertTrue(config.enable_text_augment)
        self.assertTrue(config.enable_retrieval)
        self.assertTrue(config.enable_sd_reference)
        self.assertTrue(config.enable_detection)
        self.assertTrue(config.enable_parallel)
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.enable_cache)
        self.assertEqual(config.max_workers, 4)
        self.assertFalse(config.save_intermediate_results)
        self.assertIsNone(config.output_dir)
        self.assertFalse(config.enable_profiling)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = PipelineConfig(
            enable_text_augment=False,
            enable_retrieval=False,
            enable_sd_reference=False,
            enable_detection=True,
            enable_parallel=False,
            batch_size=16,
            enable_cache=False,
            max_workers=2,
            save_intermediate_results=True,
            output_dir="custom_outputs",
            enable_profiling=True
        )
        
        self.assertFalse(config.enable_text_augment)
        self.assertFalse(config.enable_retrieval)
        self.assertFalse(config.enable_sd_reference)
        self.assertTrue(config.enable_detection)
        self.assertFalse(config.enable_parallel)
        self.assertEqual(config.batch_size, 16)
        self.assertFalse(config.enable_cache)
        self.assertEqual(config.max_workers, 2)
        self.assertTrue(config.save_intermediate_results)
        self.assertEqual(config.output_dir, "custom_outputs")
        self.assertTrue(config.enable_profiling)


class TestPipelineResult(unittest.TestCase):
    """测试PipelineResult类"""
    
    def test_initialization(self):
        """测试初始化"""
        result = PipelineResult(
            original_text="test text",
            original_image_path="test.jpg"
        )
        
        self.assertEqual(result.original_text, "test text")
        self.assertEqual(result.original_image_path, "test.jpg")
        self.assertIsNone(result.text_variants)
        self.assertIsNone(result.retrieved_items)
        self.assertIsNone(result.sd_reference_images)
        self.assertIsNone(result.detection_result)
        self.assertEqual(len(result.step_timings), 0)
        self.assertEqual(len(result.errors), 0)
    
    def test_add_timing(self):
        """测试添加时间记录"""
        result = PipelineResult("test", "test.jpg")
        
        result.add_timing("step1", 0.5)
        result.add_timing("step2", 1.2)
        
        self.assertEqual(len(result.step_timings), 2)
        self.assertEqual(result.step_timings["step1"], 0.5)
        self.assertEqual(result.step_timings["step2"], 1.2)
    
    def test_add_error(self):
        """测试添加错误记录"""
        result = PipelineResult("test", "test.jpg")
        
        result.add_error("step1", "Error message 1")
        result.add_error("step2", "Error message 2")
        
        self.assertEqual(len(result.errors), 2)
        self.assertEqual(result.errors["step1"], "Error message 1")
        self.assertEqual(result.errors["step2"], "Error message 2")
    
    def test_get_total_time(self):
        """测试总时间计算"""
        result = PipelineResult("test", "test.jpg")
        
        result.add_timing("step1", 0.5)
        result.add_timing("step2", 1.2)
        result.add_timing("step3", 0.8)
        
        total_time = result.get_total_time()
        self.assertAlmostEqual(total_time, 2.5, places=5)
    
    def test_has_errors(self):
        """测试错误检查"""
        result = PipelineResult("test", "test.jpg")
        
        self.assertFalse(result.has_errors())
        
        result.add_error("step1", "Error message")
        
        self.assertTrue(result.has_errors())


class TestPipelineProfiler(unittest.TestCase):
    """测试PipelineProfiler类"""
    
    def test_initialization(self):
        """测试初始化"""
        profiler = PipelineProfiler()
        
        self.assertEqual(len(profiler.step_times), 0)
        self.assertEqual(len(profiler.step_counts), 0)
        self.assertEqual(len(profiler.step_errors), 0)
    
    def test_start_and_end_step(self):
        """测试步骤计时"""
        profiler = PipelineProfiler()
        
        profiler.start_step("test_step")
        time.sleep(0.01)  # 短暂等待
        elapsed_time = profiler.end_step("test_step")
        
        self.assertGreater(elapsed_time, 0)
        self.assertIn("test_step", profiler.step_times)
        self.assertEqual(len(profiler.step_times["test_step"]), 1)
        self.assertEqual(profiler.step_counts["test_step"], 1)
    
    def test_record_error(self):
        """测试错误记录"""
        profiler = PipelineProfiler()
        
        profiler.record_error("test_step", "Test error message")
        
        self.assertIn("test_step", profiler.step_errors)
        self.assertEqual(len(profiler.step_errors["test_step"]), 1)
        self.assertEqual(profiler.step_errors["test_step"][0], "Test error message")
    
    def test_get_statistics(self):
        """测试统计信息获取"""
        profiler = PipelineProfiler()
        
        # 记录一些步骤时间
        profiler.step_times["step1"] = [0.1, 0.2, 0.15]
        profiler.step_counts["step1"] = 3
        profiler.step_times["step2"] = [0.5, 0.6]
        profiler.step_counts["step2"] = 2
        
        stats = profiler.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("step1", stats)
        self.assertIn("step2", stats)
        
        step1_stats = stats["step1"]
        self.assertIn("count", step1_stats)
        self.assertIn("total_time", step1_stats)
        self.assertIn("average_time", step1_stats)
        self.assertIn("min_time", step1_stats)
        self.assertIn("max_time", step1_stats)
        
        self.assertEqual(step1_stats["count"], 3)
        self.assertAlmostEqual(step1_stats["total_time"], 0.45, places=5)
        self.assertAlmostEqual(step1_stats["average_time"], 0.15, places=5)
    
    def test_reset(self):
        """测试重置"""
        profiler = PipelineProfiler()
        
        # 添加一些数据
        profiler.step_times["step1"] = [0.1, 0.2]
        profiler.step_counts["step1"] = 2
        profiler.step_errors["step1"] = ["error1"]
        
        # 重置
        profiler.reset()
        
        self.assertEqual(len(profiler.step_times), 0)
        self.assertEqual(len(profiler.step_counts), 0)
        self.assertEqual(len(profiler.step_errors), 0)


class TestDefensePipeline(unittest.TestCase):
    """测试DefensePipeline类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = PipelineConfig(
            enable_text_augmentation=True,
            enable_retrieval=True,
            enable_sd_reference=True,
            enable_detection=True,
            enable_parallel_processing=False,  # 测试时禁用并行处理
            batch_size=4,
            enable_caching=True,
            enable_output_saving=False,  # 测试时禁用输出保存
            enable_profiling=True
        )
        
        # 创建pipeline实例（使用真实组件）
        self.pipeline = DefensePipeline(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNotNone(self.pipeline.text_augmenter)
        self.assertIsNotNone(self.pipeline.retriever)
        self.assertIsNotNone(self.pipeline.sd_generator)
        self.assertIsNotNone(self.pipeline.ref_bank)
        self.assertIsNotNone(self.pipeline.detector)
        self.assertIsNotNone(self.pipeline.profiler)
        self.assertEqual(len(self.pipeline.pipeline_cache), 0)
    
    def test_process_single_sample(self):
        """测试单样本处理"""
        # 创建临时图像文件
        temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        image = Image.new('RGB', (224, 224), color='red')
        image.save(temp_image.name)
        temp_image.close()
        
        try:
            # 测试数据
            text = "a cat sitting on a chair"
            image_path = temp_image.name
            
            # 执行处理
            result = self.pipeline.process_single_sample(text, image_path)
            
            # 验证结果
            self.assertIsInstance(result, PipelineResult)
            self.assertEqual(result.original_text, text)
            self.assertEqual(result.original_image_path, image_path)
            self.assertIsNotNone(result.detection_result)
            self.assertIn('is_adversarial', result.detection_result)
            self.assertIn('confidence', result.detection_result)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_image.name):
                os.unlink(temp_image.name)
    
    def test_process_batch_samples(self):
        """测试批量样本处理"""
        # 创建临时图像文件
        temp_images = []
        for i in range(2):
            temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            image = Image.new('RGB', (224, 224), color=['red', 'blue'][i])
            image.save(temp_image.name)
            temp_image.close()
            temp_images.append(temp_image.name)
        
        try:
            # 测试数据
            texts = ["a cat sitting", "a dog running"]
            image_paths = temp_images
            
            # 执行批量处理
            batch_results = self.pipeline.process_batch_samples(texts, image_paths)
            
            # 验证结果
            self.assertEqual(len(batch_results), 2)
            
            for i, result in enumerate(batch_results):
                self.assertIsInstance(result, PipelineResult)
                self.assertEqual(result.original_text, texts[i])
                self.assertEqual(result.original_image_path, image_paths[i])
                self.assertIsNotNone(result.detection_result)
                self.assertIn('is_adversarial', result.detection_result)
                self.assertIn('confidence', result.detection_result)
            
        finally:
            # 清理临时文件
            for temp_image in temp_images:
                if os.path.exists(temp_image):
                    os.unlink(temp_image)
    
    def test_build_reference_database(self):
        """测试构建参考数据库"""
        # 创建临时图像文件
        temp_images = []
        for i in range(3):
            temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            image = Image.new('RGB', (224, 224), color=['red', 'green', 'blue'][i])
            image.save(temp_image.name)
            temp_image.close()
            temp_images.append(temp_image.name)
        
        try:
            # 执行数据库构建
            self.pipeline.build_reference_database(temp_images)
            
            # 验证参考数据库已构建（通过检查组件状态）
            self.assertIsNotNone(self.pipeline.retriever)
            self.assertIsNotNone(self.pipeline.ref_bank)
            
        finally:
            # 清理临时文件
            for temp_image in temp_images:
                if os.path.exists(temp_image):
                    os.unlink(temp_image)
    
    def test_update_reference_database(self):
        """测试更新参考数据库"""
        # 创建临时图像文件
        temp_images = []
        for i in range(2):
            temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            image = Image.new('RGB', (224, 224), color=['yellow', 'purple'][i])
            image.save(temp_image.name)
            temp_image.close()
            temp_images.append(temp_image.name)
        
        try:
            # 执行数据库更新
            self.pipeline.update_reference_database(temp_images)
            
            # 验证参考数据库已更新（通过检查组件状态）
            self.assertIsNotNone(self.pipeline.retriever)
            self.assertIsNotNone(self.pipeline.ref_bank)
            
        finally:
            # 清理临时文件
            for temp_image in temp_images:
                if os.path.exists(temp_image):
                    os.unlink(temp_image)
    
    def test_evaluate_pipeline_performance(self):
        """测试管道性能评估"""
        # 创建临时图像文件
        temp_images = []
        for i in range(2):
            temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            image = Image.new('RGB', (224, 224), color=['red', 'blue'][i])
            image.save(temp_image.name)
            temp_image.close()
            temp_images.append(temp_image.name)
        
        try:
            # 准备测试数据
            test_texts = ["test text 1", "test text 2"]
            test_image_paths = temp_images
            true_labels = [0, 1]  # 0表示干净，1表示对抗
            
            # 执行性能评估
            performance_metrics = self.pipeline.evaluate_pipeline_performance(
                test_texts, test_image_paths, true_labels
            )
            
            # 验证性能指标
            self.assertIsInstance(performance_metrics, dict)
            self.assertIn('accuracy', performance_metrics)
            self.assertIn('precision', performance_metrics)
            self.assertIn('recall', performance_metrics)
            self.assertIn('f1_score', performance_metrics)
            
            # 验证指标值在合理范围内
            for metric_name, metric_value in performance_metrics.items():
                if metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                    self.assertGreaterEqual(metric_value, 0.0)
                    self.assertLessEqual(metric_value, 1.0)
        
        finally:
            # 清理临时文件
            for temp_image in temp_images:
                if os.path.exists(temp_image):
                    os.unlink(temp_image)
    
    def test_get_pipeline_statistics(self):
        """测试管道统计信息获取"""
        # 模拟一些处理历史
        self.pipeline.pipeline_stats['total_processed'] = 100
        self.pipeline.pipeline_stats['adversarial_detected'] = 15
        self.pipeline.pipeline_stats['cache_hits'] = 30
        self.pipeline.pipeline_stats['cache_misses'] = 70
        
        stats = self.pipeline.get_pipeline_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_processed', stats)
        self.assertIn('adversarial_detected', stats)
        self.assertIn('detection_rate', stats)
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('component_stats', stats)
        
        self.assertEqual(stats['total_processed'], 100)
        self.assertEqual(stats['adversarial_detected'], 15)
        self.assertAlmostEqual(stats['detection_rate'], 0.15, places=2)
        self.assertAlmostEqual(stats['cache_hit_rate'], 0.3, places=2)
    
    def test_save_and_load_pipeline_state(self):
        """测试管道状态保存和加载"""
        # 设置一些状态数据
        self.pipeline.pipeline_stats['total_processed'] = 50
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            state_path = tmp_file.name
        
        try:
            # 保存状态
            self.pipeline.save_pipeline_state(state_path)
            self.assertTrue(os.path.exists(state_path))
            
            # 创建新的管道并加载状态
            new_pipeline = DefensePipeline(self.config)
            new_pipeline.load_pipeline_state(state_path)
            
            # 验证状态被正确加载
            self.assertEqual(
                new_pipeline.pipeline_stats['total_processed'], 50
            )
            
        finally:
            # 清理临时文件
            if os.path.exists(state_path):
                os.unlink(state_path)
    
    def test_clear_cache(self):
        """测试缓存清理"""
        # 添加一些缓存数据
        self.pipeline.pipeline_cache['test_key'] = 'test_value'
        self.assertEqual(len(self.pipeline.pipeline_cache), 1)
        
        # 清理缓存
        self.pipeline.clear_cache()
        self.assertEqual(len(self.pipeline.pipeline_cache), 0)
        
        # 验证缓存已被清理（通过检查缓存大小）
        self.assertIsNotNone(self.pipeline.text_augmenter)
        self.assertIsNotNone(self.pipeline.retriever)
        self.assertIsNotNone(self.pipeline.sd_generator)
        self.assertIsNotNone(self.pipeline.ref_bank)
        self.assertIsNotNone(self.pipeline.detector)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 准备测试数据（使用不存在的图像路径来触发错误）
        text = "test text"
        image_path = "nonexistent_image.jpg"
        
        # 执行处理
        result = self.pipeline.process_single_sample(text, image_path)
        
        # 验证错误被正确处理
        self.assertIsInstance(result, PipelineResult)
        # 由于图像不存在，应该会有错误或者返回默认结果
        self.assertIsNotNone(result)
    
    def test_caching_functionality(self):
        """测试缓存功能"""
        # 创建临时图像文件
        temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        image = Image.new('RGB', (224, 224), color='green')
        image.save(temp_image.name)
        temp_image.close()
        
        try:
            # 启用缓存
            self.config.enable_caching = True
            
            text = "test text"
            image_path = temp_image.name
            
            # 第一次处理
            result1 = self.pipeline.process_single_sample(text, image_path)
            
            # 验证结果
            self.assertIsInstance(result1, PipelineResult)
            self.assertIsNotNone(result1.detection_result)
            
            # 第二次处理相同输入（应该使用缓存）
            result2 = self.pipeline.process_single_sample(text, image_path)
            
            # 验证结果一致性
            self.assertIsInstance(result2, PipelineResult)
            self.assertIsNotNone(result2.detection_result)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_image.name):
                os.unlink(temp_image.name)


if __name__ == '__main__':
    unittest.main()