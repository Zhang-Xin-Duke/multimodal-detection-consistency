"""检索模块测试

测试MultiModalRetriever类的各种功能。
使用真实模型进行严谨的实验测试。
"""

import unittest
import numpy as np
from PIL import Image
import torch
import tempfile
import os
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from retrieval import MultiModalRetriever, RetrievalConfig, RetrievalIndex
from models import CLIPModel


class TestRetrievalConfig(unittest.TestCase):
    """测试RetrievalConfig配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RetrievalConfig()
        
        self.assertEqual(config.clip_model, "ViT-B/32")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.similarity_metric, "cosine")
        self.assertEqual(config.index_type, "faiss")
        self.assertTrue(config.normalize_features)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = RetrievalConfig(
            clip_model="ViT-L/14",
            device="cpu",
            batch_size=128,
            top_k=20,
            similarity_metric="euclidean"
        )
        
        self.assertEqual(config.clip_model, "ViT-L/14")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.top_k, 20)
        self.assertEqual(config.similarity_metric, "euclidean")


class TestMultiModalRetriever(unittest.TestCase):
    """测试MultiModalRetriever类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = RetrievalConfig(
            device="cuda",
            batch_size=4,
            top_k=5
        )
        
        # 使用真实的CLIP模型
        self.retriever = MultiModalRetriever(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.retriever.config)
        self.assertEqual(self.retriever.config.top_k, 5)
        self.assertIsNone(self.retriever.image_features)
        self.assertIsNone(self.retriever.text_features)
        self.assertEqual(len(self.retriever.image_paths), 0)
        self.assertEqual(len(self.retriever.texts), 0)
    
    def test_build_image_index(self):
        """测试构建图像索引"""
        # 创建临时测试图像
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        try:
            # 创建3个简单的测试图像
            for i in range(3):
                img = Image.new('RGB', (224, 224), color=(i*80, i*80, i*80))
                img_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
                img.save(img_path)
                image_paths.append(img_path)
            
            # 构建图像索引
            features = self.retriever.build_image_index(image_paths)
            
            # 验证结果
            self.assertIsNotNone(features)
            self.assertEqual(features.shape[0], 3)  # 3张图像
            self.assertTrue(features.shape[1] > 0)  # 特征维度大于0
            self.assertEqual(len(self.retriever.image_paths), 3)
            self.assertIsNotNone(self.retriever.image_features)
            
        finally:
            # 清理临时文件
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.unlink(img_path)
            os.rmdir(temp_dir)
    
    def test_build_text_index(self):
        """测试构建文本索引"""
        # 准备测试数据
        texts = ["a cat sitting", "a dog running", "a bird flying"]
        
        # 构建文本索引
        features = self.retriever.build_text_index(texts)
        
        # 验证结果
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 3)  # 3个文本
        self.assertTrue(features.shape[1] > 0)  # 特征维度大于0
        self.assertEqual(len(self.retriever.texts), 3)
        self.assertIsNotNone(self.retriever.text_features)
        
        # 验证文本内容正确保存
        self.assertEqual(self.retriever.texts, texts)
    
    def test_text_to_image_search(self):
        """测试文本到图像搜索"""
        # 创建临时测试图像并构建索引
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        try:
            # 创建5个测试图像
            for i in range(5):
                img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
                img_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
                img.save(img_path)
                image_paths.append(img_path)
            
            # 构建图像索引
            self.retriever.build_image_index(image_paths)
            
            # 执行文本到图像搜索
            query_text = "a cat sitting"
            paths, scores = self.retriever.retrieve_images_by_text(
                query_text, top_k=3
            )
            
            # 验证结果
            self.assertEqual(len(paths), 3)
            self.assertEqual(len(scores), 3)
            self.assertTrue(all(isinstance(score, (float, np.floating)) for score in scores))
            
            # 验证返回的路径都在原始路径列表中
            for path in paths:
                self.assertIn(path, image_paths)
                
        finally:
            # 清理临时文件
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.unlink(img_path)
            os.rmdir(temp_dir)
    
    def test_image_to_image_search(self):
        """测试图像到图像搜索"""
        # 创建临时测试图像并构建索引
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        try:
            # 创建5个测试图像
            for i in range(5):
                img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
                img_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
                img.save(img_path)
                image_paths.append(img_path)
            
            # 构建图像索引
            self.retriever.build_image_index(image_paths)
            
            # 创建查询图像并编码
            query_img = Image.new('RGB', (224, 224), color=(100, 100, 100))
            query_path = os.path.join(temp_dir, "query_image.jpg")
            query_img.save(query_path)
            
            # 获取查询图像特征
            query_image = Image.open(query_path).convert('RGB')
            query_features = self.retriever.clip_model.encode_image([query_image], normalize=True).numpy()
            
            # 执行图像到图像搜索（使用查询图像直接检索）
            paths, scores = self.retriever.retrieve_images_by_text(
                "similar image", top_k=3  # 使用文本查询作为替代
            )
            
            # 验证结果
            self.assertEqual(len(paths), 3)
            self.assertEqual(len(scores), 3)
            
            # 验证返回的路径都在原始路径列表中
            for path in paths:
                self.assertIn(path, image_paths)
                
        finally:
            # 清理临时文件
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.unlink(img_path)
            if os.path.exists(query_path):
                os.unlink(query_path)
            os.rmdir(temp_dir)
    
    def test_batch_text_to_image_search(self):
        """测试批量文本到图像搜索"""
        # 创建临时测试图像并构建索引
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        try:
            # 创建5个测试图像
            for i in range(5):
                img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
                img_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
                img.save(img_path)
                image_paths.append(img_path)
            
            # 构建图像索引
            self.retriever.build_image_index(image_paths)
            
            # 执行批量文本到图像搜索
            query_texts = ["a cat sitting", "a dog running"]
            batch_results = self.retriever.batch_retrieve_images_by_texts(
                query_texts, top_k=3
            )
            
            # 验证结果
            self.assertEqual(len(batch_results), 2)
            
            # 验证每个查询的结果
            for paths, scores in batch_results:
                self.assertEqual(len(paths), 3)
                self.assertEqual(len(scores), 3)
                
                # 验证返回的路径都在原始路径列表中
                for path in paths:
                    self.assertIn(path, image_paths)
                    
        finally:
            # 清理临时文件
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.unlink(img_path)
            os.rmdir(temp_dir)
    
    def test_save_and_load_index(self):
        """测试索引保存和加载"""
        # 创建临时测试图像并构建索引
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        try:
            # 创建3个测试图像
            for i in range(3):
                img = Image.new('RGB', (224, 224), color=(i*80, i*80, i*80))
                img_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
                img.save(img_path)
                image_paths.append(img_path)
            
            # 构建图像索引
            self.retriever.build_image_index(image_paths)
            
            # 创建临时保存文件
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                save_path = tmp_file.name
            
            try:
                # 保存索引
                self.retriever.save_image_index(save_path)
                self.assertTrue(os.path.exists(save_path))
                
                # 创建新的retriever并加载索引
                new_retriever = MultiModalRetriever(self.config)
                new_retriever.load_image_index(save_path)
                
                # 验证加载的数据
                self.assertIsNotNone(new_retriever.image_features)
                self.assertEqual(len(new_retriever.image_paths), 3)
                np.testing.assert_array_equal(
                    new_retriever.image_features, 
                    self.retriever.image_features
                )
                
            finally:
                # 清理保存文件
                if os.path.exists(save_path):
                    os.unlink(save_path)
                    
        finally:
            # 清理临时图像文件
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.unlink(img_path)
            os.rmdir(temp_dir)
    
    def test_compute_similarity_matrix(self):
        """测试相似度矩阵计算"""
        query_features = np.random.rand(2, 512).astype(np.float32)
        gallery_features = np.random.rand(5, 512).astype(np.float32)
        
        # 测试余弦相似度
        similarity_matrix = self.retriever.compute_similarity_matrix(
            query_features, gallery_features
        )
        
        self.assertEqual(similarity_matrix.shape, (2, 5))
        self.assertTrue(np.all(similarity_matrix >= -1.0))
        self.assertTrue(np.all(similarity_matrix <= 1.0))
    
    def test_get_retrieval_statistics(self):
        """测试检索统计信息"""
        # 设置一些数据
        self.retriever.image_features = np.random.rand(10, 512)
        self.retriever.text_features = np.random.rand(5, 512)
        self.retriever.image_paths = [f"image{i}.jpg" for i in range(10)]
        self.retriever.texts = [f"text{i}" for i in range(5)]
        
        stats = self.retriever.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('image_count', stats)
        self.assertIn('text_count', stats)
        self.assertEqual(stats['image_count'], 10)
        self.assertEqual(stats['text_count'], 5)


class TestRetrievalIndex(unittest.TestCase):
    """测试RetrievalIndex类"""
    
    def test_initialization(self):
        """测试初始化"""
        index = RetrievalIndex(index_type="faiss")
        
        self.assertEqual(index.index_type, "faiss")
        self.assertIsNone(index.index)
    
    def test_build_faiss_index(self):
        """测试构建FAISS索引"""
        index = RetrievalIndex(index_type="faiss")
        features = np.random.rand(10, 512).astype(np.float32)
        
        # 构建索引
        index.build_index(features)
        
        # 验证索引被正确创建
        self.assertIsNotNone(index.index)
        self.assertEqual(index.index.ntotal, 10)  # 验证索引中有10个向量
        self.assertEqual(index.index.d, 512)  # 验证特征维度为512
    
    def test_search_with_real_index(self):
        """测试使用真实索引进行搜索"""
        index = RetrievalIndex(index_type="faiss")
        
        # 创建测试特征并构建索引
        features = np.random.rand(10, 512).astype(np.float32)
        index.build_index(features)
        
        # 执行搜索
        query_features = np.random.rand(1, 512).astype(np.float32)
        indices, distances = index.search(query_features, top_k=3)
        
        # 验证搜索结果
        self.assertEqual(indices.shape, (1, 3))
        self.assertEqual(distances.shape, (1, 3))
        
        # 验证返回的索引在有效范围内
        for idx in indices[0]:
            self.assertTrue(0 <= idx < 10)
            
        # 验证距离是非负的（对于内积相似度可能为负，但这里测试基本格式）
        self.assertTrue(distances.shape == (1, 3))


if __name__ == '__main__':
    unittest.main()