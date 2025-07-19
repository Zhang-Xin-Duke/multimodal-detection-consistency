#!/usr/bin/env python3
"""
GPU使用测试脚本
"""

import torch
import sys
import os

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from retrieval import RetrievalConfig, MultiModalRetriever
from detector import DetectorConfig, AdversarialDetector
from models import CLIPConfig, CLIPModel

def test_gpu_usage():
    """测试GPU使用情况"""
    print("=== GPU 状态检查 ===")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n=== 测试 CLIP 模型 GPU 使用 ===")
    try:
        # 创建CLIP配置，使用CUDA
        clip_config = CLIPConfig(device="cuda")
        print(f"CLIP配置设备: {clip_config.device}")
        
        # 初始化CLIP模型
        clip_model = CLIPModel(clip_config)
        print(f"CLIP模型设备: {clip_model.device}")
        
        # 测试文本编码
        test_texts = ["a cat sitting on a table", "a dog running in the park"]
        print(f"编码文本: {test_texts}")
        
        # 检查GPU内存使用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            print(f"初始GPU内存使用: {initial_memory / 1024**2:.2f} MB")
        
        # 编码文本
        text_features = clip_model.encode_text(test_texts)
        print(f"文本特征形状: {text_features.shape}")
        print(f"文本特征设备: {text_features.device if hasattr(text_features, 'device') else 'numpy array'}")
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            print(f"编码后GPU内存使用: {final_memory / 1024**2:.2f} MB")
            print(f"内存增量: {(final_memory - initial_memory) / 1024**2:.2f} MB")
        
        print("✅ CLIP模型GPU测试成功")
        
    except Exception as e:
        print(f"❌ CLIP模型GPU测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试检索配置 GPU 使用 ===")
    try:
        # 创建检索配置，使用CUDA
        retrieval_config = RetrievalConfig(device="cuda")
        print(f"检索配置设备: {retrieval_config.device}")
        print("✅ 检索配置GPU设置成功")
        
    except Exception as e:
        print(f"❌ 检索配置GPU测试失败: {e}")
    
    print("\n=== 测试检测器配置 GPU 使用 ===")
    try:
        # 创建检测器配置，使用CUDA
        detector_config = DetectorConfig(device="cuda")
        print(f"检测器配置设备: {detector_config.device}")
        print("✅ 检测器配置GPU设置成功")
        
    except Exception as e:
        print(f"❌ 检测器配置GPU测试失败: {e}")

if __name__ == "__main__":
    test_gpu_usage()