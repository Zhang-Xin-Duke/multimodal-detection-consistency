#!/usr/bin/env python3
"""
GPU密集测试脚本 - 充分利用GPU
"""

import torch
import sys
import os
import time
import numpy as np

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import CLIPConfig, CLIPModel
from retrieval import RetrievalConfig, RetrievalIndex

def test_gpu_intensive():
    """GPU密集测试"""
    print("=== GPU 密集测试开始 ===")
    
    # 检查GPU状态
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行GPU测试")
        return
    
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 清空GPU缓存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    print(f"初始GPU内存: {initial_memory / 1024**2:.2f} MB")
    
    try:
        # 创建CLIP模型配置，使用更大的批处理
        clip_config = CLIPConfig(
            device="cuda",
            batch_size=512,  # 增大批处理大小
            model_name="ViT-B/32"
        )
        
        print(f"\n=== 初始化CLIP模型 (批处理大小: {clip_config.batch_size}) ===")
        clip_model = CLIPModel(clip_config)
        
        after_model_memory = torch.cuda.memory_allocated()
        print(f"模型加载后GPU内存: {after_model_memory / 1024**2:.2f} MB")
        print(f"模型占用内存: {(after_model_memory - initial_memory) / 1024**2:.2f} MB")
        
        # 生成大量测试文本
        print("\n=== 生成大量测试数据 ===")
        test_texts = []
        base_texts = [
            "a cat sitting on a table",
            "a dog running in the park", 
            "a bird flying in the sky",
            "a car driving on the road",
            "a person walking on the street",
            "a tree growing in the forest",
            "a flower blooming in the garden",
            "a fish swimming in the ocean",
            "a mountain standing tall",
            "a river flowing through the valley"
        ]
        
        # 扩展到更多文本
        for i in range(100):  # 生成1000个文本
            for base_text in base_texts:
                test_texts.append(f"{base_text} variant {i}")
        
        print(f"生成了 {len(test_texts)} 个测试文本")
        
        # 批量编码文本
        print("\n=== 开始批量文本编码 ===")
        start_time = time.time()
        
        # 分批处理以避免内存溢出
        batch_size = clip_config.batch_size
        all_features = []
        
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(test_texts) + batch_size - 1)//batch_size}, 大小: {len(batch_texts)}")
            
            # 监控GPU内存
            before_batch_memory = torch.cuda.memory_allocated()
            
            # 编码文本
            batch_features = clip_model.encode_text(batch_texts)
            all_features.append(batch_features)
            
            after_batch_memory = torch.cuda.memory_allocated()
            print(f"  批次处理前GPU内存: {before_batch_memory / 1024**2:.2f} MB")
            print(f"  批次处理后GPU内存: {after_batch_memory / 1024**2:.2f} MB")
            print(f"  批次内存增量: {(after_batch_memory - before_batch_memory) / 1024**2:.2f} MB")
            
            # GPU利用率检查已移除（需要pynvml）
        
        # 合并所有特征
        if all_features:
            if isinstance(all_features[0], torch.Tensor):
                final_features = torch.cat(all_features, dim=0)
            else:
                final_features = np.vstack(all_features)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n=== 编码完成 ===")
        print(f"总处理时间: {processing_time:.2f} 秒")
        print(f"平均每个文本: {processing_time/len(test_texts)*1000:.2f} 毫秒")
        print(f"吞吐量: {len(test_texts)/processing_time:.2f} 文本/秒")
        print(f"最终特征形状: {final_features.shape}")
        
        final_memory = torch.cuda.memory_allocated()
        print(f"最终GPU内存: {final_memory / 1024**2:.2f} MB")
        print(f"总内存使用: {(final_memory - initial_memory) / 1024**2:.2f} MB")
        
        # 测试检索索引
        print("\n=== 测试检索索引 ===")
        if isinstance(final_features, torch.Tensor):
            features_np = final_features.cpu().numpy()
        else:
            features_np = final_features
        
        # 创建检索索引
        index = RetrievalIndex(index_type='faiss')
        index.build_index(features_np)
        
        # 测试搜索
        query_features = features_np[:10]  # 使用前10个作为查询
        search_start = time.time()
        indices, distances = index.search(query_features, top_k=20)
        search_time = time.time() - search_start
        
        print(f"检索索引构建完成")
        print(f"搜索时间: {search_time*1000:.2f} 毫秒")
        print(f"搜索结果形状: indices={indices.shape}, distances={distances.shape}")
        
        print("\n✅ GPU密集测试完成")
        
    except Exception as e:
        print(f"❌ GPU密集测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理GPU内存
        torch.cuda.empty_cache()
        final_cleanup_memory = torch.cuda.memory_allocated()
        print(f"\n清理后GPU内存: {final_cleanup_memory / 1024**2:.2f} MB")

if __name__ == "__main__":
    test_gpu_intensive()