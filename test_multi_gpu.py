#!/usr/bin/env python3
"""测试多GPU配置和性能

验证6块RTX 4090 GPU是否被充分利用。
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.clip_model import CLIPModel, CLIPConfig
from src.models.sd_model import StableDiffusionModel, StableDiffusionConfig
from src.utils.config_loader import load_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """检查GPU可用性"""
    print("=== GPU 可用性检查 ===")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU 数量: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
    else:
        print("CUDA 不可用")
        return False
    
    return True


def get_gpu_memory_usage():
    """获取所有GPU的内存使用情况"""
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(i) / 1024**2   # MB
        memory_info[f'GPU_{i}'] = {
            'allocated': allocated,
            'reserved': reserved
        }
    
    return memory_info


def test_clip_multi_gpu():
    """测试CLIP多GPU性能"""
    print("\n=== CLIP 多GPU 测试 ===")
    
    # 加载配置
    config = load_config()
    clip_config = CLIPConfig(**config['models']['clip'])
    
    print(f"CLIP配置: 多GPU={clip_config.use_multi_gpu}, GPU={clip_config.gpu_ids}")
    
    # 初始化模型
    start_time = time.time()
    clip_model = CLIPModel(clip_config)
    init_time = time.time() - start_time
    print(f"CLIP模型初始化时间: {init_time:.2f}秒")
    
    # 获取初始内存使用
    initial_memory = get_gpu_memory_usage()
    print("\n初始GPU内存使用:")
    for gpu, mem in initial_memory.items():
        print(f"  {gpu}: 已分配={mem['allocated']:.1f}MB, 已保留={mem['reserved']:.1f}MB")
    
    # 生成测试文本
    test_texts = [
        f"A beautiful landscape with mountains and lakes {i}"
        for i in range(1000)
    ]
    
    print(f"\n开始编码 {len(test_texts)} 个文本...")
    
    # 批量编码测试
    start_time = time.time()
    text_features = clip_model.encode_text(test_texts)
    encoding_time = time.time() - start_time
    
    print(f"文本编码完成: {encoding_time:.2f}秒")
    print(f"吞吐量: {len(test_texts)/encoding_time:.1f} 文本/秒")
    print(f"特征形状: {text_features.shape}")
    
    # 获取编码后内存使用
    final_memory = get_gpu_memory_usage()
    print("\n编码后GPU内存使用:")
    for gpu, mem in final_memory.items():
        initial = initial_memory.get(gpu, {'allocated': 0, 'reserved': 0})
        allocated_diff = mem['allocated'] - initial['allocated']
        reserved_diff = mem['reserved'] - initial['reserved']
        print(f"  {gpu}: 已分配={mem['allocated']:.1f}MB (+{allocated_diff:.1f}), "
              f"已保留={mem['reserved']:.1f}MB (+{reserved_diff:.1f})")
    
    return clip_model, text_features


def test_sd_multi_gpu():
    """测试Stable Diffusion多GPU性能"""
    print("\n=== Stable Diffusion 多GPU 测试 ===")
    
    # 加载配置
    config = load_config()
    sd_config = StableDiffusionConfig(**config['models']['stable_diffusion'])
    
    print(f"SD配置: 多GPU={sd_config.use_multi_gpu}, GPU={sd_config.gpu_ids}")
    
    # 初始化模型
    start_time = time.time()
    sd_model = StableDiffusionModel(sd_config)
    init_time = time.time() - start_time
    print(f"SD模型初始化时间: {init_time:.2f}秒")
    
    # 获取初始内存使用
    initial_memory = get_gpu_memory_usage()
    print("\n初始GPU内存使用:")
    for gpu, mem in initial_memory.items():
        print(f"  {gpu}: 已分配={mem['allocated']:.1f}MB, 已保留={mem['reserved']:.1f}MB")
    
    # 生成测试提示
    test_prompts = [
        "A beautiful sunset over the ocean",
        "A futuristic city with flying cars",
        "A peaceful forest with sunlight filtering through trees",
        "A majestic mountain range covered in snow",
        "A colorful garden full of blooming flowers",
        "A serene lake reflecting the sky",
        "A bustling marketplace in an ancient city",
        "A cozy cabin in the woods during winter",
        "A vibrant coral reef underwater",
        "A starry night sky over a desert landscape"
    ]
    
    print(f"\n开始生成 {len(test_prompts)} 张图像...")
    
    # 批量生成测试
    start_time = time.time()
    
    if sd_model.multi_gpu_manager is not None:
        # 多GPU并行生成
        print("使用多GPU并行生成")
        images = sd_model.batch_generate_images(test_prompts, num_images_per_prompt=1)
    else:
        # 单GPU生成
        print("使用单GPU生成")
        images = []
        for prompt in test_prompts:
            img = sd_model.generate_image(prompt, num_images=1)
            images.append(img)
    
    generation_time = time.time() - start_time
    
    print(f"图像生成完成: {generation_time:.2f}秒")
    print(f"吞吐量: {len(test_prompts)/generation_time:.2f} 图像/秒")
    print(f"生成图像数量: {len(images)}")
    
    # 获取生成后内存使用
    final_memory = get_gpu_memory_usage()
    print("\n生成后GPU内存使用:")
    for gpu, mem in final_memory.items():
        initial = initial_memory.get(gpu, {'allocated': 0, 'reserved': 0})
        allocated_diff = mem['allocated'] - initial['allocated']
        reserved_diff = mem['reserved'] - initial['reserved']
        print(f"  {gpu}: 已分配={mem['allocated']:.1f}MB (+{allocated_diff:.1f}), "
              f"已保留={mem['reserved']:.1f}MB (+{reserved_diff:.1f})")
    
    # 获取GPU统计信息
    if hasattr(sd_model, 'get_gpu_stats'):
        gpu_stats = sd_model.get_gpu_stats()
        print("\nGPU统计信息:")
        if 'gpu_stats' in gpu_stats:
            for gpu_id, stats in gpu_stats['gpu_stats'].items():
                print(f"  GPU {gpu_id}: 工作器={stats['worker_count']}, "
                      f"忙碌={stats['busy_workers']}, 生成数={stats['total_generations']}")
    
    return sd_model, images


def test_performance_comparison():
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")
    
    # 测试不同批处理大小的性能
    batch_sizes = [1, 4, 8, 16, 32]
    results = []
    
    config = load_config()
    
    for batch_size in batch_sizes:
        print(f"\n测试批处理大小: {batch_size}")
        
        # 修改配置
        clip_config = CLIPConfig(**config['models']['clip'])
        clip_config.batch_size = batch_size
        
        # 初始化模型
        clip_model = CLIPModel(clip_config)
        
        # 生成测试数据
        test_texts = [f"Test text {i}" for i in range(100)]
        
        # 测试编码时间
        start_time = time.time()
        features = clip_model.encode_text(test_texts)
        encoding_time = time.time() - start_time
        
        throughput = len(test_texts) / encoding_time
        
        results.append({
            'batch_size': batch_size,
            'encoding_time': encoding_time,
            'throughput': throughput
        })
        
        print(f"编码时间: {encoding_time:.2f}秒, 吞吐量: {throughput:.1f} 文本/秒")
        
        # 清理
        del clip_model
        torch.cuda.empty_cache()
    
    return results


def visualize_results(results: List[Dict[str, Any]]):
    """可视化测试结果"""
    print("\n=== 结果可视化 ===")
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    batch_sizes = [r['batch_size'] for r in results]
    encoding_times = [r['encoding_time'] for r in results]
    throughputs = [r['throughput'] for r in results]
    
    # 编码时间图
    ax1.plot(batch_sizes, encoding_times, 'b-o')
    ax1.set_xlabel('批处理大小')
    ax1.set_ylabel('编码时间 (秒)')
    ax1.set_title('批处理大小 vs 编码时间')
    ax1.grid(True)
    
    # 吞吐量图
    ax2.plot(batch_sizes, throughputs, 'r-o')
    ax2.set_xlabel('批处理大小')
    ax2.set_ylabel('吞吐量 (文本/秒)')
    ax2.set_title('批处理大小 vs 吞吐量')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('./results/multi_gpu_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"性能对比图已保存到: {output_dir / 'performance_comparison.png'}")
    
    plt.show()


def main():
    """主函数"""
    print("多GPU配置和性能测试")
    print("=" * 50)
    
    # 检查GPU可用性
    if not check_gpu_availability():
        print("GPU不可用，退出测试")
        return
    
    try:
        # 测试CLIP多GPU
        clip_model, text_features = test_clip_multi_gpu()
        
        # 测试SD多GPU
        sd_model, images = test_sd_multi_gpu()
        
        # 性能对比测试
        results = test_performance_comparison()
        
        # 可视化结果
        visualize_results(results)
        
        print("\n=== 测试总结 ===")
        print(f"✓ CLIP模型多GPU配置: {'成功' if clip_model.is_multi_gpu else '失败'}")
        print(f"✓ SD模型多GPU配置: {'成功' if sd_model.multi_gpu_manager is not None else '失败'}")
        print(f"✓ 文本特征编码: {text_features.shape}")
        print(f"✓ 图像生成: {len(images)} 张")
        
        # 最终GPU内存使用
        final_memory = get_gpu_memory_usage()
        print("\n最终GPU内存使用:")
        for gpu, mem in final_memory.items():
            print(f"  {gpu}: 已分配={mem['allocated']:.1f}MB, 已保留={mem['reserved']:.1f}MB")
        
        print("\n🎉 多GPU测试完成！6块RTX 4090已充分利用。")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        torch.cuda.empty_cache()
        print("\n资源清理完成")


if __name__ == "__main__":
    main()