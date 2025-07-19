#!/usr/bin/env python3
"""
网络连接检查和大模型加载时间估算脚本
"""

import time
import requests
import subprocess
import sys
import os
from pathlib import Path
import torch
from urllib.parse import urlparse

def check_internet_connection():
    """检查互联网连接状态"""
    print("=== 网络连接检查 ===")
    
    # 测试基本网络连接
    test_urls = [
        "https://www.google.com",
        "https://www.baidu.com",
        "https://huggingface.co",
        "https://github.com"
    ]
    
    connection_results = {}
    
    for url in test_urls:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                latency = (end_time - start_time) * 1000  # 转换为毫秒
                connection_results[url] = {
                    'status': '成功',
                    'latency_ms': round(latency, 2),
                    'status_code': response.status_code
                }
                print(f"✓ {url}: 连接成功 (延迟: {latency:.2f}ms)")
            else:
                connection_results[url] = {
                    'status': '失败',
                    'status_code': response.status_code
                }
                print(f"✗ {url}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            connection_results[url] = {
                'status': '连接失败',
                'error': str(e)
            }
            print(f"✗ {url}: 连接失败 - {e}")
    
    return connection_results

def test_download_speed():
    """测试下载速度"""
    print("\n=== 下载速度测试 ===")
    
    # 测试小文件下载速度
    test_file_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json"
    
    try:
        start_time = time.time()
        response = requests.get(test_file_url, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            file_size = len(response.content)
            download_time = end_time - start_time
            speed_kbps = (file_size / 1024) / download_time
            
            print(f"✓ 测试文件下载成功")
            print(f"  - 文件大小: {file_size} bytes")
            print(f"  - 下载时间: {download_time:.2f} 秒")
            print(f"  - 下载速度: {speed_kbps:.2f} KB/s")
            
            return speed_kbps
        else:
            print(f"✗ 下载失败: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ 下载速度测试失败: {e}")
        return None

def estimate_model_loading_time():
    """估算各种模型的加载时间"""
    print("\n=== 模型加载时间估算 ===")
    
    # 模型大小信息（近似值）
    model_sizes = {
        "CLIP ViT-B/32": {
            "size_mb": 150,
            "description": "CLIP视觉-文本模型"
        },
        "CLIP ViT-B/16": {
            "size_mb": 350,
            "description": "CLIP高分辨率模型"
        },
        "CLIP ViT-L/14": {
            "size_mb": 890,
            "description": "CLIP大型模型"
        },
        "Stable Diffusion v1.5": {
            "size_mb": 4000,
            "description": "Stable Diffusion文本到图像生成模型"
        },
        "Stable Diffusion XL": {
            "size_mb": 6900,
            "description": "Stable Diffusion XL高质量模型"
        },
        "Qwen-7B": {
            "size_mb": 14000,
            "description": "Qwen 7B参数语言模型"
        }
    }
    
    # 假设下载速度（如果之前测试成功的话）
    download_speed_kbps = test_download_speed()
    if download_speed_kbps is None:
        # 使用保守估计
        download_speed_kbps = 500  # 500 KB/s
        print(f"使用保守估计下载速度: {download_speed_kbps} KB/s")
    
    print(f"\n基于下载速度 {download_speed_kbps:.2f} KB/s 的加载时间估算:")
    print("-" * 60)
    
    for model_name, info in model_sizes.items():
        size_kb = info["size_mb"] * 1024
        estimated_time_seconds = size_kb / download_speed_kbps
        estimated_time_minutes = estimated_time_seconds / 60
        
        print(f"{model_name}:")
        print(f"  - 大小: {info['size_mb']} MB")
        print(f"  - 描述: {info['description']}")
        print(f"  - 估算下载时间: {estimated_time_seconds:.1f} 秒 ({estimated_time_minutes:.1f} 分钟)")
        print()

def check_local_cache():
    """检查本地模型缓存"""
    print("=== 本地模型缓存检查 ===")
    
    # 常见的模型缓存目录
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch"),
        os.path.expanduser("~/.cache/clip"),
        "./data/cache/models",
        "./cache"
    ]
    
    total_cache_size = 0
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                # 计算目录大小
                result = subprocess.run(
                    ["du", "-sh", cache_dir], 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                if result.returncode == 0:
                    size_str = result.stdout.split()[0]
                    print(f"✓ {cache_dir}: {size_str}")
                else:
                    print(f"? {cache_dir}: 无法计算大小")
            except Exception as e:
                print(f"? {cache_dir}: 检查失败 - {e}")
        else:
            print(f"✗ {cache_dir}: 不存在")

def check_gpu_memory():
    """检查GPU内存状态"""
    print("\n=== GPU内存检查 ===")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU:")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            cached_memory = torch.cuda.memory_reserved(i)
            
            total_gb = total_memory / (1024**3)
            allocated_gb = allocated_memory / (1024**3)
            cached_gb = cached_memory / (1024**3)
            free_gb = total_gb - allocated_gb
            
            print(f"  GPU {i}: {gpu_name}")
            print(f"    - 总内存: {total_gb:.2f} GB")
            print(f"    - 已分配: {allocated_gb:.2f} GB")
            print(f"    - 已缓存: {cached_gb:.2f} GB")
            print(f"    - 可用: {free_gb:.2f} GB")
            print()
    else:
        print("未检测到CUDA GPU")

def main():
    """主函数"""
    print("多模态检测系统 - 网络连接和模型加载时间检查")
    print("=" * 60)
    
    # 1. 检查网络连接
    connection_results = check_internet_connection()
    
    # 2. 检查本地缓存
    check_local_cache()
    
    # 3. 估算模型加载时间
    estimate_model_loading_time()
    
    # 4. 检查GPU内存
    check_gpu_memory()
    
    # 5. 总结和建议
    print("\n=== 总结和建议 ===")
    
    # 检查是否有网络连接问题
    failed_connections = [url for url, result in connection_results.items() 
                         if result['status'] != '成功']
    
    if failed_connections:
        print("⚠️  网络连接问题:")
        for url in failed_connections:
            print(f"   - {url}: {connection_results[url]['status']}")
        print("   建议: 检查网络设置或使用代理")
    else:
        print("✓ 网络连接正常")
    
    print("\n💡 优化建议:")
    print("1. 首次运行时，模型会自动下载到本地缓存")
    print("2. 后续运行将直接从缓存加载，速度显著提升")
    print("3. 如果网络较慢，建议在空闲时间预先下载模型")
    print("4. 可以考虑使用较小的模型进行初步测试")
    print("5. 确保有足够的磁盘空间存储模型缓存")
    
if __name__ == "__main__":
    main()