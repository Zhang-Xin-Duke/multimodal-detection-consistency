#!/usr/bin/env python3
"""
国内镜像源网络速度测试脚本
"""

import time
import requests
import subprocess
import sys
import os
from pathlib import Path
import json
from urllib.parse import urlparse

def test_domestic_mirrors():
    """测试国内主要镜像源的连接速度"""
    print("=== 国内镜像源连接测试 ===")
    
    # 国内主要镜像源
    domestic_mirrors = {
        "清华大学镜像": "https://mirrors.tuna.tsinghua.edu.cn",
        "阿里云镜像": "https://mirrors.aliyun.com",
        "华为云镜像": "https://mirrors.huaweicloud.com",
        "中科大镜像": "https://mirrors.ustc.edu.cn",
        "网易镜像": "https://mirrors.163.com",
        "腾讯云镜像": "https://mirrors.cloud.tencent.com",
        "百度云镜像": "https://mirror.baidu.com"
    }
    
    results = {}
    
    for name, url in domestic_mirrors.items():
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10, allow_redirects=True)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                results[name] = {
                    'status': '成功',
                    'latency_ms': round(latency, 2),
                    'status_code': response.status_code,
                    'url': url
                }
                print(f"✓ {name}: 连接成功 (延迟: {latency:.2f}ms)")
            else:
                results[name] = {
                    'status': '失败',
                    'status_code': response.status_code,
                    'url': url
                }
                print(f"✗ {name}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            results[name] = {
                'status': '连接失败',
                'error': str(e),
                'url': url
            }
            print(f"✗ {name}: 连接失败 - {e}")
    
    return results

def test_huggingface_mirrors():
    """测试Hugging Face国内镜像源"""
    print("\n=== Hugging Face 国内镜像测试 ===")
    
    hf_mirrors = {
        "Hugging Face 官方": "https://huggingface.co",
        "魔搭社区": "https://www.modelscope.cn",
        "始智AI": "https://www.wisemodel.cn",
        "OpenI启智": "https://openi.pcl.ac.cn"
    }
    
    results = {}
    
    for name, url in hf_mirrors.items():
        try:
            start_time = time.time()
            response = requests.get(url, timeout=15, allow_redirects=True)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                results[name] = {
                    'status': '成功',
                    'latency_ms': round(latency, 2),
                    'status_code': response.status_code,
                    'url': url
                }
                print(f"✓ {name}: 连接成功 (延迟: {latency:.2f}ms)")
            else:
                results[name] = {
                    'status': '失败',
                    'status_code': response.status_code,
                    'url': url
                }
                print(f"✗ {name}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            results[name] = {
                'status': '连接失败',
                'error': str(e),
                'url': url
            }
            print(f"✗ {name}: 连接失败 - {e}")
    
    return results

def test_download_speed_domestic():
    """测试国内镜像的下载速度"""
    print("\n=== 国内镜像下载速度测试 ===")
    
    # 测试文件列表（选择较小的文件进行测试）
    test_files = [
        {
            'name': '清华镜像 - Ubuntu索引',
            'url': 'https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ls-lR.gz',
            'expected_size_mb': 1
        },
        {
            'name': '阿里云镜像 - CentOS文件',
            'url': 'https://mirrors.aliyun.com/centos/README',
            'expected_size_mb': 0.001
        },
        {
            'name': '中科大镜像 - Debian文件',
            'url': 'https://mirrors.ustc.edu.cn/debian/README',
            'expected_size_mb': 0.001
        }
    ]
    
    download_results = []
    
    for test_file in test_files:
        try:
            print(f"\n测试下载: {test_file['name']}")
            start_time = time.time()
            
            response = requests.get(test_file['url'], timeout=30, stream=True)
            
            if response.status_code == 200:
                # 下载内容
                content = b''
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                
                end_time = time.time()
                
                file_size = len(content)
                download_time = end_time - start_time
                speed_kbps = (file_size / 1024) / download_time if download_time > 0 else 0
                speed_mbps = speed_kbps / 1024
                
                result = {
                    'name': test_file['name'],
                    'url': test_file['url'],
                    'file_size_bytes': file_size,
                    'file_size_kb': round(file_size / 1024, 2),
                    'download_time_seconds': round(download_time, 2),
                    'speed_kbps': round(speed_kbps, 2),
                    'speed_mbps': round(speed_mbps, 2),
                    'status': '成功'
                }
                
                download_results.append(result)
                
                print(f"  ✓ 下载成功")
                print(f"    - 文件大小: {file_size} bytes ({file_size/1024:.2f} KB)")
                print(f"    - 下载时间: {download_time:.2f} 秒")
                print(f"    - 下载速度: {speed_kbps:.2f} KB/s ({speed_mbps:.2f} MB/s)")
            else:
                print(f"  ✗ 下载失败: HTTP {response.status_code}")
                download_results.append({
                    'name': test_file['name'],
                    'url': test_file['url'],
                    'status': '失败',
                    'status_code': response.status_code
                })
                
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            download_results.append({
                'name': test_file['name'],
                'url': test_file['url'],
                'status': '异常',
                'error': str(e)
            })
    
    return download_results

def test_pip_mirrors():
    """测试pip国内镜像源"""
    print("\n=== pip 国内镜像源测试 ===")
    
    pip_mirrors = {
        "清华大学": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "阿里云": "https://mirrors.aliyun.com/pypi/simple",
        "中科大": "https://pypi.mirrors.ustc.edu.cn/simple",
        "华为云": "https://mirrors.huaweicloud.com/repository/pypi/simple",
        "腾讯云": "https://mirrors.cloud.tencent.com/pypi/simple",
        "豆瓣": "https://pypi.douban.com/simple"
    }
    
    results = {}
    
    for name, url in pip_mirrors.items():
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                results[name] = {
                    'status': '成功',
                    'latency_ms': round(latency, 2),
                    'status_code': response.status_code,
                    'url': url
                }
                print(f"✓ {name}: 连接成功 (延迟: {latency:.2f}ms)")
            else:
                results[name] = {
                    'status': '失败',
                    'status_code': response.status_code,
                    'url': url
                }
                print(f"✗ {name}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            results[name] = {
                'status': '连接失败',
                'error': str(e),
                'url': url
            }
            print(f"✗ {name}: 连接失败 - {e}")
    
    return results

def estimate_model_download_time_domestic():
    """基于国内网络速度估算模型下载时间"""
    print("\n=== 基于国内网络的模型下载时间估算 ===")
    
    # 先测试一个实际的下载来估算速度
    download_results = test_download_speed_domestic()
    
    # 计算平均下载速度
    successful_downloads = [r for r in download_results if r.get('status') == '成功' and 'speed_mbps' in r]
    
    if successful_downloads:
        avg_speed_mbps = sum(r['speed_mbps'] for r in successful_downloads) / len(successful_downloads)
        max_speed_mbps = max(r['speed_mbps'] for r in successful_downloads)
        min_speed_mbps = min(r['speed_mbps'] for r in successful_downloads)
        
        print(f"\n国内网络下载速度统计:")
        print(f"  - 平均速度: {avg_speed_mbps:.2f} MB/s")
        print(f"  - 最快速度: {max_speed_mbps:.2f} MB/s")
        print(f"  - 最慢速度: {min_speed_mbps:.2f} MB/s")
        
        # 模型大小信息
        model_sizes = {
            "CLIP ViT-B/32": 150,
            "CLIP ViT-B/16": 350,
            "CLIP ViT-L/14": 890,
            "Stable Diffusion v1.5": 4000,
            "Stable Diffusion XL": 6900,
            "Qwen-7B": 14000
        }
        
        print(f"\n基于国内网络速度的模型下载时间估算:")
        print("-" * 70)
        
        for model_name, size_mb in model_sizes.items():
            # 使用平均速度估算
            avg_time_seconds = size_mb / avg_speed_mbps if avg_speed_mbps > 0 else float('inf')
            avg_time_minutes = avg_time_seconds / 60
            
            # 使用最快速度估算
            best_time_seconds = size_mb / max_speed_mbps if max_speed_mbps > 0 else float('inf')
            best_time_minutes = best_time_seconds / 60
            
            print(f"{model_name} ({size_mb} MB):")
            print(f"  - 平均速度下载时间: {avg_time_seconds:.1f} 秒 ({avg_time_minutes:.1f} 分钟)")
            print(f"  - 最佳速度下载时间: {best_time_seconds:.1f} 秒 ({best_time_minutes:.1f} 分钟)")
            print()
    else:
        print("⚠️  无法获取有效的下载速度数据")

def check_dns_resolution():
    """检查DNS解析速度"""
    print("\n=== DNS解析速度测试 ===")
    
    test_domains = [
        "huggingface.co",
        "github.com",
        "mirrors.tuna.tsinghua.edu.cn",
        "mirrors.aliyun.com",
        "www.modelscope.cn"
    ]
    
    for domain in test_domains:
        try:
            start_time = time.time()
            import socket
            socket.gethostbyname(domain)
            end_time = time.time()
            
            dns_time = (end_time - start_time) * 1000
            print(f"✓ {domain}: DNS解析时间 {dns_time:.2f}ms")
            
        except Exception as e:
            print(f"✗ {domain}: DNS解析失败 - {e}")

def main():
    """主函数"""
    print("国内镜像源网络速度测试")
    print("=" * 50)
    
    # 1. 测试国内镜像源连接
    domestic_results = test_domestic_mirrors()
    
    # 2. 测试Hugging Face相关镜像
    hf_results = test_huggingface_mirrors()
    
    # 3. 测试pip镜像源
    pip_results = test_pip_mirrors()
    
    # 4. 测试下载速度
    download_results = test_download_speed_domestic()
    
    # 5. 估算模型下载时间
    estimate_model_download_time_domestic()
    
    # 6. DNS解析测试
    check_dns_resolution()
    
    # 7. 总结和建议
    print("\n=== 总结和建议 ===")
    
    # 统计成功的连接
    successful_domestic = sum(1 for r in domestic_results.values() if r['status'] == '成功')
    successful_hf = sum(1 for r in hf_results.values() if r['status'] == '成功')
    successful_pip = sum(1 for r in pip_results.values() if r['status'] == '成功')
    
    print(f"📊 连接成功率统计:")
    print(f"  - 国内镜像源: {successful_domestic}/{len(domestic_results)} ({successful_domestic/len(domestic_results)*100:.1f}%)")
    print(f"  - AI模型平台: {successful_hf}/{len(hf_results)} ({successful_hf/len(hf_results)*100:.1f}%)")
    print(f"  - pip镜像源: {successful_pip}/{len(pip_results)} ({successful_pip/len(pip_results)*100:.1f}%)")
    
    print(f"\n💡 优化建议:")
    
    if successful_domestic > len(domestic_results) * 0.7:
        print("✓ 国内镜像源连接良好，建议优先使用国内镜像")
        print("  - 配置pip使用国内镜像: pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple")
        print("  - 使用魔搭社区替代Hugging Face: https://www.modelscope.cn")
    else:
        print("⚠️  国内镜像源连接不稳定，可能需要检查网络配置")
    
    if successful_hf == 0:
        print("⚠️  Hugging Face连接失败，强烈建议使用国内AI模型平台")
        print("  - 魔搭社区: https://www.modelscope.cn")
        print("  - 始智AI: https://www.wisemodel.cn")
    
    # 检查下载速度
    successful_downloads = [r for r in download_results if r.get('status') == '成功']
    if successful_downloads:
        avg_speed = sum(r.get('speed_mbps', 0) for r in successful_downloads) / len(successful_downloads)
        if avg_speed > 1:  # 大于1MB/s
            print(f"✓ 国内网络下载速度良好 (平均 {avg_speed:.2f} MB/s)")
        else:
            print(f"⚠️  网络下载速度较慢 (平均 {avg_speed:.2f} MB/s)，建议在空闲时间下载大模型")
    
if __name__ == "__main__":
    main()