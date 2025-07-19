#!/usr/bin/env python3
"""
使用国内镜像下载大模型脚本
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import requests
from huggingface_hub import snapshot_download, hf_hub_download
from modelscope import snapshot_download as ms_snapshot_download
from modelscope.hub.file_download import model_file_download
import torch

def setup_domestic_mirrors():
    """配置国内镜像源"""
    print("=== 配置国内镜像源 ===")
    
    # 设置Hugging Face镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 设置pip镜像
    pip_config_commands = [
        "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple",
        "pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn"
    ]
    
    for cmd in pip_config_commands:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {cmd}")
            else:
                print(f"✗ {cmd}: {result.stderr}")
        except Exception as e:
            print(f"✗ {cmd}: {e}")
    
    print("镜像源配置完成\n")

def download_clip_models():
    """下载CLIP模型"""
    print("=== 下载CLIP模型 ===")
    
    clip_models = {
        "CLIP ViT-B/32": "openai/clip-vit-base-patch32",
        "CLIP ViT-B/16": "openai/clip-vit-base-patch16", 
        "CLIP ViT-L/14": "openai/clip-vit-large-patch14"
    }
    
    cache_dir = "./data/cache/models/clip"
    os.makedirs(cache_dir, exist_ok=True)
    
    for model_name, model_id in clip_models.items():
        print(f"\n下载 {model_name} ({model_id})...")
        
        try:
            # 方法1: 尝试使用魔搭社区
            print(f"  尝试从魔搭社区下载...")
            start_time = time.time()
            
            # 魔搭社区的模型ID映射
            ms_model_mapping = {
                "openai/clip-vit-base-patch32": "AI-ModelScope/clip-vit-base-patch32",
                "openai/clip-vit-base-patch16": "AI-ModelScope/clip-vit-base-patch16",
                "openai/clip-vit-large-patch14": "AI-ModelScope/clip-vit-large-patch14"
            }
            
            if model_id in ms_model_mapping:
                ms_model_id = ms_model_mapping[model_id]
                model_dir = ms_snapshot_download(
                    ms_model_id,
                    cache_dir=cache_dir,
                    revision='master'
                )
                end_time = time.time()
                download_time = end_time - start_time
                print(f"  ✓ 魔搭社区下载成功: {model_dir}")
                print(f"  下载时间: {download_time:.2f} 秒")
                continue
                
        except Exception as e:
            print(f"  ✗ 魔搭社区下载失败: {e}")
        
        try:
            # 方法2: 使用HF镜像
            print(f"  尝试从HF镜像下载...")
            start_time = time.time()
            
            model_dir = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False
            )
            
            end_time = time.time()
            download_time = end_time - start_time
            print(f"  ✓ HF镜像下载成功: {model_dir}")
            print(f"  下载时间: {download_time:.2f} 秒")
            
        except Exception as e:
            print(f"  ✗ HF镜像下载失败: {e}")
            
        try:
            # 方法3: 直接使用transformers加载（会自动下载）
            print(f"  尝试使用transformers自动下载...")
            start_time = time.time()
            
            from transformers import CLIPModel, CLIPProcessor
            
            model = CLIPModel.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                resume_download=True
            )
            processor = CLIPProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                resume_download=True
            )
            
            end_time = time.time()
            download_time = end_time - start_time
            print(f"  ✓ transformers自动下载成功")
            print(f"  下载时间: {download_time:.2f} 秒")
            
            # 清理内存
            del model, processor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ transformers自动下载失败: {e}")
            print(f"  {model_name} 下载失败，请检查网络连接")

def download_stable_diffusion_models():
    """下载Stable Diffusion模型"""
    print("\n=== 下载Stable Diffusion模型 ===")
    
    sd_models = {
        "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5"
    }
    
    cache_dir = "./data/cache/models/stable_diffusion"
    os.makedirs(cache_dir, exist_ok=True)
    
    for model_name, model_id in sd_models.items():
        print(f"\n下载 {model_name} ({model_id})...")
        
        try:
            # 方法1: 尝试使用魔搭社区
            print(f"  尝试从魔搭社区下载...")
            start_time = time.time()
            
            # 魔搭社区的SD模型ID
            ms_model_mapping = {
                "runwayml/stable-diffusion-v1-5": "AI-ModelScope/stable-diffusion-v1-5"
            }
            
            if model_id in ms_model_mapping:
                ms_model_id = ms_model_mapping[model_id]
                model_dir = ms_snapshot_download(
                    ms_model_id,
                    cache_dir=cache_dir,
                    revision='master'
                )
                end_time = time.time()
                download_time = end_time - start_time
                print(f"  ✓ 魔搭社区下载成功: {model_dir}")
                print(f"  下载时间: {download_time:.2f} 秒 ({download_time/60:.2f} 分钟)")
                continue
                
        except Exception as e:
            print(f"  ✗ 魔搭社区下载失败: {e}")
        
        try:
            # 方法2: 使用diffusers库下载
            print(f"  尝试使用diffusers下载...")
            start_time = time.time()
            
            from diffusers import StableDiffusionPipeline
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                resume_download=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            end_time = time.time()
            download_time = end_time - start_time
            print(f"  ✓ diffusers下载成功")
            print(f"  下载时间: {download_time:.2f} 秒 ({download_time/60:.2f} 分钟)")
            
            # 清理内存
            del pipeline
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ diffusers下载失败: {e}")
        
        try:
            # 方法3: 使用HF镜像
            print(f"  尝试从HF镜像下载...")
            start_time = time.time()
            
            model_dir = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False
            )
            
            end_time = time.time()
            download_time = end_time - start_time
            print(f"  ✓ HF镜像下载成功: {model_dir}")
            print(f"  下载时间: {download_time:.2f} 秒 ({download_time/60:.2f} 分钟)")
            
        except Exception as e:
            print(f"  ✗ HF镜像下载失败: {e}")
            print(f"  {model_name} 下载失败，请检查网络连接")

def check_downloaded_models():
    """检查已下载的模型"""
    print("\n=== 检查已下载模型 ===")
    
    cache_dirs = [
        "./data/cache/models/clip",
        "./data/cache/models/stable_diffusion",
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/modelscope")
    ]
    
    total_size = 0
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                result = subprocess.run(
                    ["du", "-sh", cache_dir], 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                if result.returncode == 0:
                    size_str = result.stdout.split()[0]
                    print(f"✓ {cache_dir}: {size_str}")
                    
                    # 尝试解析大小
                    try:
                        if 'G' in size_str:
                            size_gb = float(size_str.replace('G', ''))
                            total_size += size_gb
                        elif 'M' in size_str:
                            size_mb = float(size_str.replace('M', ''))
                            total_size += size_mb / 1024
                    except:
                        pass
                else:
                    print(f"? {cache_dir}: 无法计算大小")
            except Exception as e:
                print(f"? {cache_dir}: 检查失败 - {e}")
        else:
            print(f"✗ {cache_dir}: 不存在")
    
    if total_size > 0:
        print(f"\n总缓存大小约: {total_size:.2f} GB")

def test_model_loading():
    """测试模型加载"""
    print("\n=== 测试模型加载 ===")
    
    # 测试CLIP模型加载
    try:
        print("测试CLIP模型加载...")
        from transformers import CLIPModel, CLIPProcessor
        
        model_id = "openai/clip-vit-base-patch32"
        start_time = time.time()
        
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        
        end_time = time.time()
        load_time = end_time - start_time
        
        print(f"✓ CLIP模型加载成功 (耗时: {load_time:.2f} 秒)")
        
        # 清理内存
        del model, processor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ CLIP模型加载失败: {e}")
    
    # 测试Stable Diffusion模型加载
    try:
        print("测试Stable Diffusion模型加载...")
        from diffusers import StableDiffusionPipeline
        
        model_id = "runwayml/stable-diffusion-v1-5"
        start_time = time.time()
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        
        end_time = time.time()
        load_time = end_time - start_time
        
        print(f"✓ Stable Diffusion模型加载成功 (耗时: {load_time:.2f} 秒)")
        
        # 清理内存
        del pipeline
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ Stable Diffusion模型加载失败: {e}")

def install_required_packages():
    """安装必要的包"""
    print("=== 安装必要的包 ===")
    
    packages = [
        "modelscope",
        "huggingface_hub",
        "transformers",
        "diffusers",
        "accelerate"
    ]
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "--upgrade"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"✓ {package} 安装成功")
            else:
                print(f"✗ {package} 安装失败: {result.stderr}")
                
        except Exception as e:
            print(f"✗ {package} 安装异常: {e}")

def main():
    """主函数"""
    print("使用国内镜像下载大模型")
    print("=" * 50)
    
    # 1. 安装必要的包
    install_required_packages()
    
    # 2. 配置国内镜像源
    setup_domestic_mirrors()
    
    # 3. 下载CLIP模型
    download_clip_models()
    
    # 4. 下载Stable Diffusion模型
    download_stable_diffusion_models()
    
    # 5. 检查已下载的模型
    check_downloaded_models()
    
    # 6. 测试模型加载
    test_model_loading()
    
    print("\n=== 下载完成 ===")
    print("💡 提示:")
    print("1. 模型已下载到本地缓存，后续使用会直接从缓存加载")
    print("2. 如果某些模型下载失败，可以单独重试")
    print("3. 建议定期清理不需要的模型缓存以节省磁盘空间")
    print("4. 可以使用 'huggingface-cli scan-cache' 查看缓存详情")

if __name__ == "__main__":
    main()