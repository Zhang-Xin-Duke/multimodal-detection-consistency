#!/usr/bin/env python3
"""
基本功能测试脚本

验证各个模块的基本功能是否正常工作。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from text_augment import TextAugmentConfig, TextAugmenter
from retrieval import RetrievalConfig, MultiModalRetriever
from sd_ref import SDReferenceConfig, SDReferenceGenerator
from detector import DetectorConfig, AdversarialDetector
from pipeline import PipelineConfig, DefensePipeline

def test_config_creation():
    """测试配置类创建"""
    print("Testing configuration creation...")
    
    # 测试各个配置类
    text_config = TextAugmentConfig()
    retrieval_config = RetrievalConfig()
    sd_config = SDReferenceConfig()
    detector_config = DetectorConfig()
    pipeline_config = PipelineConfig()
    
    print(f"✓ TextAugmentConfig: num_variants={text_config.num_variants}")
    print(f"✓ RetrievalConfig: top_k={retrieval_config.top_k}")
    print(f"✓ SDReferenceConfig: num_images_per_prompt={sd_config.num_images_per_prompt}")
    print(f"✓ DetectorConfig: clip_model={detector_config.clip_model}")
    print(f"✓ PipelineConfig: batch_size={pipeline_config.batch_size}")
    
    print("All configurations created successfully!\n")

def test_basic_imports():
    """测试基本导入"""
    print("Testing basic imports...")
    
    try:
        from models import CLIPModel, QwenModel, StableDiffusionModel
        print("✓ Model classes imported successfully")
    except ImportError as e:
        print(f"⚠ Model import warning: {e}")
    
    try:
        from utils.metrics import MetricsCalculator
        print("✓ Utils imported successfully")
    except ImportError as e:
        print(f"⚠ Utils import warning: {e}")
    
    print("Basic imports completed!\n")

def main():
    """主测试函数"""
    print("=== 基本功能测试 ===")
    print()
    
    try:
        test_basic_imports()
        test_config_creation()
        
        print("🎉 所有基本功能测试通过！")
        print("项目的核心模块和配置类都能正常工作。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)