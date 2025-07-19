#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态检测一致性实验代码 - 使用示例

本示例展示了如何使用项目的主要功能：
1. 文本变体生成
2. 多模态检索
3. Stable Diffusion参考生成
4. 对抗检测
5. 完整检测流水线
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 导入项目模块
import src
from src import (
    TextAugmenter, TextAugmentConfig,
    MultiModalRetriever, RetrievalConfig,
    SDReferenceGenerator, SDReferenceConfig,
    AdversarialDetector, DetectorConfig,
    MultiModalDetectionPipeline, PipelineConfig
)

def main():
    """
    主函数：演示项目的基本使用方法
    """
    print("🚀 多模态检测一致性实验代码 - 使用示例")
    print("=" * 60)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 使用设备: {device}")
    
    # 1. 文本变体生成示例
    print("\n1️⃣ 文本变体生成")
    print("-" * 30)
    
    # 创建文本增强器配置
    text_config = TextAugmentConfig(
        num_variants=3,
        similarity_threshold=0.8,
        synonym_prob=0.3,
        paraphrase_temperature=0.8
    )
    
    # 创建文本增强器
    text_augmenter = TextAugmenter(text_config)
    
    # 示例文本
    original_text = "A beautiful sunset over the ocean"
    print(f"原始文本: {original_text}")
    
    # 生成变体（注意：这里只是示例，实际运行可能需要模型文件）
    try:
        variants = text_augmenter.generate_variants(original_text, methods=["synonym", "paraphrase"])
        print(f"生成的变体: {variants}")
    except Exception as e:
        print(f"文本变体生成需要额外的模型文件: {e}")
    
    # 2. 多模态检索示例
    print("\n2️⃣ 多模态检索")
    print("-" * 30)
    
    # 创建检索器配置
    retrieval_config = RetrievalConfig(
        clip_model="ViT-B/32",
        device=device,
        batch_size=16
    )
    
    # 创建检索器
    try:
        retriever = MultiModalRetriever(retrieval_config)
        print("✅ 多模态检索器创建成功")
        
        # 示例：编码文本
        text_features = retriever.encode_text([original_text])
        print(f"文本特征维度: {text_features.shape}")
        
    except Exception as e:
        print(f"检索器初始化需要下载CLIP模型: {e}")
    
    # 3. Stable Diffusion参考生成示例
    print("\n3️⃣ Stable Diffusion参考生成")
    print("-" * 30)
    
    # 创建SD生成器配置
    sd_config = SDReferenceConfig(
        model_name="runwayml/stable-diffusion-v1-5",
        device=device,
        num_inference_steps=20
    )
    
    # 创建SD生成器
    try:
        sd_generator = SDReferenceGenerator(sd_config)
        print("✅ SD参考生成器创建成功")
        
        # 示例：生成参考图像（需要实际模型）
        # reference_images = sd_generator.generate_references([original_text])
        print("SD参考生成需要下载Stable Diffusion模型")
        
    except Exception as e:
        print(f"SD生成器初始化需要下载模型: {e}")
    
    # 4. 对抗检测示例
    print("\n4️⃣ 对抗检测")
    print("-" * 30)
    
    # 创建检测器配置
    detector_config = DetectorConfig(
        detection_method="similarity",
        threshold=0.8,
        device=device
    )
    
    # 创建检测器
    try:
        detector = AdversarialDetector(detector_config)
        print("✅ 对抗检测器创建成功")
        
        # 示例检测（需要实际数据）
        print("对抗检测需要输入图像和文本数据")
        
    except Exception as e:
        print(f"检测器初始化错误: {e}")
    
    # 5. 完整检测流水线示例
    print("\n5️⃣ 完整检测流水线")
    print("-" * 30)
    
    # 创建流水线配置
    pipeline_config = PipelineConfig(
        text_augment_config=text_config,
        retrieval_config=retrieval_config,
        sd_config=sd_config,
        detector_config=detector_config
    )
    
    # 创建检测流水线
    try:
        pipeline = MultiModalDetectionPipeline(pipeline_config)
        print("✅ 多模态检测流水线创建成功")
        
        # 示例：运行检测（需要实际数据）
        print("完整流水线需要输入图像和文本数据进行检测")
        
    except Exception as e:
        print(f"流水线初始化错误: {e}")
    
    # 6. 项目信息
    print("\n6️⃣ 项目信息")
    print("-" * 30)
    print(f"项目版本: {src.get_version()}")
    print(f"支持的CLIP模型: {src.get_supported_models()['clip'][:3]}...")
    print(f"支持的数据集: {src.get_supported_datasets()}")
    print(f"默认配置: {src.get_default_config()}")
    
    print("\n🎉 示例运行完成！")
    print("\n💡 提示:")
    print("- 完整功能需要下载相应的预训练模型")
    print("- 请参考configs/目录中的配置文件")
    print("- 查看experiments/目录了解实验设置")
    print("- 使用src.utils中的工具函数进行数据处理")

if __name__ == "__main__":
    main()