#!/usr/bin/env python3
"""
简单的导入测试脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("测试基础导入...")

# 测试基础模块导入（不初始化模型）
try:
    import torch
    print("✓ PyTorch 导入成功")
except Exception as e:
    print(f"✗ PyTorch 导入失败: {e}")

try:
    from PIL import Image
    print("✓ PIL 导入成功")
except Exception as e:
    print(f"✗ PIL 导入失败: {e}")

try:
    import numpy as np
    print("✓ NumPy 导入成功")
except Exception as e:
    print(f"✗ NumPy 导入失败: {e}")

# 测试项目模块导入（不初始化）
try:
    from src.utils.metrics import SimilarityMetrics, DetectionEvaluator
    print("✓ utils.metrics 导入成功")
except Exception as e:
    print(f"✗ utils.metrics 导入失败: {e}")

try:
    # 只导入类定义，不初始化
    from src.models.clip_model import CLIPConfig
    print("✓ CLIPConfig 导入成功")
except Exception as e:
    print(f"✗ CLIPConfig 导入失败: {e}")

try:
    from src.text_augment import TextAugmentConfig
    print("✓ TextAugmentConfig 导入成功")
except Exception as e:
    print(f"✗ TextAugmentConfig 导入失败: {e}")

try:
    from src.sd_ref import SDReferenceConfig
    print("✓ SDReferenceConfig 导入成功")
except Exception as e:
    print(f"✗ SDReferenceConfig 导入失败: {e}")

try:
    from src.detector import DetectorConfig
    print("✓ DetectorConfig 导入成功")
except Exception as e:
    print(f"✗ DetectorConfig 导入失败: {e}")

print("\n基础导入测试完成")