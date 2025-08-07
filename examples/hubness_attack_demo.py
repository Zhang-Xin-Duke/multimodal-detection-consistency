#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hubness Attack 最小可运行示例 (MRE)

演示《Adversarial Hubness in Multi-Modal Retrieval》论文复现实现的使用方法

作者: 张昕 (ZHANG XIN)
学校: Duke University
邮箱: zhang.xin@duke.edu
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.attacks.hubness_attack import (
    HubnessAttacker, 
    HubnessAttackConfig, 
    HubnessAttackPresets
)


def create_sample_image(size=(224, 224), color='red'):
    """创建示例图像"""
    return Image.new('RGB', size, color=color)


def create_sample_queries():
    """创建示例查询"""
    return [
        "a photo of a dog",
        "a picture of a cat",
        "an image of a car",
        "a photo of a person",
        "a picture of a building",
        "an image of food",
        "a landscape photo",
        "a close-up shot"
    ]


def visualize_attack_result(original_image, adversarial_image, perturbation, result):
    """可视化攻击结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 转换张量为numpy数组用于显示
    if isinstance(original_image, torch.Tensor):
        original_np = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        original_np = np.array(original_image) / 255.0
    
    adversarial_np = adversarial_image.squeeze().permute(1, 2, 0).cpu().numpy()
    perturbation_np = perturbation.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # 显示原始图像
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示对抗图像
    axes[1].imshow(np.clip(adversarial_np, 0, 1))
    axes[1].set_title(f'Adversarial Image\n(Hubness: {result["hubness_score"]:.4f})')
    axes[1].axis('off')
    
    # 显示扰动（放大显示）
    perturbation_scaled = (perturbation_np - perturbation_np.min()) / (perturbation_np.max() - perturbation_np.min())
    axes[2].imshow(perturbation_scaled)
    axes[2].set_title('Perturbation (Scaled)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hubness_attack_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"攻击结果可视化已保存为 'hubness_attack_result.png'")


def demo_basic_attack():
    """演示基本攻击"""
    print("=" * 60)
    print("基本Hubness攻击演示")
    print("=" * 60)
    
    # 创建快速配置用于演示
    config = HubnessAttackConfig(
        epsilon=16.0 / 255.0,
        num_iterations=50,  # 减少迭代次数以加快演示
        step_size=0.02,
        device="cpu",  # 使用CPU确保兼容性
        random_seed=42
    )
    
    print(f"配置参数:")
    print(f"  - 扰动边界 (ε): {config.epsilon:.4f}")
    print(f"  - 迭代次数: {config.num_iterations}")
    print(f"  - 步长: {config.step_size}")
    print(f"  - 设备: {config.device}")
    print()
    
    # 初始化攻击器
    print("初始化Hubness攻击器...")
    attacker = HubnessAttacker(config)
    print("攻击器初始化完成！")
    print()
    
    # 创建示例图像和查询
    print("创建示例数据...")
    sample_image = create_sample_image(color='blue')
    sample_queries = create_sample_queries()
    
    print(f"图像尺寸: {sample_image.size}")
    print(f"查询数量: {len(sample_queries)}")
    print(f"查询示例: {sample_queries[:3]}")
    print()
    
    # 执行攻击
    print("执行Hubness攻击...")
    start_time = time.time()
    
    result = attacker.attack(sample_image, sample_queries)
    
    attack_time = time.time() - start_time
    print(f"攻击完成！耗时: {attack_time:.2f}秒")
    print()
    
    # 显示结果
    print("攻击结果:")
    print(f"  - 攻击成功: {'是' if result['success'] else '否'}")
    print(f"  - Hubness分数: {result['hubness_score']:.6f}")
    print(f"  - 迭代次数: {result['iterations']}")
    print(f"  - 攻击时间: {result['attack_time']:.3f}秒")
    
    # 计算扰动统计
    perturbation = result['perturbation']
    max_perturbation = torch.max(torch.abs(perturbation)).item()
    mean_perturbation = torch.mean(torch.abs(perturbation)).item()
    
    print(f"  - 最大扰动: {max_perturbation:.6f}")
    print(f"  - 平均扰动: {mean_perturbation:.6f}")
    print(f"  - 扰动约束: {config.epsilon:.6f}")
    print()
    
    # 获取攻击统计
    stats = attacker.get_attack_stats()
    print("攻击统计:")
    print(f"  - 总攻击次数: {stats['total_attacks']}")
    print(f"  - 成功次数: {stats['successful_attacks']}")
    print(f"  - 成功率: {stats['success_rate']:.2%}")
    print(f"  - 平均时间: {stats['average_time']:.3f}秒")
    print()
    
    return result, sample_image


def demo_preset_configs():
    """演示预设配置"""
    print("=" * 60)
    print("预设配置演示")
    print("=" * 60)
    
    presets = {
        "弱攻击": HubnessAttackPresets.weak_attack(),
        "强攻击": HubnessAttackPresets.strong_attack(),
        "针对性攻击": HubnessAttackPresets.targeted_attack(),
        "论文标准": HubnessAttackPresets.paper_config()
    }
    
    for name, config in presets.items():
        print(f"{name}配置:")
        print(f"  - 扰动边界: {config.epsilon:.4f}")
        print(f"  - 迭代次数: {config.num_iterations}")
        print(f"  - 目标查询数: {config.num_target_queries}")
        print(f"  - 攻击模式: {config.attack_mode}")
        print()


def demo_multiple_attacks():
    """演示多次攻击"""
    print("=" * 60)
    print("多次攻击演示")
    print("=" * 60)
    
    # 使用快速配置
    config = HubnessAttackConfig(
        num_iterations=20,
        device="cpu",
        random_seed=42
    )
    
    attacker = HubnessAttacker(config)
    
    # 不同颜色的图像
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    results = []
    
    print("执行多次攻击...")
    for i, color in enumerate(colors):
        print(f"攻击 {i+1}/5: {color} 图像")
        
        image = create_sample_image(color=color)
        result = attacker.attack(image)
        results.append((color, result))
        
        print(f"  - 成功: {'是' if result['success'] else '否'}")
        print(f"  - Hubness: {result['hubness_score']:.4f}")
        print(f"  - 时间: {result['attack_time']:.2f}秒")
        print()
    
    # 总结统计
    final_stats = attacker.get_attack_stats()
    print("最终统计:")
    print(f"  - 总攻击次数: {final_stats['total_attacks']}")
    print(f"  - 成功率: {final_stats['success_rate']:.2%}")
    print(f"  - 平均时间: {final_stats['average_time']:.3f}秒")
    print()
    
    return results


def demo_custom_queries():
    """演示自定义查询"""
    print("=" * 60)
    print("自定义查询演示")
    print("=" * 60)
    
    # 创建不同主题的查询集
    query_sets = {
        "动物": [
            "a cute puppy",
            "a sleeping cat",
            "a wild lion",
            "a flying bird"
        ],
        "交通工具": [
            "a red sports car",
            "a large truck",
            "a passenger airplane",
            "a racing motorcycle"
        ],
        "自然景观": [
            "a beautiful sunset",
            "a snowy mountain",
            "a tropical beach",
            "a dense forest"
        ]
    }
    
    config = HubnessAttackConfig(
        num_iterations=30,
        device="cpu"
    )
    
    attacker = HubnessAttacker(config)
    sample_image = create_sample_image(color='orange')
    
    for theme, queries in query_sets.items():
        print(f"测试主题: {theme}")
        print(f"查询: {queries}")
        
        result = attacker.attack(sample_image, queries)
        
        print(f"结果:")
        print(f"  - 成功: {'是' if result['success'] else '否'}")
        print(f"  - Hubness分数: {result['hubness_score']:.4f}")
        print(f"  - 攻击时间: {result['attack_time']:.2f}秒")
        print()


def main():
    """主函数"""
    print("Hubness Attack 演示程序")
    print("基于《Adversarial Hubness in Multi-Modal Retrieval》论文复现")
    print("作者: 张昕 (ZHANG XIN) - Duke University")
    print()
    
    try:
        # 检查依赖
        print("检查系统环境...")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print()
        
        # 演示1: 基本攻击
        result, original_image = demo_basic_attack()
        
        # 演示2: 预设配置
        demo_preset_configs()
        
        # 演示3: 多次攻击
        demo_multiple_attacks()
        
        # 演示4: 自定义查询
        demo_custom_queries()
        
        # 可视化结果（如果matplotlib可用）
        try:
            print("生成攻击结果可视化...")
            visualize_attack_result(
                original_image,
                result['adversarial_image'],
                result['perturbation'],
                result
            )
        except ImportError:
            print("matplotlib不可用，跳过可视化")
        except Exception as e:
            print(f"可视化失败: {e}")
        
        print("=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()