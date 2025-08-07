#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hubness攻击基准测试脚本

基于《Adversarial Hubness in Multi-Modal Retrieval》论文的性能基准测试

作者: 张昕 (ZHANG XIN)
学校: Duke University
邮箱: zhang.xin@duke.edu
创建日期: 2025-01-05
"""

import sys
import os
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from attacks.hubness_attack import HubnessAttacker, HubnessAttackConfig, HubnessAttackPresets

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HubnessBenchmark:
    """Hubness攻击基准测试类"""
    
    def __init__(self, output_dir: str = "results/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 基准测试结果
        self.results = {
            'configs': {},
            'performance': {},
            'effectiveness': {},
            'memory_usage': {},
            'timing': {}
        }
    
    def create_test_images(self, num_images: int = 10) -> List[Image.Image]:
        """创建测试图像"""
        images = []
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray']
        
        for i in range(num_images):
            color = colors[i % len(colors)]
            image = Image.new('RGB', (224, 224), color=color)
            images.append(image)
        
        return images
    
    def create_test_queries(self) -> List[str]:
        """创建测试查询"""
        return [
            "a photo of a person",
            "a picture of an animal",
            "a view of a building",
            "a scene of nature",
            "a close-up of an object",
            "a landscape image",
            "a portrait photo",
            "an indoor scene",
            "a street view",
            "a food image"
        ]
    
    def benchmark_config(self, config_name: str, config: HubnessAttackConfig, 
                        test_images: List[Image.Image], test_queries: List[str]) -> Dict[str, Any]:
        """基准测试单个配置"""
        logger.info(f"开始基准测试配置: {config_name}")
        
        # 创建攻击器
        attacker = HubnessAttacker(config)
        
        # 性能指标
        attack_times = []
        success_count = 0
        memory_usage = []
        hubness_scores = []
        
        # 测试每个图像
        for i, image in enumerate(test_images):
            logger.info(f"测试图像 {i+1}/{len(test_images)}")
            
            # 记录内存使用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            
            # 执行攻击
            start_time = time.time()
            result = attacker.attack(image, test_queries)
            attack_time = time.time() - start_time
            
            # 记录结果
            attack_times.append(attack_time)
            if result['success']:
                success_count += 1
            
            if result['hubness_score'] is not None:
                hubness_scores.append(float(result['hubness_score']))
            
            # 记录内存使用
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_usage.append(memory_after - memory_before)
        
        # 计算统计信息
        stats = attacker.get_attack_stats()
        
        return {
            'config_name': config_name,
            'success_rate': success_count / len(test_images),
            'avg_attack_time': np.mean(attack_times),
            'std_attack_time': np.std(attack_times),
            'min_attack_time': np.min(attack_times),
            'max_attack_time': np.max(attack_times),
            'avg_hubness_score': np.mean(hubness_scores) if hubness_scores else 0.0,
            'std_hubness_score': np.std(hubness_scores) if hubness_scores else 0.0,
            'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0.0,
            'total_attacks': stats['total_attacks'],
            'attack_times': attack_times,
            'hubness_scores': hubness_scores,
            'memory_usage': memory_usage
        }
    
    def run_comprehensive_benchmark(self):
        """运行综合基准测试"""
        logger.info("开始综合基准测试")
        
        # 创建测试数据
        test_images = self.create_test_images(10)
        test_queries = self.create_test_queries()
        
        # 测试配置
        configs = {
            'weak_attack': HubnessAttackPresets.weak_attack(),
            'strong_attack': HubnessAttackPresets.strong_attack(),
            'targeted_attack': HubnessAttackPresets.targeted_attack(),
            'paper_config': HubnessAttackPresets.paper_config(),
            'fast_test': HubnessAttackConfig(
                epsilon=4.0/255.0,
                num_iterations=5,
                step_size=0.01,
                k_neighbors=5,
                num_target_queries=10,
                device="auto"
            )
        }
        
        # 运行基准测试
        for config_name, config in configs.items():
            try:
                result = self.benchmark_config(config_name, config, test_images, test_queries)
                self.results['performance'][config_name] = result
                self.results['configs'][config_name] = {
                    'epsilon': config.epsilon,
                    'num_iterations': config.num_iterations,
                    'step_size': config.step_size,
                    'k_neighbors': config.k_neighbors,
                    'num_target_queries': config.num_target_queries,
                    'attack_mode': config.attack_mode
                }
            except Exception as e:
                logger.error(f"配置 {config_name} 测试失败: {e}")
                continue
    
    def generate_performance_report(self):
        """生成性能报告"""
        logger.info("生成性能报告")
        
        report = []
        report.append("# Hubness攻击基准测试报告\n")
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"测试环境: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
        report.append("\n## 配置对比\n")
        
        # 创建对比表格
        report.append("| 配置 | 成功率 | 平均时间(s) | 平均Hubness | 内存使用(MB) |\n")
        report.append("|------|--------|-------------|-------------|--------------|\n")
        
        for config_name, result in self.results['performance'].items():
            memory_mb = result['avg_memory_usage'] / (1024 * 1024) if result['avg_memory_usage'] > 0 else 0
            report.append(
                f"| {config_name} | {result['success_rate']:.3f} | "
                f"{result['avg_attack_time']:.3f} | {result['avg_hubness_score']:.3f} | "
                f"{memory_mb:.1f} |\n"
            )
        
        # 详细分析
        report.append("\n## 详细分析\n")
        
        for config_name, result in self.results['performance'].items():
            report.append(f"\n### {config_name}\n")
            report.append(f"- 成功率: {result['success_rate']:.3f}\n")
            report.append(f"- 平均攻击时间: {result['avg_attack_time']:.3f} ± {result['std_attack_time']:.3f} 秒\n")
            report.append(f"- 时间范围: {result['min_attack_time']:.3f} - {result['max_attack_time']:.3f} 秒\n")
            report.append(f"- 平均Hubness分数: {result['avg_hubness_score']:.3f} ± {result['std_hubness_score']:.3f}\n")
            
            if result['avg_memory_usage'] > 0:
                memory_mb = result['avg_memory_usage'] / (1024 * 1024)
                report.append(f"- 平均内存使用: {memory_mb:.1f} MB\n")
        
        # 保存报告
        report_path = self.output_dir / "benchmark_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        logger.info(f"性能报告已保存到: {report_path}")
    
    def generate_visualizations(self):
        """生成可视化图表"""
        logger.info("生成可视化图表")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 成功率对比
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 成功率
        config_names = list(self.results['performance'].keys())
        success_rates = [self.results['performance'][name]['success_rate'] for name in config_names]
        
        axes[0, 0].bar(config_names, success_rates)
        axes[0, 0].set_title('攻击成功率对比')
        axes[0, 0].set_ylabel('成功率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 攻击时间
        attack_times = [self.results['performance'][name]['avg_attack_time'] for name in config_names]
        time_stds = [self.results['performance'][name]['std_attack_time'] for name in config_names]
        
        axes[0, 1].bar(config_names, attack_times, yerr=time_stds, capsize=5)
        axes[0, 1].set_title('平均攻击时间对比')
        axes[0, 1].set_ylabel('时间 (秒)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Hubness分数
        hubness_scores = [self.results['performance'][name]['avg_hubness_score'] for name in config_names]
        hubness_stds = [self.results['performance'][name]['std_hubness_score'] for name in config_names]
        
        axes[1, 0].bar(config_names, hubness_scores, yerr=hubness_stds, capsize=5)
        axes[1, 0].set_title('平均Hubness分数对比')
        axes[1, 0].set_ylabel('Hubness分数')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 内存使用
        memory_usage = [self.results['performance'][name]['avg_memory_usage'] / (1024*1024) 
                       for name in config_names if self.results['performance'][name]['avg_memory_usage'] > 0]
        memory_names = [name for name in config_names 
                       if self.results['performance'][name]['avg_memory_usage'] > 0]
        
        if memory_usage:
            axes[1, 1].bar(memory_names, memory_usage)
            axes[1, 1].set_title('平均内存使用对比')
            axes[1, 1].set_ylabel('内存 (MB)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, '无内存使用数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('内存使用')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "benchmark_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 时间分布图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, config_name in enumerate(config_names[:6]):
            if i < len(config_names):
                times = self.results['performance'][config_name]['attack_times']
                axes[i].hist(times, bins=10, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{config_name} - 攻击时间分布')
                axes[i].set_xlabel('时间 (秒)')
                axes[i].set_ylabel('频次')
        
        # 隐藏多余的子图
        for i in range(len(config_names), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化图表已保存到: {self.output_dir}")
    
    def save_results(self):
        """保存基准测试结果"""
        results_path = self.output_dir / "benchmark_results.json"
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        serializable_results[key][subkey] = {}
                        for subsubkey, subsubvalue in subvalue.items():
                            if isinstance(subsubvalue, (np.ndarray, list)):
                                serializable_results[key][subkey][subsubkey] = list(subsubvalue)
                            else:
                                serializable_results[key][subkey][subsubkey] = subsubvalue
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"基准测试结果已保存到: {results_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Hubness攻击基准测试")
    parser.add_argument("--output-dir", default="results/benchmarks", help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建基准测试实例
    benchmark = HubnessBenchmark(args.output_dir)
    
    try:
        # 运行基准测试
        benchmark.run_comprehensive_benchmark()
        
        # 生成报告和可视化
        benchmark.generate_performance_report()
        benchmark.generate_visualizations()
        benchmark.save_results()
        
        logger.info("基准测试完成！")
        
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        raise


if __name__ == "__main__":
    main()