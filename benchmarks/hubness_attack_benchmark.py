#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hubness Attack 基准测试脚本

用于验证《Adversarial Hubness in Multi-Modal Retrieval》论文复现实现的性能和正确性

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
import json
import psutil
import gc
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.attacks.hubness_attack import (
    HubnessAttacker, 
    HubnessAttackConfig, 
    HubnessAttackPresets
)


class HubnessAttackBenchmark:
    """Hubness攻击基准测试类"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform
        }
    
    def _create_test_data(self, batch_size: int = 1) -> Tuple[List[Image.Image], List[str]]:
        """创建测试数据"""
        images = []
        for i in range(batch_size):
            # 创建不同颜色的测试图像
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan']
            color = colors[i % len(colors)]
            image = Image.new('RGB', (224, 224), color=color)
            images.append(image)
        
        queries = [
            "a photo of a dog",
            "a picture of a cat", 
            "an image of a car",
            "a photo of a person",
            "a picture of a building",
            "an image of food",
            "a landscape photo",
            "a close-up shot",
            "a street scene",
            "a nature photograph"
        ]
        
        return images, queries
    
    def _measure_memory_usage(self) -> Dict[str, float]:
        """测量内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        result = {
            'rss_mb': memory_info.rss / (1024**2),
            'vms_mb': memory_info.vms / (1024**2)
        }
        
        if torch.cuda.is_available():
            result['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            result['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
        
        return result
    
    def benchmark_config_performance(self) -> Dict[str, Any]:
        """基准测试不同配置的性能"""
        print("运行配置性能基准测试...")
        
        configs = {
            'weak_attack': HubnessAttackPresets.weak_attack(),
            'strong_attack': HubnessAttackPresets.strong_attack(), 
            'targeted_attack': HubnessAttackPresets.targeted_attack(),
            'paper_config': HubnessAttackPresets.paper_config()
        }
        
        # 调整配置以适合基准测试
        for config in configs.values():
            config.device = "cpu"  # 使用CPU确保一致性
            config.num_iterations = min(config.num_iterations, 50)  # 限制迭代次数
        
        results = {}
        test_images, test_queries = self._create_test_data(5)
        
        for config_name, config in configs.items():
            print(f"  测试配置: {config_name}")
            
            config_results = {
                'config_params': {
                    'epsilon': config.epsilon,
                    'num_iterations': config.num_iterations,
                    'step_size': config.step_size,
                    'attack_mode': config.attack_mode
                },
                'performance_metrics': []
            }
            
            attacker = HubnessAttacker(config)
            
            # 多次运行以获得稳定的性能指标
            for run_idx in range(3):
                gc.collect()  # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                memory_before = self._measure_memory_usage()
                start_time = time.time()
                
                # 执行攻击
                image = test_images[run_idx % len(test_images)]
                result = attacker.attack(image, test_queries[:5])
                
                end_time = time.time()
                memory_after = self._measure_memory_usage()
                
                # 计算内存增长
                memory_growth = {
                    key: memory_after[key] - memory_before[key] 
                    for key in memory_before.keys()
                }
                
                config_results['performance_metrics'].append({
                    'run_index': run_idx,
                    'attack_time': end_time - start_time,
                    'success': result['success'],
                    'hubness_score': result['hubness_score'],
                    'iterations': result['iterations'],
                    'memory_growth_mb': memory_growth
                })
            
            # 计算平均性能指标
            metrics = config_results['performance_metrics']
            config_results['average_metrics'] = {
                'avg_attack_time': np.mean([m['attack_time'] for m in metrics]),
                'std_attack_time': np.std([m['attack_time'] for m in metrics]),
                'success_rate': np.mean([m['success'] for m in metrics]),
                'avg_hubness_score': np.mean([m['hubness_score'] for m in metrics]),
                'avg_iterations': np.mean([m['iterations'] for m in metrics])
            }
            
            results[config_name] = config_results
        
        return results
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """基准测试可扩展性"""
        print("运行可扩展性基准测试...")
        
        config = HubnessAttackConfig(
            epsilon=8.0 / 255.0,
            num_iterations=20,
            device="cpu",
            random_seed=42
        )
        
        attacker = HubnessAttacker(config)
        
        # 测试不同的查询数量
        query_counts = [1, 5, 10, 20, 50]
        results = {'query_scaling': []}
        
        test_images, all_queries = self._create_test_data(1)
        test_image = test_images[0]
        
        for query_count in query_counts:
            print(f"  测试查询数量: {query_count}")
            
            queries = all_queries[:query_count]
            
            # 多次运行取平均
            times = []
            memory_usages = []
            
            for _ in range(3):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                memory_before = self._measure_memory_usage()
                start_time = time.time()
                
                result = attacker.attack(test_image, queries)
                
                end_time = time.time()
                memory_after = self._measure_memory_usage()
                
                times.append(end_time - start_time)
                memory_usages.append(memory_after['rss_mb'] - memory_before['rss_mb'])
            
            results['query_scaling'].append({
                'query_count': query_count,
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_memory_mb': np.mean(memory_usages),
                'time_per_query': np.mean(times) / query_count
            })
        
        return results
    
    def benchmark_accuracy(self) -> Dict[str, Any]:
        """基准测试准确性和一致性"""
        print("运行准确性基准测试...")
        
        config = HubnessAttackConfig(
            epsilon=16.0 / 255.0,
            num_iterations=100,
            device="cpu",
            random_seed=42
        )
        
        results = {
            'reproducibility': [],
            'perturbation_bounds': [],
            'hubness_consistency': []
        }
        
        test_images, test_queries = self._create_test_data(3)
        
        # 测试可重现性
        print("  测试可重现性...")
        for seed in [42, 123, 456]:
            config.random_seed = seed
            attacker = HubnessAttacker(config)
            
            # 同一图像多次攻击应该产生相同结果
            image = test_images[0]
            results_for_seed = []
            
            for _ in range(3):
                result = attacker.attack(image, test_queries[:5])
                results_for_seed.append({
                    'hubness_score': result['hubness_score'],
                    'success': result['success'],
                    'iterations': result['iterations']
                })
            
            # 检查一致性
            hubness_scores = [r['hubness_score'] for r in results_for_seed]
            hubness_std = np.std(hubness_scores)
            
            results['reproducibility'].append({
                'seed': seed,
                'hubness_std': hubness_std,
                'consistent': hubness_std < 1e-6,
                'results': results_for_seed
            })
        
        # 测试扰动边界
        print("  测试扰动边界...")
        config.random_seed = 42
        attacker = HubnessAttacker(config)
        
        for i, image in enumerate(test_images):
            result = attacker.attack(image, test_queries[:5])
            
            if result['perturbation'] is not None:
                perturbation = result['perturbation']
                max_perturbation = torch.max(torch.abs(perturbation)).item()
                
                results['perturbation_bounds'].append({
                    'image_index': i,
                    'max_perturbation': max_perturbation,
                    'epsilon': config.epsilon,
                    'within_bounds': max_perturbation <= config.epsilon + 1e-6
                })
        
        return results
    
    def benchmark_hubness_computation(self) -> Dict[str, Any]:
        """基准测试Hubness计算的正确性"""
        print("运行Hubness计算基准测试...")
        
        config = HubnessAttackConfig(device="cpu")
        attacker = HubnessAttacker(config)
        
        results = {
            'computation_accuracy': [],
            'performance': []
        }
        
        # 测试不同规模的特征
        test_sizes = [(10, 5, 128), (50, 20, 256), (100, 50, 512)]
        
        for num_images, num_queries, feature_dim in test_sizes:
            print(f"  测试规模: {num_images}图像, {num_queries}查询, {feature_dim}维特征")
            
            # 生成测试特征
            torch.manual_seed(42)
            image_features = torch.randn(num_images, feature_dim)
            text_features = torch.randn(num_queries, feature_dim)
            
            # 归一化
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            
            # 测试计算时间
            times = []
            for _ in range(5):
                start_time = time.time()
                hubness_scores = attacker.compute_hubness(image_features, text_features)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # 验证计算正确性
            # Hubness分数应该在合理范围内
            valid_scores = torch.all(hubness_scores >= 0) and torch.all(hubness_scores <= num_queries)
            
            results['computation_accuracy'].append({
                'size': (num_images, num_queries, feature_dim),
                'valid_scores': valid_scores.item(),
                'score_range': (hubness_scores.min().item(), hubness_scores.max().item()),
                'score_mean': hubness_scores.mean().item(),
                'score_std': hubness_scores.std().item()
            })
            
            results['performance'].append({
                'size': (num_images, num_queries, feature_dim),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'throughput': num_images / np.mean(times)  # 图像/秒
            })
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """运行所有基准测试"""
        print("开始运行Hubness攻击基准测试套件")
        print("=" * 60)
        
        start_time = time.time()
        
        # 运行各项基准测试
        self.results['benchmarks']['config_performance'] = self.benchmark_config_performance()
        self.results['benchmarks']['scalability'] = self.benchmark_scalability()
        self.results['benchmarks']['accuracy'] = self.benchmark_accuracy()
        self.results['benchmarks']['hubness_computation'] = self.benchmark_hubness_computation()
        
        total_time = time.time() - start_time
        self.results['benchmark_summary'] = {
            'total_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'completed'
        }
        
        print(f"\n基准测试完成！总耗时: {total_time:.2f}秒")
        
        return self.results
    
    def save_results(self, filename: str = None) -> str:
        """保存基准测试结果"""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"hubness_attack_benchmark_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy(self.results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        print(f"基准测试结果已保存到: {filepath}")
        return str(filepath)
    
    def generate_report(self) -> str:
        """生成基准测试报告"""
        report = []
        report.append("Hubness Attack 基准测试报告")
        report.append("=" * 50)
        report.append("")
        
        # 系统信息
        report.append("系统信息:")
        sys_info = self.results['system_info']
        report.append(f"  - Python版本: {sys_info['python_version'].split()[0]}")
        report.append(f"  - PyTorch版本: {sys_info['pytorch_version']}")
        report.append(f"  - CUDA可用: {'是' if sys_info['cuda_available'] else '否'}")
        report.append(f"  - CPU核心数: {sys_info['cpu_count']}")
        report.append(f"  - 内存总量: {sys_info['memory_total_gb']:.1f} GB")
        report.append("")
        
        # 配置性能
        if 'config_performance' in self.results['benchmarks']:
            report.append("配置性能对比:")
            config_perf = self.results['benchmarks']['config_performance']
            
            for config_name, data in config_perf.items():
                avg_metrics = data['average_metrics']
                report.append(f"  {config_name}:")
                report.append(f"    - 平均攻击时间: {avg_metrics['avg_attack_time']:.3f}s")
                report.append(f"    - 成功率: {avg_metrics['success_rate']:.2%}")
                report.append(f"    - 平均Hubness分数: {avg_metrics['avg_hubness_score']:.4f}")
            report.append("")
        
        # 可扩展性
        if 'scalability' in self.results['benchmarks']:
            report.append("可扩展性测试:")
            scalability = self.results['benchmarks']['scalability']
            
            for data in scalability['query_scaling']:
                report.append(f"  查询数量 {data['query_count']}:")
                report.append(f"    - 平均时间: {data['avg_time']:.3f}s")
                report.append(f"    - 每查询时间: {data['time_per_query']:.4f}s")
            report.append("")
        
        # 准确性
        if 'accuracy' in self.results['benchmarks']:
            report.append("准确性测试:")
            accuracy = self.results['benchmarks']['accuracy']
            
            # 可重现性
            reproducible_count = sum(1 for r in accuracy['reproducibility'] if r['consistent'])
            total_tests = len(accuracy['reproducibility'])
            report.append(f"  - 可重现性: {reproducible_count}/{total_tests} 通过")
            
            # 扰动边界
            within_bounds = sum(1 for r in accuracy['perturbation_bounds'] if r['within_bounds'])
            total_bounds = len(accuracy['perturbation_bounds'])
            report.append(f"  - 扰动边界: {within_bounds}/{total_bounds} 在限制内")
            report.append("")
        
        # 总结
        summary = self.results['benchmark_summary']
        report.append(f"基准测试总结:")
        report.append(f"  - 完成时间: {summary['timestamp']}")
        report.append(f"  - 总耗时: {summary['total_time']:.2f}秒")
        report.append(f"  - 状态: {summary['status']}")
        
        return "\n".join(report)


def main():
    """主函数"""
    print("Hubness Attack 基准测试")
    print("基于《Adversarial Hubness in Multi-Modal Retrieval》论文复现")
    print("作者: 张昕 (ZHANG XIN) - Duke University")
    print()
    
    try:
        # 创建基准测试实例
        benchmark = HubnessAttackBenchmark()
        
        # 运行所有基准测试
        results = benchmark.run_all_benchmarks()
        
        # 保存结果
        result_file = benchmark.save_results()
        
        # 生成并显示报告
        report = benchmark.generate_report()
        print("\n" + report)
        
        # 保存报告
        report_file = benchmark.output_dir / "benchmark_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n详细报告已保存到: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"基准测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 基准测试成功完成！")
    else:
        print("\n❌ 基准测试执行失败！")
        sys.exit(1)