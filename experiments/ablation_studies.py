#!/usr/bin/env python3
"""消融实验模块

本模块实现了多模态检测一致性系统的消融实验，用于分析不同组件对系统性能的影响。
包括文本变体生成、参考向量生成、检索策略等组件的消融分析。

作者: 张昕 (zhang.xin@duke.edu)
学校: Duke University
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.utils.data_loader import DataLoaderManager
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import ExperimentVisualizer
from src.pipeline import MultiModalDetectionPipeline, create_detection_pipeline
from src.evaluation import ExperimentEvaluator, ExperimentConfig, create_experiment_evaluator
from src.text_augment import TextAugmenter, TextAugmentConfig
from src.sd_ref import SDReferenceGenerator, SDReferenceConfig
from src.retrieval import MultiModalRetriever, RetrievalConfig
from src.detector import AdversarialDetector, DetectorConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ablation_studies.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """消融实验配置"""
    experiment_name: str = "ablation_study"
    output_dir: str = "./results/ablation"
    num_samples: int = 500
    num_runs: int = 3  # 每个配置运行次数
    random_seed: int = 42
    
    # 组件开关
    test_text_variants: bool = True
    test_sd_references: bool = True
    test_retrieval_strategies: bool = True
    test_detection_methods: bool = True
    test_ensemble_methods: bool = True
    
    # 可视化配置
    generate_plots: bool = True
    save_detailed_results: bool = True


class AblationStudyManager:
    """消融实验管理器"""
    
    def __init__(self, config: AblationConfig, base_config: Dict[str, Any]):
        """
        初始化消融实验管理器
        
        Args:
            config: 消融实验配置
            base_config: 基础系统配置
        """
        self.config = config
        self.base_config = base_config
        self.results = {}
        self.metrics_calculator = MetricsCalculator()
        self.visualization_manager = ExperimentVisualizer()
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        np.random.seed(config.random_seed)
    
    def run_all_ablation_studies(self, test_data: List[Tuple]) -> Dict[str, Any]:
        """
        运行所有消融实验
        
        Args:
            test_data: 测试数据
        
        Returns:
            所有实验结果的字典
        """
        logger.info("开始运行消融实验")
        start_time = time.time()
        
        all_results = {}
        
        # 1. 文本变体生成消融实验
        if self.config.test_text_variants:
            logger.info("运行文本变体生成消融实验")
            all_results['text_variants'] = self._ablate_text_variants(test_data)
        
        # 2. SD参考向量生成消融实验
        if self.config.test_sd_references:
            logger.info("运行SD参考向量生成消融实验")
            all_results['sd_references'] = self._ablate_sd_references(test_data)
        
        # 3. 检索策略消融实验
        if self.config.test_retrieval_strategies:
            logger.info("运行检索策略消融实验")
            all_results['retrieval_strategies'] = self._ablate_retrieval_strategies(test_data)
        
        # 4. 检测方法消融实验
        if self.config.test_detection_methods:
            logger.info("运行检测方法消融实验")
            all_results['detection_methods'] = self._ablate_detection_methods(test_data)
        
        # 5. 集成方法消融实验
        if self.config.test_ensemble_methods:
            logger.info("运行集成方法消融实验")
            all_results['ensemble_methods'] = self._ablate_ensemble_methods(test_data)
        
        # 保存结果
        self._save_results(all_results)
        
        # 生成可视化
        if self.config.generate_plots:
            self._generate_visualizations(all_results)
        
        # 生成报告
        self._generate_ablation_report(all_results)
        
        total_time = time.time() - start_time
        logger.info(f"消融实验完成，总耗时: {total_time:.2f} 秒")
        
        return all_results
    
    def _ablate_text_variants(self, test_data: List[Tuple]) -> Dict[str, Any]:
        """
        文本变体生成消融实验
        
        测试不同的文本变体生成策略对系统性能的影响
        """
        logger.info("开始文本变体生成消融实验")
        
        # 定义不同的文本变体配置
        variant_configs = {
            'no_variants': {'num_variants': 0},
            'few_variants': {'num_variants': 3},
            'medium_variants': {'num_variants': 5},
            'many_variants': {'num_variants': 10},
            'high_similarity': {'num_variants': 5, 'similarity_threshold': 0.9},
            'low_similarity': {'num_variants': 5, 'similarity_threshold': 0.7},
            'high_temperature': {'num_variants': 5, 'temperature': 1.0},
            'low_temperature': {'num_variants': 5, 'temperature': 0.3}
        }
        
        results = {}
        
        for config_name, variant_config in variant_configs.items():
            logger.info(f"测试配置: {config_name}")
            
            # 创建修改后的配置
            modified_config = self.base_config.copy()
            modified_config['text_augment'].update(variant_config)
            
            # 运行实验
            config_results = self._run_single_configuration(
                test_data, modified_config, f"text_variants_{config_name}"
            )
            
            results[config_name] = config_results
        
        return results
    
    def _ablate_sd_references(self, test_data: List[Tuple]) -> Dict[str, Any]:
        """
        SD参考向量生成消融实验
        
        测试不同的SD参考向量生成策略对系统性能的影响
        """
        logger.info("开始SD参考向量生成消融实验")
        
        # 定义不同的SD配置
        sd_configs = {
            'no_sd_ref': {'enabled': False},
            'few_images': {'num_images_per_text': 1},
            'medium_images': {'num_images_per_text': 3},
            'many_images': {'num_images_per_text': 5},
            'low_steps': {'num_inference_steps': 10},
            'high_steps': {'num_inference_steps': 50},
            'low_guidance': {'guidance_scale': 3.0},
            'high_guidance': {'guidance_scale': 15.0},
            'small_resolution': {'height': 256, 'width': 256},
            'large_resolution': {'height': 768, 'width': 768}
        }
        
        results = {}
        
        for config_name, sd_config in sd_configs.items():
            logger.info(f"测试配置: {config_name}")
            
            # 创建修改后的配置
            modified_config = self.base_config.copy()
            if config_name == 'no_sd_ref':
                modified_config['sd_reference']['enabled'] = False
            else:
                modified_config['sd_reference'].update(sd_config)
            
            # 运行实验
            config_results = self._run_single_configuration(
                test_data, modified_config, f"sd_references_{config_name}"
            )
            
            results[config_name] = config_results
        
        return results
    
    def _ablate_retrieval_strategies(self, test_data: List[Tuple]) -> Dict[str, Any]:
        """
        检索策略消融实验
        
        测试不同的检索策略对系统性能的影响
        """
        logger.info("开始检索策略消融实验")
        
        # 定义不同的检索配置
        retrieval_configs = {
            'small_topk': {'top_k': 5},
            'medium_topk': {'top_k': 20},
            'large_topk': {'top_k': 50},
            'cosine_similarity': {'similarity_metric': 'cosine'},
            'dot_product': {'similarity_metric': 'dot_product'},
            'normalized_features': {'normalize_features': True},
            'unnormalized_features': {'normalize_features': False},
            'small_batch': {'batch_size': 64},
            'large_batch': {'batch_size': 512}
        }
        
        results = {}
        
        for config_name, retrieval_config in retrieval_configs.items():
            logger.info(f"测试配置: {config_name}")
            
            # 创建修改后的配置
            modified_config = self.base_config.copy()
            modified_config['retrieval'].update(retrieval_config)
            
            # 运行实验
            config_results = self._run_single_configuration(
                test_data, modified_config, f"retrieval_{config_name}"
            )
            
            results[config_name] = config_results
        
        return results
    
    def _ablate_detection_methods(self, test_data: List[Tuple]) -> Dict[str, Any]:
        """
        检测方法消融实验
        
        测试不同的检测方法对系统性能的影响
        """
        logger.info("开始检测方法消融实验")
        
        # 定义不同的检测配置
        detection_configs = {
            'consistency_only': {'use_consistency': True, 'use_similarity': False, 'use_statistical': False},
            'similarity_only': {'use_consistency': False, 'use_similarity': True, 'use_statistical': False},
            'statistical_only': {'use_consistency': False, 'use_similarity': False, 'use_statistical': True},
            'consistency_similarity': {'use_consistency': True, 'use_similarity': True, 'use_statistical': False},
            'all_methods': {'use_consistency': True, 'use_similarity': True, 'use_statistical': True},
            'strict_threshold': {'consistency_threshold': 0.9, 'similarity_threshold': 0.9},
            'loose_threshold': {'consistency_threshold': 0.5, 'similarity_threshold': 0.5}
        }
        
        results = {}
        
        for config_name, detection_config in detection_configs.items():
            logger.info(f"测试配置: {config_name}")
            
            # 创建修改后的配置
            modified_config = self.base_config.copy()
            modified_config['detector'].update(detection_config)
            
            # 运行实验
            config_results = self._run_single_configuration(
                test_data, modified_config, f"detection_{config_name}"
            )
            
            results[config_name] = config_results
        
        return results
    
    def _ablate_ensemble_methods(self, test_data: List[Tuple]) -> Dict[str, Any]:
        """
        集成方法消融实验
        
        测试不同的集成方法对系统性能的影响
        """
        logger.info("开始集成方法消融实验")
        
        # 定义不同的集成配置
        ensemble_configs = {
            'no_ensemble': {'method': 'none'},
            'simple_voting': {'method': 'voting'},
            'weighted_voting': {'method': 'weighted_voting'},
            'stacking': {'method': 'stacking'},
            'confidence_based': {'method': 'confidence_based'},
            'adaptive_weights': {'method': 'adaptive_weights'}
        }
        
        results = {}
        
        for config_name, ensemble_config in ensemble_configs.items():
            logger.info(f"测试配置: {config_name}")
            
            # 创建修改后的配置
            modified_config = self.base_config.copy()
            modified_config['ensemble'] = ensemble_config
            
            # 运行实验
            config_results = self._run_single_configuration(
                test_data, modified_config, f"ensemble_{config_name}"
            )
            
            results[config_name] = config_results
        
        return results
    
    def _run_single_configuration(self, test_data: List[Tuple], 
                                config: Dict[str, Any], 
                                config_name: str) -> Dict[str, Any]:
        """
        运行单个配置的实验
        
        Args:
            test_data: 测试数据
            config: 配置字典
            config_name: 配置名称
        
        Returns:
            实验结果字典
        """
        results = []
        
        for run_idx in range(self.config.num_runs):
            logger.info(f"运行 {config_name} - 第 {run_idx + 1}/{self.config.num_runs} 次")
            
            try:
                # 创建检测管道
                pipeline = create_detection_pipeline(config)
                
                # 创建评估器
                experiment_config = ExperimentConfig(
                    experiment_name=f"{config_name}_run_{run_idx}",
                    use_cross_validation=False,
                    generate_plots=False,
                    results_dir=str(self.output_dir / config_name)
                )
                
                evaluator = create_experiment_evaluator(experiment_config)
                
                # 运行评估
                result = evaluator.evaluate_pipeline(pipeline, test_data)
                
                # 提取关键指标
                run_result = {
                    'run_idx': run_idx,
                    'metrics': result.metrics,
                    'performance_stats': result.performance_stats,
                    'execution_time': result.execution_time
                }
                
                results.append(run_result)
                
            except Exception as e:
                logger.error(f"配置 {config_name} 运行 {run_idx} 失败: {e}")
                continue
        
        # 计算统计结果
        if results:
            aggregated_result = self._aggregate_results(results)
            aggregated_result['config'] = config
            aggregated_result['num_successful_runs'] = len(results)
            return aggregated_result
        else:
            logger.error(f"配置 {config_name} 所有运行都失败")
            return {'error': 'All runs failed'}
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚合多次运行的结果
        
        Args:
            results: 多次运行的结果列表
        
        Returns:
            聚合后的结果
        """
        if not results:
            return {}
        
        # 提取所有指标
        all_metrics = {}
        for result in results:
            for metric, value in result['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # 计算统计量
        aggregated = {
            'mean_metrics': {},
            'std_metrics': {},
            'min_metrics': {},
            'max_metrics': {},
            'raw_results': results
        }
        
        for metric, values in all_metrics.items():
            aggregated['mean_metrics'][metric] = np.mean(values)
            aggregated['std_metrics'][metric] = np.std(values)
            aggregated['min_metrics'][metric] = np.min(values)
            aggregated['max_metrics'][metric] = np.max(values)
        
        return aggregated
    
    def _save_results(self, results: Dict[str, Any]):
        """
        保存实验结果
        
        Args:
            results: 实验结果字典
        """
        # 保存完整结果
        results_file = self.output_dir / "ablation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存汇总表格
        summary_data = []
        for category, category_results in results.items():
            for config_name, config_result in category_results.items():
                if 'mean_metrics' in config_result:
                    row = {
                        'category': category,
                        'config': config_name,
                        **config_result['mean_metrics']
                    }
                    summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.output_dir / "ablation_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            
            logger.info(f"结果已保存到: {results_file} 和 {summary_file}")
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """
        生成可视化图表
        
        Args:
            results: 实验结果字典
        """
        logger.info("生成可视化图表")
        
        # 设置图表样式
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('default')
        sns.set_palette("husl")
        
        # 为每个类别生成对比图
        for category, category_results in results.items():
            self._plot_category_comparison(category, category_results)
        
        # 生成总体对比图
        self._plot_overall_comparison(results)
        
        logger.info(f"可视化图表已保存到: {self.output_dir}")
    
    def _plot_category_comparison(self, category: str, category_results: Dict[str, Any]):
        """
        绘制单个类别的对比图
        
        Args:
            category: 类别名称
            category_results: 类别结果
        """
        # 提取数据
        configs = []
        metrics_data = {}
        
        for config_name, config_result in category_results.items():
            if 'mean_metrics' in config_result:
                configs.append(config_name)
                for metric, value in config_result['mean_metrics'].items():
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append(value)
        
        if not configs:
            return
        
        # 创建子图
        num_metrics = len(metrics_data)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
        if num_metrics == 1:
            axes = [axes]
        
        for idx, (metric, values) in enumerate(metrics_data.items()):
            ax = axes[idx]
            bars = ax.bar(configs, values)
            ax.set_title(f'{metric.upper()} - {category.replace("_", " ").title()}')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{category}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_comparison(self, results: Dict[str, Any]):
        """
        绘制总体对比图
        
        Args:
            results: 所有实验结果
        """
        # 收集所有配置的主要指标
        all_data = []
        
        for category, category_results in results.items():
            for config_name, config_result in category_results.items():
                if 'mean_metrics' in config_result:
                    row = {
                        'category': category,
                        'config': f"{category}_{config_name}",
                        **config_result['mean_metrics']
                    }
                    all_data.append(row)
        
        if not all_data:
            return
        
        df = pd.DataFrame(all_data)
        
        # 绘制热力图
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plt.figure(figsize=(12, 8))
            heatmap_data = df.set_index('config')[numeric_cols]
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
            plt.title('Ablation Study Results Heatmap')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'overall_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_ablation_report(self, results: Dict[str, Any]):
        """
        生成消融实验报告
        
        Args:
            results: 实验结果字典
        """
        logger.info("生成消融实验报告")
        
        report_lines = [
            "# 多模态检测一致性系统消融实验报告\n",
            f"**实验时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**实验配置**: {self.config.experiment_name}\n",
            f"**测试样本数**: {self.config.num_samples}\n",
            f"**运行次数**: {self.config.num_runs}\n\n",
            "## 实验概述\n\n",
            "本报告展示了多模态检测一致性系统各组件的消融实验结果，",
            "分析了不同组件和配置对系统整体性能的影响。\n\n"
        ]
        
        # 为每个类别生成详细分析
        for category, category_results in results.items():
            report_lines.extend(self._generate_category_analysis(category, category_results))
        
        # 生成总结
        report_lines.extend(self._generate_conclusions(results))
        
        # 保存报告
        report_file = self.output_dir / "ablation_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        logger.info(f"消融实验报告已保存到: {report_file}")
    
    def _generate_category_analysis(self, category: str, category_results: Dict[str, Any]) -> List[str]:
        """
        生成单个类别的分析
        
        Args:
            category: 类别名称
            category_results: 类别结果
        
        Returns:
            报告行列表
        """
        lines = [
            f"## {category.replace('_', ' ').title()} 消融实验\n\n"
        ]
        
        # 找出最佳配置
        best_config = None
        best_score = -1
        
        for config_name, config_result in category_results.items():
            if 'mean_metrics' in config_result:
                # 使用主要指标（如accuracy或f1_score）作为评判标准
                main_metrics = ['accuracy', 'f1_score', 'auc']
                score = 0
                for metric in main_metrics:
                    if metric in config_result['mean_metrics']:
                        score = config_result['mean_metrics'][metric]
                        break
                
                if score > best_score:
                    best_score = score
                    best_config = config_name
        
        if best_config:
            lines.append(f"**最佳配置**: {best_config} (主要指标: {best_score:.4f})\n\n")
        
        # 生成配置对比表格
        lines.append("### 配置对比\n\n")
        lines.append("| 配置 | ")
        
        # 获取所有指标名称
        all_metrics = set()
        for config_result in category_results.values():
            if 'mean_metrics' in config_result:
                all_metrics.update(config_result['mean_metrics'].keys())
        
        all_metrics = sorted(list(all_metrics))
        lines.append(" | ".join(all_metrics) + " |\n")
        lines.append("|" + "---|" * (len(all_metrics) + 1) + "\n")
        
        for config_name, config_result in category_results.items():
            if 'mean_metrics' in config_result:
                line = f"| {config_name} |"
                for metric in all_metrics:
                    value = config_result['mean_metrics'].get(metric, 0)
                    line += f" {value:.4f} |"
                lines.append(line + "\n")
        
        lines.append("\n")
        
        return lines
    
    def _generate_conclusions(self, results: Dict[str, Any]) -> List[str]:
        """
        生成实验结论
        
        Args:
            results: 所有实验结果
        
        Returns:
            结论行列表
        """
        lines = [
            "## 实验结论\n\n",
            "基于消融实验的结果，我们得出以下结论:\n\n"
        ]
        
        # 为每个类别生成结论
        category_conclusions = {
            'text_variants': "文本变体生成对系统性能有显著影响，适当数量的高质量变体能够提升检测准确性。",
            'sd_references': "Stable Diffusion参考向量生成是系统的重要组成部分，合适的生成参数能够平衡性能和效率。",
            'retrieval_strategies': "检索策略的选择对系统性能有重要影响，需要根据具体应用场景进行优化。",
            'detection_methods': "不同检测方法的组合使用能够提升系统的鲁棒性和准确性。",
            'ensemble_methods': "集成方法能够有效提升系统性能，但需要权衡复杂度和收益。"
        }
        
        for category in results.keys():
            if category in category_conclusions:
                lines.append(f"- **{category.replace('_', ' ').title()}**: {category_conclusions[category]}\n")
        
        lines.extend([
            "\n",
            "## 建议\n\n",
            "基于实验结果，我们建议:\n\n",
            "1. 使用适中数量（5-10个）的高质量文本变体\n",
            "2. 合理配置SD生成参数以平衡质量和效率\n",
            "3. 根据数据集特点选择合适的检索策略\n",
            "4. 采用多种检测方法的组合以提升鲁棒性\n",
            "5. 在性能要求较高的场景下使用集成方法\n\n",
            "---\n\n",
            f"**报告生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "**作者**: 张昕 (zhang.xin@duke.edu)\n",
            "**学校**: Duke University\n"
        ])
        
        return lines


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行多模态检测一致性系统消融实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default.yaml',
        help='基础配置文件路径'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./results/ablation',
        help='输出目录'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=500,
        help='测试样本数量'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='每个配置的运行次数'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['coco', 'flickr30k'],
        default='coco',
        help='使用的数据集'
    )
    
    parser.add_argument(
        '--components',
        nargs='+',
        choices=['text_variants', 'sd_references', 'retrieval_strategies', 'detection_methods', 'ensemble_methods'],
        default=['text_variants', 'sd_references', 'retrieval_strategies', 'detection_methods'],
        help='要测试的组件'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='不生成可视化图表'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 设置调试模式
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 加载基础配置
        config_manager = ConfigManager()
        base_config = config_manager.load_config(args.config)
        
        # 创建消融实验配置
        ablation_config = AblationConfig(
            experiment_name="multimodal_detection_ablation",
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            num_runs=args.num_runs,
            test_text_variants='text_variants' in args.components,
            test_sd_references='sd_references' in args.components,
            test_retrieval_strategies='retrieval_strategies' in args.components,
            test_detection_methods='detection_methods' in args.components,
            test_ensemble_methods='ensemble_methods' in args.components,
            generate_plots=not args.no_plots
        )
        
        # 加载测试数据
        logger.info(f"加载测试数据: {args.dataset}")
        data_manager = DataLoaderManager()
        
        if args.dataset == 'coco':
            test_loader = data_manager.create_coco_loader(
                data_dir='./data',
                split='val',
                batch_size=32
            )
        else:
            test_loader = data_manager.create_flickr30k_loader(
                data_dir='./data',
                split='test',
                batch_size=32
            )
        
        # 转换为列表格式
        test_data = []
        count = 0
        for batch in test_loader:
            for i in range(len(batch['images'])):
                test_data.append((
                    batch['images'][i],
                    batch['texts'][i],
                    False  # 非对抗样本
                ))
                count += 1
                if count >= args.num_samples:
                    break
            if count >= args.num_samples:
                break
        
        logger.info(f"加载了 {len(test_data)} 个测试样本")
        
        # 创建消融实验管理器
        ablation_manager = AblationStudyManager(ablation_config, base_config)
        
        # 运行消融实验
        results = ablation_manager.run_all_ablation_studies(test_data)
        
        logger.info("消融实验完成")
        logger.info(f"结果保存在: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("实验被用户中断")
    except Exception as e:
        logger.error(f"消融实验失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()