#!/usr/bin/env python3
"""消融实验运行器

该脚本用于运行消融实验，分析不同防御组件的贡献。
作者: 张昕 (zhang.xin@duke.edu)
机构: Duke University
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dataclasses import dataclass, asdict

import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.defenses import (
    MultiModalDefenseDetector,
    DetectionConfig,
    ConsistencyChecker,
    TextVariantGenerator,
    GenerativeReferenceGenerator,
    RetrievalReferenceGenerator
)
from experiments.utils.logger import setup_logger
from experiments.utils.config_loader import load_config
from experiments.utils.metrics import (
    compute_detection_metrics,
    DetectionMetrics,
    MetricsAggregator
)
from experiments.utils.visualization import VisualizationManager
from experiments.utils.seed import set_random_seed
from experiments.datasets import get_dataset_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AblationConfig:
    """消融实验配置"""
    components: List[str]  # 要测试的组件列表
    baseline_components: List[str]  # 基线组件
    num_runs: int = 5  # 每个配置的运行次数
    save_individual_results: bool = True  # 是否保存单次运行结果
    statistical_test: str = 'paired_t_test'  # 统计检验方法
    significance_level: float = 0.05  # 显著性水平

class AblationRunner:
    """消融实验运行器"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        """初始化消融实验运行器
        
        Args:
            config: 实验配置
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        seed = config.get('debug', {}).get('seed', 42)
        set_random_seed(seed)
        
        # 设备配置
        self.device = config.get('hardware', {}).get('device', 'cuda')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化组件
        self._init_components()
        
        # 可视化管理器
        self.visualizer = VisualizationManager(self.output_dir / 'figures')
        
        # 指标聚合器
        self.metrics_aggregator = MetricsAggregator()
        
        logger.info(f"消融实验运行器初始化完成，输出目录: {self.output_dir}")
    
    def _init_components(self):
        """初始化防御组件"""
        defense_config = self.config.get('defense', {})
        
        # 文本变体生成器
        text_config = defense_config.get('text_variants', {})
        self.text_generator = TextVariantGenerator(
            num_variants=text_config.get('num_variants', 3),
            strategies=text_config.get('strategies', ['synonym', 'paraphrase']),
            device=self.device
        )
        
        # 检索参考生成器
        retrieval_config = defense_config.get('retrieval_reference', {})
        self.retrieval_generator = RetrievalReferenceGenerator(
            reference_db_path=retrieval_config.get('reference_db_path'),
            num_references=retrieval_config.get('num_references', 5),
            similarity_threshold=retrieval_config.get('similarity_threshold', 0.7),
            device=self.device
        )
        
        # 生成参考生成器
        generative_config = defense_config.get('generative_reference', {})
        self.generative_generator = GenerativeReferenceGenerator(
            model_name=generative_config.get('model_name', 'runwayml/stable-diffusion-v1-5'),
            num_references=generative_config.get('num_references', 3),
            guidance_scale=generative_config.get('guidance_scale', 7.5),
            device=self.device
        )
        
        # 一致性检查器
        consistency_config = defense_config.get('consistency_checker', {})
        self.consistency_checker = ConsistencyChecker(
            similarity_threshold=consistency_config.get('similarity_threshold', 0.8),
            voting_strategy=consistency_config.get('voting_strategy', 'weighted'),
            device=self.device
        )
    
    def run_ablation_study(self, dataset_name: str = None) -> Dict[str, Any]:
        """运行消融实验
        
        Args:
            dataset_name: 数据集名称，如果为None则使用配置中的数据集
            
        Returns:
            消融实验结果
        """
        logger.info("开始消融实验...")
        start_time = time.time()
        
        # 加载数据集
        if dataset_name is None:
            dataset_name = self.config.get('dataset', {}).get('name', 'coco')
        
        dataset_loader = get_dataset_loader(dataset_name)
        dataset_config = self.config.get('dataset', {})
        dataset = dataset_loader(
            data_dir=dataset_config.get('data_dir', './data'),
            split=dataset_config.get('split', 'val'),
            max_samples=dataset_config.get('max_samples', 1000)
        )
        
        # 定义消融实验配置
        ablation_configs = self._generate_ablation_configs()
        
        # 运行每个配置
        results = {}
        for config_name, component_config in ablation_configs.items():
            logger.info(f"运行配置: {config_name}")
            
            config_results = []
            for run_idx in range(self.config.get('ablation', {}).get('num_runs', 3)):
                logger.info(f"  运行 {run_idx + 1}/{self.config.get('ablation', {}).get('num_runs', 3)}")
                
                # 创建检测器
                detector = self._create_detector(component_config)
                
                # 运行检测
                run_results = self._run_detection(detector, dataset, f"{config_name}_run_{run_idx}")
                config_results.append(run_results)
            
            # 聚合结果
            aggregated_results = self._aggregate_results(config_results)
            results[config_name] = {
                'individual_runs': config_results,
                'aggregated': aggregated_results,
                'config': component_config
            }
        
        # 计算统计显著性
        statistical_results = self._compute_statistical_significance(results)
        
        # 生成可视化
        self._generate_visualizations(results, statistical_results)
        
        # 保存结果
        final_results = {
            'ablation_results': results,
            'statistical_analysis': statistical_results,
            'experiment_config': self.config,
            'dataset_info': {
                'name': dataset_name,
                'size': len(dataset)
            },
            'execution_time': time.time() - start_time,
            'timestamp': time.time()
        }
        
        self._save_results(final_results)
        
        logger.info(f"消融实验完成，耗时: {time.time() - start_time:.2f}秒")
        return final_results
    
    def _generate_ablation_configs(self) -> Dict[str, Dict[str, bool]]:
        """生成消融实验配置
        
        Returns:
            配置字典，键为配置名称，值为组件启用状态
        """
        configs = {
            'baseline': {
                'text_variants': False,
                'retrieval_reference': False,
                'generative_reference': False
            },
            'text_only': {
                'text_variants': True,
                'retrieval_reference': False,
                'generative_reference': False
            },
            'retrieval_only': {
                'text_variants': False,
                'retrieval_reference': True,
                'generative_reference': False
            },
            'generative_only': {
                'text_variants': False,
                'retrieval_reference': False,
                'generative_reference': True
            },
            'text_retrieval': {
                'text_variants': True,
                'retrieval_reference': True,
                'generative_reference': False
            },
            'text_generative': {
                'text_variants': True,
                'retrieval_reference': False,
                'generative_reference': True
            },
            'retrieval_generative': {
                'text_variants': False,
                'retrieval_reference': True,
                'generative_reference': True
            },
            'full_system': {
                'text_variants': True,
                'retrieval_reference': True,
                'generative_reference': True
            }
        }
        
        # 允许通过配置文件自定义消融配置
        custom_configs = self.config.get('ablation', {}).get('custom_configs', {})
        configs.update(custom_configs)
        
        return configs
    
    def _create_detector(self, component_config: Dict[str, bool]) -> MultiModalDefenseDetector:
        """创建检测器
        
        Args:
            component_config: 组件配置
            
        Returns:
            检测器实例
        """
        detection_config = DetectionConfig(
            use_text_variants=component_config['text_variants'],
            use_retrieval_reference=component_config['retrieval_reference'],
            use_generative_reference=component_config['generative_reference'],
            consistency_threshold=self.config.get('defense', {}).get('consistency_checker', {}).get('similarity_threshold', 0.8)
        )
        
        detector = MultiModalDefenseDetector(
            text_generator=self.text_generator if component_config['text_variants'] else None,
            retrieval_generator=self.retrieval_generator if component_config['retrieval_reference'] else None,
            generative_generator=self.generative_generator if component_config['generative_reference'] else None,
            consistency_checker=self.consistency_checker,
            config=detection_config
        )
        
        return detector
    
    def _run_detection(self, detector: MultiModalDefenseDetector, 
                      dataset, run_name: str) -> Dict[str, Any]:
        """运行检测
        
        Args:
            detector: 检测器
            dataset: 数据集
            run_name: 运行名称
            
        Returns:
            检测结果
        """
        predictions = []
        ground_truth = []
        confidence_scores = []
        
        for batch in tqdm(dataset, desc=f"检测 - {run_name}"):
            images = batch['images']
            texts = batch['texts']
            labels = batch.get('is_adversarial', [False] * len(images))
            
            # 批量检测
            batch_results = detector.batch_detect(images, texts)
            
            for result, label in zip(batch_results, labels):
                predictions.append(result['is_adversarial'])
                confidence_scores.append(result['confidence'])
                ground_truth.append(label)
        
        # 计算指标
        metrics = compute_detection_metrics(
            y_true=ground_truth,
            y_pred=predictions,
            y_scores=confidence_scores
        )
        
        return {
            'metrics': asdict(metrics),
            'predictions': predictions,
            'ground_truth': ground_truth,
            'confidence_scores': confidence_scores
        }
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多次运行的结果
        
        Args:
            results: 多次运行的结果列表
            
        Returns:
            聚合后的结果
        """
        metrics_list = [result['metrics'] for result in results]
        
        # 计算平均值和标准差
        aggregated = {}
        for metric_name in metrics_list[0].keys():
            values = [metrics[metric_name] for metrics in metrics_list]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return aggregated
    
    def _compute_statistical_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计显著性
        
        Args:
            results: 消融实验结果
            
        Returns:
            统计分析结果
        """
        from scipy import stats
        
        statistical_results = {}
        baseline_metrics = results['baseline']['aggregated']
        
        for config_name, config_results in results.items():
            if config_name == 'baseline':
                continue
            
            config_metrics = config_results['aggregated']
            statistical_results[config_name] = {}
            
            for metric_name in baseline_metrics.keys():
                baseline_values = baseline_metrics[metric_name]['values']
                config_values = config_metrics[metric_name]['values']
                
                # 配对t检验
                t_stat, p_value = stats.ttest_rel(config_values, baseline_values)
                
                statistical_results[config_name][metric_name] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'improvement': np.mean(config_values) - np.mean(baseline_values)
                }
        
        return statistical_results
    
    def _generate_visualizations(self, results: Dict[str, Any], 
                               statistical_results: Dict[str, Any]):
        """生成可视化图表
        
        Args:
            results: 消融实验结果
            statistical_results: 统计分析结果
        """
        # 绘制消融实验结果
        self.visualizer.plot_ablation_results(
            results, 
            save_path=self.output_dir / 'figures' / 'ablation_results.png'
        )
        
        # 绘制统计显著性
        self.visualizer.plot_statistical_significance(
            statistical_results,
            save_path=self.output_dir / 'figures' / 'statistical_significance.png'
        )
        
        # 绘制组件贡献分析
        self.visualizer.plot_component_contribution(
            results,
            save_path=self.output_dir / 'figures' / 'component_contribution.png'
        )
    
    def _save_results(self, results: Dict[str, Any]):
        """保存实验结果
        
        Args:
            results: 实验结果
        """
        # 保存完整结果
        results_path = self.output_dir / 'ablation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存摘要
        summary = self._create_summary(results)
        summary_path = self.output_dir / 'ablation_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存至: {results_path}")
        logger.info(f"摘要已保存至: {summary_path}")
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """创建结果摘要
        
        Args:
            results: 完整结果
            
        Returns:
            结果摘要
        """
        summary = {
            'best_configuration': None,
            'best_accuracy': 0,
            'component_rankings': {},
            'significant_improvements': [],
            'execution_time': results['execution_time']
        }
        
        # 找到最佳配置
        for config_name, config_results in results['ablation_results'].items():
            accuracy = config_results['aggregated']['accuracy']['mean']
            if accuracy > summary['best_accuracy']:
                summary['best_accuracy'] = accuracy
                summary['best_configuration'] = config_name
        
        # 统计显著改进
        for config_name, stats in results['statistical_analysis'].items():
            if stats['accuracy']['significant'] and stats['accuracy']['improvement'] > 0:
                summary['significant_improvements'].append({
                    'configuration': config_name,
                    'improvement': stats['accuracy']['improvement'],
                    'p_value': stats['accuracy']['p_value']
                })
        
        return summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--dataset', type=str, default=None, help='数据集名称')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建运行器
    runner = AblationRunner(config, args.output_dir)
    
    # 运行消融实验
    results = runner.run_ablation_study(args.dataset)
    
    print(f"消融实验完成，结果保存至: {args.output_dir}")

if __name__ == '__main__':
    main()