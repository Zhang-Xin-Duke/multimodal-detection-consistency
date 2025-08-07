#!/usr/bin/env python3
"""检测和评估主入口

该脚本是多模态对抗检测的主要运行入口，支持完整的检测流水线。
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.pipeline import MultiModalDetectionPipeline
from src.datasets import COCOLoader, FlickrLoader, CCLoader, VGLoader
from src.utils import (
    ConfigManager, CUDADeviceManager, MetricsAggregator, 
    VisualizationManager, DetectionEvaluator
)
from experiments.utils import ExperimentLogger, set_random_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionRunner:
    """检测运行器"""
    
    def __init__(self, config_path: str, output_dir: str):
        """初始化检测运行器
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        self.config = Config.load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        set_random_seed(self.config.random_seed)
        
        # 设备管理
        self.device_manager = CUDADeviceManager()
        self.device = self.device_manager.get_optimal_device()
        
        # 初始化检测流水线
        self.pipeline = MultiModalDetectionPipeline(self.config)
        
        # 初始化评估器
        self.evaluator = DetectionEvaluator()
        self.metrics_aggregator = MetricsAggregator()
        self.visualizer = VisualizationManager(self.output_dir / 'figures')
        
        # 实验日志
        self.exp_logger = ExperimentLogger(self.output_dir / 'detection_log.json')
        
        logger.info(f"检测运行器初始化完成，输出目录: {self.output_dir}")
    
    def load_dataset(self, dataset_name: str, split: str = 'val'):
        """加载数据集
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            
        Returns:
            数据加载器
        """
        if dataset_name == 'coco':
            loader = COCOLoader(self.config.datasets.coco)
        elif dataset_name == 'flickr30k':
            loader = FlickrLoader(self.config.datasets.flickr30k)
        elif dataset_name == 'cc3m':
            loader = CCLoader(self.config.datasets.cc3m)
        elif dataset_name == 'visual_genome':
            loader = VGLoader(self.config.datasets.visual_genome)
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        dataset = loader.load_dataset(split=split)
        
        # 限制样本数量（如果配置中指定）
        if hasattr(self.config.experiment, 'max_samples'):
            max_samples = self.config.experiment.max_samples
            if max_samples > 0 and len(dataset) > max_samples:
                indices = torch.randperm(len(dataset))[:max_samples]
                dataset = torch.utils.data.Subset(dataset, indices)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    def load_adversarial_samples(self, attack_results_path: str) -> Dict:
        """加载对抗样本
        
        Args:
            attack_results_path: 攻击结果文件路径
            
        Returns:
            对抗样本数据
        """
        import json
        with open(attack_results_path, 'r', encoding='utf-8') as f:
            attack_results = json.load(f)
        
        logger.info(f"加载对抗样本: {len(attack_results['samples'])} 个样本")
        return attack_results
    
    def run_detection(self, dataset_name: str, split: str = 'val', 
                     attack_results_path: Optional[str] = None) -> Dict:
        """运行检测
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            attack_results_path: 攻击结果文件路径（可选）
            
        Returns:
            检测结果
        """
        logger.info(f"运行检测，数据集: {dataset_name}")
        
        # 加载数据
        dataloader = self.load_dataset(dataset_name, split)
        
        # 加载对抗样本（如果提供）
        adversarial_data = None
        if attack_results_path:
            adversarial_data = self.load_adversarial_samples(attack_results_path)
        
        # 检测结果存储
        detection_results = {
            'dataset_name': dataset_name,
            'split': split,
            'attack_results_path': attack_results_path,
            'samples': [],
            'statistics': {
                'total_samples': 0,
                'clean_samples': 0,
                'adversarial_samples': 0,
                'detection_metrics': {}
            }
        }
        
        all_predictions = []
        all_labels = []  # 0: clean, 1: adversarial
        all_scores = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="运行检测")):
                images = batch['image'].to(self.device)
                texts = batch['text']
                image_ids = batch['image_id']
                
                try:
                    # 运行检测流水线
                    detection_outputs = self.pipeline.detect(
                        images=images,
                        texts=texts,
                        return_details=True
                    )
                    
                    # 处理检测结果
                    for i in range(len(images)):
                        image_id = image_ids[i]
                        
                        # 确定真实标签
                        is_adversarial = False
                        if adversarial_data:
                            # 检查是否为对抗样本
                            for adv_sample in adversarial_data['samples']:
                                if adv_sample['image_id'] == image_id:
                                    is_adversarial = adv_sample.get('attack_success', False)
                                    break
                        
                        label = 1 if is_adversarial else 0
                        prediction = detection_outputs['predictions'][i]
                        score = detection_outputs['scores'][i]
                        
                        # 记录样本结果
                        sample_result = {
                            'image_id': image_id,
                            'text': texts[i],
                            'true_label': label,
                            'predicted_label': prediction,
                            'detection_score': score,
                            'is_adversarial': is_adversarial,
                            'detection_details': detection_outputs['details'][i]
                        }
                        detection_results['samples'].append(sample_result)
                        
                        # 收集用于评估的数据
                        all_predictions.append(prediction)
                        all_labels.append(label)
                        all_scores.append(score)
                        
                        # 更新统计信息
                        detection_results['statistics']['total_samples'] += 1
                        if is_adversarial:
                            detection_results['statistics']['adversarial_samples'] += 1
                        else:
                            detection_results['statistics']['clean_samples'] += 1
                
                except Exception as e:
                    logger.error(f"检测批次 {batch_idx} 时出错: {e}")
                    continue
        
        # 计算检测指标
        if len(all_labels) > 0:
            detection_metrics = self.evaluator.evaluate_detection(
                y_true=np.array(all_labels),
                y_pred=np.array(all_predictions),
                y_scores=np.array(all_scores)
            )
            detection_results['statistics']['detection_metrics'] = detection_metrics
            
            logger.info(f"检测完成:")
            logger.info(f"  总样本数: {detection_results['statistics']['total_samples']}")
            logger.info(f"  干净样本: {detection_results['statistics']['clean_samples']}")
            logger.info(f"  对抗样本: {detection_results['statistics']['adversarial_samples']}")
            logger.info(f"  准确率: {detection_metrics.accuracy:.3f}")
            logger.info(f"  精确率: {detection_metrics.precision:.3f}")
            logger.info(f"  召回率: {detection_metrics.recall:.3f}")
            logger.info(f"  F1分数: {detection_metrics.f1_score:.3f}")
            logger.info(f"  AUC: {detection_metrics.auc:.3f}")
        
        return detection_results
    
    def run_ablation_study(self, dataset_name: str, split: str = 'val',
                          attack_results_path: Optional[str] = None) -> Dict:
        """运行消融研究
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            attack_results_path: 攻击结果文件路径
            
        Returns:
            消融研究结果
        """
        logger.info(f"运行消融研究，数据集: {dataset_name}")
        
        # 定义消融配置
        ablation_configs = {
            'base_only': {'use_text_variants': False, 'use_generated_refs': False},
            'with_text_variants': {'use_text_variants': True, 'use_generated_refs': False},
            'with_generated_refs': {'use_text_variants': False, 'use_generated_refs': True},
            'full_method': {'use_text_variants': True, 'use_generated_refs': True}
        }
        
        ablation_results = {}
        
        for config_name, config_changes in ablation_configs.items():
            logger.info(f"运行消融配置: {config_name}")
            
            # 临时修改配置
            original_config = self.config.defenses.copy()
            for key, value in config_changes.items():
                setattr(self.config.defenses, key, value)
            
            # 重新初始化流水线
            self.pipeline = MultiModalDetectionPipeline(self.config)
            
            # 运行检测
            results = self.run_detection(dataset_name, split, attack_results_path)
            ablation_results[config_name] = results['statistics']['detection_metrics']
            
            # 恢复原始配置
            self.config.defenses = original_config
        
        # 重新初始化流水线
        self.pipeline = MultiModalDetectionPipeline(self.config)
        
        logger.info("消融研究完成")
        return ablation_results
    
    def generate_visualizations(self, detection_results: Dict, 
                              ablation_results: Optional[Dict] = None):
        """生成可视化图表
        
        Args:
            detection_results: 检测结果
            ablation_results: 消融研究结果（可选）
        """
        logger.info("生成可视化图表...")
        
        # 提取数据
        y_true = [sample['true_label'] for sample in detection_results['samples']]
        y_scores = [sample['detection_score'] for sample in detection_results['samples']]
        
        # ROC曲线
        self.visualizer.plot_roc_curve(
            y_true=y_true,
            y_scores=y_scores,
            title=f"ROC Curve - {detection_results['dataset_name']}",
            save_path='roc_curve.png'
        )
        
        # PR曲线
        self.visualizer.plot_pr_curve(
            y_true=y_true,
            y_scores=y_scores,
            title=f"PR Curve - {detection_results['dataset_name']}",
            save_path='pr_curve.png'
        )
        
        # 分数分布
        clean_scores = [s['detection_score'] for s in detection_results['samples'] if not s['is_adversarial']]
        adv_scores = [s['detection_score'] for s in detection_results['samples'] if s['is_adversarial']]
        
        self.visualizer.plot_score_distribution(
            clean_scores=clean_scores,
            adversarial_scores=adv_scores,
            title=f"Detection Score Distribution - {detection_results['dataset_name']}",
            save_path='score_distribution.png'
        )
        
        # 消融研究结果
        if ablation_results:
            self.visualizer.plot_ablation_results(
                ablation_results,
                title="Ablation Study Results",
                save_path='ablation_results.png'
            )
        
        logger.info(f"可视化图表已保存到: {self.visualizer.output_dir}")
    
    def save_results(self, results: Dict, filename: str):
        """保存检测结果
        
        Args:
            results: 检测结果
            filename: 文件名
        """
        output_path = self.output_dir / filename
        
        # 保存详细结果
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存CSV格式的样本结果
        if 'samples' in results:
            import pandas as pd
            
            samples_df = pd.DataFrame(results['samples'])
            csv_path = output_path.with_suffix('.csv')
            samples_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            logger.info(f"CSV结果已保存到: {csv_path}")
        
        logger.info(f"结果已保存到: {output_path}")
    
    def run_full_evaluation(self, dataset_name: str, split: str = 'val',
                           attack_results_path: Optional[str] = None,
                           run_ablation: bool = True) -> Dict:
        """运行完整评估
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            attack_results_path: 攻击结果文件路径
            run_ablation: 是否运行消融研究
            
        Returns:
            完整评估结果
        """
        logger.info(f"运行完整评估，数据集: {dataset_name}")
        
        # 运行主检测
        detection_results = self.run_detection(dataset_name, split, attack_results_path)
        
        # 保存检测结果
        detection_filename = f"detection_{dataset_name}_{split}_results.json"
        self.save_results(detection_results, detection_filename)
        
        # 运行消融研究
        ablation_results = None
        if run_ablation:
            ablation_results = self.run_ablation_study(dataset_name, split, attack_results_path)
            ablation_filename = f"ablation_{dataset_name}_{split}_results.json"
            self.save_results(ablation_results, ablation_filename)
        
        # 生成可视化
        self.generate_visualizations(detection_results, ablation_results)
        
        # 汇总结果
        full_results = {
            'detection_results': detection_results,
            'ablation_results': ablation_results,
            'experiment_config': self.config.to_dict()
        }
        
        # 保存汇总结果
        summary_filename = f"full_evaluation_{dataset_name}_{split}_summary.json"
        self.save_results(full_results, summary_filename)
        
        logger.info("完整评估完成")
        return full_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行多模态对抗检测')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['coco', 'flickr30k', 'cc3m', 'visual_genome'],
                       help='数据集名称')
    parser.add_argument('--split', type=str, default='val', help='数据集分割')
    parser.add_argument('--attack-results', type=str, help='攻击结果文件路径')
    parser.add_argument('--no-ablation', action='store_true', help='跳过消融研究')
    parser.add_argument('--detection-only', action='store_true', help='仅运行检测，不运行完整评估')
    
    args = parser.parse_args()
    
    # 初始化检测运行器
    runner = DetectionRunner(args.config, args.output_dir)
    
    # 运行评估
    if args.detection_only:
        results = runner.run_detection(args.dataset, args.split, args.attack_results)
        filename = f"detection_{args.dataset}_{args.split}_results.json"
        runner.save_results(results, filename)
    else:
        runner.run_full_evaluation(
            dataset_name=args.dataset,
            split=args.split,
            attack_results_path=args.attack_results,
            run_ablation=not args.no_ablation
        )
    
    logger.info("检测运行完成！")

if __name__ == '__main__':
    main()