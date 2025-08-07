#!/usr/bin/env python3
"""攻击生成器

该脚本用于生成对抗样本，支持多种攻击方法。
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
from src.attacks import PGDAttack, HubnessAttack, FSTAAttacker, SMAAttacker
from src.datasets import COCOLoader, FlickrLoader, CCLoader, VGLoader
from src.models import CLIPModel
from src.utils import ConfigManager, CUDADeviceManager
from experiments.utils import ExperimentLogger, set_random_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttackRunner:
    """攻击运行器"""
    
    def __init__(self, config_path: str, output_dir: str):
        """初始化攻击运行器
        
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
        
        # 初始化模型
        self.clip_model = CLIPModel(self.config.models.clip)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # 初始化攻击器
        self.attackers = self._init_attackers()
        
        # 实验日志
        self.exp_logger = ExperimentLogger(self.output_dir / 'attack_log.json')
    
    def _init_attackers(self) -> Dict[str, object]:
        """初始化攻击器
        
        Returns:
            攻击器字典
        """
        attackers = {}
        
        if hasattr(self.config.attacks, 'pgd'):
            attackers['pgd'] = PGDAttack(
                model=self.clip_model,
                config=self.config.attacks.pgd
            )
        
        if hasattr(self.config.attacks, 'hubness'):
            attackers['hubness'] = HubnessAttack(
                model=self.clip_model,
                config=self.config.attacks.hubness
            )
        
        if hasattr(self.config.attacks, 'fsta'):
            attackers['fsta'] = FSTAAttacker(
                clip_model=self.clip_model,
                config=self.config.attacks.fsta
            )
        
        if hasattr(self.config.attacks, 'sma'):
            attackers['sma'] = SMAAttacker(
                clip_model=self.clip_model,
                config=self.config.attacks.sma
            )
        
        return attackers
    
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
    
    def run_attack(self, attack_name: str, dataset_name: str, split: str = 'val') -> Dict:
        """运行攻击
        
        Args:
            attack_name: 攻击名称
            dataset_name: 数据集名称
            split: 数据集分割
            
        Returns:
            攻击结果
        """
        logger.info(f"运行 {attack_name} 攻击，数据集: {dataset_name}")
        
        if attack_name not in self.attackers:
            raise ValueError(f"不支持的攻击方法: {attack_name}")
        
        attacker = self.attackers[attack_name]
        dataloader = self.load_dataset(dataset_name, split)
        
        # 攻击结果存储
        attack_results = {
            'attack_name': attack_name,
            'dataset_name': dataset_name,
            'split': split,
            'samples': [],
            'statistics': {
                'total_samples': 0,
                'successful_attacks': 0,
                'avg_perturbation_norm': 0.0,
                'avg_similarity_drop': 0.0
            }
        }
        
        total_perturbation_norm = 0.0
        total_similarity_drop = 0.0
        successful_attacks = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"运行{attack_name}攻击")):
                images = batch['image'].to(self.device)
                texts = batch['text']
                image_ids = batch['image_id']
                
                # 计算原始相似度
                original_img_features = self.clip_model.encode_image(images)
                original_txt_features = self.clip_model.encode_text(texts)
                original_similarities = torch.cosine_similarity(
                    original_img_features, original_txt_features, dim=-1
                )
                
                # 执行攻击 - 使用批量攻击方法
                try:
                    # 将批量数据转换为列表格式
                    batch_images = [images[i] for i in range(images.size(0))]
                    batch_texts = list(texts)
                    
                    # 调用批量攻击方法
                    batch_results = attacker.batch_attack(batch_images, batch_texts)
                    
                    # 从批量结果中提取对抗图像
                    adversarial_images = torch.stack([
                        result.get('adversarial_image', images[i]) 
                        for i, result in enumerate(batch_results)
                    ])
                    
                    # 提取攻击信息
                    attack_info = [result.get('attack_info', {}) for result in batch_results]
                    
                    # 计算攻击后的相似度
                    adv_img_features = self.clip_model.encode_image(adversarial_images)
                    adv_similarities = torch.cosine_similarity(
                        adv_img_features, original_txt_features, dim=-1
                    )
                    
                    # 计算扰动范数
                    perturbations = adversarial_images - images
                    perturbation_norms = torch.norm(perturbations.view(perturbations.size(0), -1), dim=1)
                    
                    # 计算相似度下降
                    similarity_drops = original_similarities - adv_similarities
                    
                    # 判断攻击是否成功（相似度下降超过阈值）
                    success_threshold = getattr(self.config.attacks.get(attack_name, {}), 'success_threshold', 0.1)
                    attack_success = similarity_drops > success_threshold
                    
                    # 记录结果
                    for i in range(len(images)):
                        sample_result = {
                            'image_id': image_ids[i],
                            'text': texts[i],
                            'original_similarity': original_similarities[i].item(),
                            'adversarial_similarity': adv_similarities[i].item(),
                            'similarity_drop': similarity_drops[i].item(),
                            'perturbation_norm': perturbation_norms[i].item(),
                            'attack_success': attack_success[i].item(),
                            'attack_info': attack_info[i] if isinstance(attack_info, list) else attack_info
                        }
                        attack_results['samples'].append(sample_result)
                        
                        # 更新统计信息
                        total_perturbation_norm += perturbation_norms[i].item()
                        total_similarity_drop += similarity_drops[i].item()
                        if attack_success[i]:
                            successful_attacks += 1
                    
                    attack_results['statistics']['total_samples'] += len(images)
                    
                except Exception as e:
                    logger.error(f"攻击批次 {batch_idx} 时出错: {e}")
                    continue
        
        # 计算最终统计信息
        total_samples = attack_results['statistics']['total_samples']
        if total_samples > 0:
            attack_results['statistics']['successful_attacks'] = successful_attacks
            attack_results['statistics']['success_rate'] = successful_attacks / total_samples
            attack_results['statistics']['avg_perturbation_norm'] = total_perturbation_norm / total_samples
            attack_results['statistics']['avg_similarity_drop'] = total_similarity_drop / total_samples
        
        logger.info(f"{attack_name} 攻击完成:")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  成功攻击: {successful_attacks}")
        logger.info(f"  成功率: {attack_results['statistics']['success_rate']:.3f}")
        logger.info(f"  平均扰动范数: {attack_results['statistics']['avg_perturbation_norm']:.6f}")
        logger.info(f"  平均相似度下降: {attack_results['statistics']['avg_similarity_drop']:.6f}")
        
        return attack_results
    
    def save_results(self, results: Dict, filename: str):
        """保存攻击结果
        
        Args:
            results: 攻击结果
            filename: 文件名
        """
        output_path = self.output_dir / filename
        
        # 保存详细结果
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存统计摘要
        summary_path = self.output_dir / f"summary_{filename}"
        summary = {
            'attack_name': results['attack_name'],
            'dataset_name': results['dataset_name'],
            'statistics': results['statistics']
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path}")
        logger.info(f"摘要已保存到: {summary_path}")
    
    def run_all_attacks(self, dataset_name: str, split: str = 'val'):
        """运行所有配置的攻击
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
        """
        logger.info(f"运行所有攻击，数据集: {dataset_name}")
        
        all_results = {}
        
        for attack_name in self.attackers.keys():
            try:
                results = self.run_attack(attack_name, dataset_name, split)
                all_results[attack_name] = results
                
                # 保存单个攻击结果
                filename = f"{attack_name}_{dataset_name}_{split}_results.json"
                self.save_results(results, filename)
                
            except Exception as e:
                logger.error(f"运行 {attack_name} 攻击时出错: {e}")
                continue
        
        # 保存汇总结果
        summary_filename = f"all_attacks_{dataset_name}_{split}_summary.json"
        summary_path = self.output_dir / summary_filename
        
        summary = {}
        for attack_name, results in all_results.items():
            summary[attack_name] = results['statistics']
        
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"汇总结果已保存到: {summary_path}")
        
        return all_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行对抗攻击')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['coco', 'flickr30k', 'cc3m', 'visual_genome'],
                       help='数据集名称')
    parser.add_argument('--split', type=str, default='val', help='数据集分割')
    parser.add_argument('--attack', type=str, choices=['pgd', 'hubness', 'fsta', 'sma'],
                       help='攻击方法（如果不指定则运行所有攻击）')
    
    args = parser.parse_args()
    
    # 初始化攻击运行器
    runner = AttackRunner(args.config, args.output_dir)
    
    # 运行攻击
    if args.attack:
        results = runner.run_attack(args.attack, args.dataset, args.split)
        filename = f"{args.attack}_{args.dataset}_{args.split}_results.json"
        runner.save_results(results, filename)
    else:
        runner.run_all_attacks(args.dataset, args.split)
    
    logger.info("攻击运行完成！")

if __name__ == '__main__':
    main()