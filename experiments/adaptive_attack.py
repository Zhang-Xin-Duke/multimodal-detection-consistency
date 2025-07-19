#!/usr/bin/env python3
"""自适应攻击实验模块

本模块实现了针对多模态检测一致性系统的自适应攻击实验，
包括多种攻击策略的实现和评估，用于测试系统的鲁棒性。

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
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigManager
from utils.data_loader import DataLoaderManager
from utils.metrics import MetricsCalculator
from utils.visualization import VisualizationManager
from pipeline import MultiModalDetectionPipeline, create_detection_pipeline
from evaluation import ExperimentEvaluator, ExperimentConfig, create_experiment_evaluator
from attacks import HubnessAttacker, create_hubness_attacker
from models.clip_model import CLIPModel
from models.qwen_model import QwenModel
from models.sd_model import StableDiffusionModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adaptive_attack.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AdaptiveAttackConfig:
    """自适应攻击实验配置"""
    experiment_name: str = "adaptive_attack_experiment"
    output_dir: str = "./results/adaptive_attack"
    num_samples: int = 1000
    num_attack_iterations: int = 10
    attack_budget: float = 0.1  # 攻击预算（扰动强度）
    
    # 攻击策略配置
    use_gradient_based: bool = True
    use_query_based: bool = True
    use_transfer_based: bool = True
    use_ensemble_attack: bool = True
    
    # 自适应策略配置
    adaptive_step_size: bool = True
    adaptive_target_selection: bool = True
    adaptive_loss_function: bool = True
    
    # 评估配置
    evaluate_robustness: bool = True
    evaluate_transferability: bool = True
    generate_visualizations: bool = True
    save_attack_samples: bool = True


class AdaptiveAttacker:
    """自适应攻击器基类"""
    
    def __init__(self, target_pipeline: MultiModalDetectionPipeline, config: Dict[str, Any]):
        """
        初始化自适应攻击器
        
        Args:
            target_pipeline: 目标检测管道
            config: 攻击配置
        """
        self.target_pipeline = target_pipeline
        self.config = config
        self.attack_history = []
        self.success_rate = 0.0
        
    def attack(self, image: Union[Image.Image, torch.Tensor], 
              text: str, 
              target_label: Optional[int] = None) -> Dict[str, Any]:
        """
        执行自适应攻击
        
        Args:
            image: 输入图像
            text: 输入文本
            target_label: 目标标签（可选）
        
        Returns:
            攻击结果字典
        """
        raise NotImplementedError
    
    def update_strategy(self, attack_result: Dict[str, Any]):
        """
        根据攻击结果更新策略
        
        Args:
            attack_result: 攻击结果
        """
        self.attack_history.append(attack_result)
        
        # 计算成功率
        recent_results = self.attack_history[-10:]  # 最近10次攻击
        success_count = sum(1 for r in recent_results if r.get('success', False))
        self.success_rate = success_count / len(recent_results)


class GradientBasedAdaptiveAttacker(AdaptiveAttacker):
    """基于梯度的自适应攻击器"""
    
    def __init__(self, target_pipeline: MultiModalDetectionPipeline, config: Dict[str, Any]):
        super().__init__(target_pipeline, config)
        self.step_size = config.get('initial_step_size', 0.01)
        self.max_iterations = config.get('max_iterations', 20)
        
    def attack(self, image: Union[Image.Image, torch.Tensor], 
              text: str, 
              target_label: Optional[int] = None) -> Dict[str, Any]:
        """
        执行基于梯度的自适应攻击
        """
        start_time = time.time()
        
        # 转换输入格式
        if isinstance(image, Image.Image):
            image_tensor = self._pil_to_tensor(image)
        else:
            image_tensor = image.clone()
        
        image_tensor.requires_grad_(True)
        
        # 获取原始预测
        original_result = self.target_pipeline.detect(image, text)
        original_prediction = original_result['prediction']
        
        # 初始化攻击
        perturbed_image = image_tensor.clone()
        best_perturbation = None
        best_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # 前向传播
            perturbed_pil = self._tensor_to_pil(perturbed_image)
            result = self.target_pipeline.detect(perturbed_pil, text)
            
            # 计算损失
            loss = self._compute_attack_loss(result, original_prediction, target_label)
            
            # 反向传播
            loss.backward()
            
            # 获取梯度
            grad = perturbed_image.grad.data
            
            # 自适应步长调整
            if self.config.get('adaptive_step_size', True):
                self.step_size = self._adapt_step_size(iteration, loss.item())
            
            # 更新扰动
            perturbation = self.step_size * grad.sign()
            perturbed_image = perturbed_image - perturbation
            
            # 投影到有效范围
            perturbed_image = self._project_perturbation(image_tensor, perturbed_image)
            
            # 清零梯度
            perturbed_image.grad = None
            
            # 检查攻击成功
            if self._check_attack_success(result, original_prediction):
                best_perturbation = perturbed_image - image_tensor
                break
            
            # 更新最佳扰动
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_perturbation = perturbed_image - image_tensor
        
        # 生成最终结果
        if best_perturbation is not None:
            final_image = image_tensor + best_perturbation
            final_pil = self._tensor_to_pil(final_image)
            final_result = self.target_pipeline.detect(final_pil, text)
            
            attack_result = {
                'success': self._check_attack_success(final_result, original_prediction),
                'original_prediction': original_prediction,
                'adversarial_prediction': final_result['prediction'],
                'adversarial_image': final_pil,
                'perturbation_norm': torch.norm(best_perturbation).item(),
                'iterations': iteration + 1,
                'execution_time': time.time() - start_time,
                'attack_type': 'gradient_based'
            }
        else:
            attack_result = {
                'success': False,
                'original_prediction': original_prediction,
                'execution_time': time.time() - start_time,
                'attack_type': 'gradient_based'
            }
        
        # 更新策略
        self.update_strategy(attack_result)
        
        return attack_result
    
    def _compute_attack_loss(self, result: Dict[str, Any], 
                           original_prediction: int, 
                           target_label: Optional[int] = None) -> torch.Tensor:
        """
        计算攻击损失函数
        """
        if target_label is not None:
            # 目标攻击：最大化目标类别的概率
            target_prob = result.get('confidence', {}).get(str(target_label), 0.0)
            loss = -torch.log(torch.tensor(target_prob + 1e-8))
        else:
            # 非目标攻击：最小化原始预测的概率
            original_prob = result.get('confidence', {}).get(str(original_prediction), 0.0)
            loss = torch.log(torch.tensor(original_prob + 1e-8))
        
        return loss
    
    def _adapt_step_size(self, iteration: int, current_loss: float) -> float:
        """
        自适应调整步长
        """
        # 基于成功率和损失调整步长
        if self.success_rate < 0.3:
            # 成功率低，增加步长
            self.step_size *= 1.1
        elif self.success_rate > 0.7:
            # 成功率高，减少步长以提高精度
            self.step_size *= 0.9
        
        # 限制步长范围
        self.step_size = np.clip(self.step_size, 0.001, 0.1)
        
        return self.step_size
    
    def _project_perturbation(self, original: torch.Tensor, 
                            perturbed: torch.Tensor) -> torch.Tensor:
        """
        将扰动投影到有效范围内
        """
        # L∞范数约束
        epsilon = self.config.get('attack_budget', 0.1)
        perturbation = perturbed - original
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        
        # 像素值约束
        result = original + perturbation
        result = torch.clamp(result, 0, 1)
        
        return result
    
    def _check_attack_success(self, result: Dict[str, Any], 
                            original_prediction: int) -> bool:
        """
        检查攻击是否成功
        """
        return result['prediction'] != original_prediction
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        将PIL图像转换为张量
        """
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        将张量转换为PIL图像
        """
        import torchvision.transforms as transforms
        transform = transforms.ToPILImage()
        return transform(tensor.squeeze(0).clamp(0, 1))


class QueryBasedAdaptiveAttacker(AdaptiveAttacker):
    """基于查询的自适应攻击器（黑盒攻击）"""
    
    def __init__(self, target_pipeline: MultiModalDetectionPipeline, config: Dict[str, Any]):
        super().__init__(target_pipeline, config)
        self.query_budget = config.get('query_budget', 1000)
        self.query_count = 0
        
    def attack(self, image: Union[Image.Image, torch.Tensor], 
              text: str, 
              target_label: Optional[int] = None) -> Dict[str, Any]:
        """
        执行基于查询的自适应攻击
        """
        start_time = time.time()
        self.query_count = 0
        
        # 转换输入格式
        if isinstance(image, torch.Tensor):
            image = self._tensor_to_pil(image)
        
        # 获取原始预测
        original_result = self._query_model(image, text)
        original_prediction = original_result['prediction']
        
        # 初始化攻击
        best_image = image
        best_distance = float('inf')
        
        # 使用进化算法进行攻击
        population_size = 20
        population = self._initialize_population(image, population_size)
        
        for generation in range(self.config.get('max_generations', 50)):
            if self.query_count >= self.query_budget:
                break
            
            # 评估种群
            fitness_scores = []
            for individual in population:
                if self.query_count >= self.query_budget:
                    break
                
                result = self._query_model(individual, text)
                fitness = self._compute_fitness(result, original_prediction, target_label)
                fitness_scores.append(fitness)
                
                # 检查攻击成功
                if self._check_attack_success(result, original_prediction):
                    distance = self._compute_distance(image, individual)
                    if distance < best_distance:
                        best_image = individual
                        best_distance = distance
            
            # 选择和变异
            population = self._evolve_population(population, fitness_scores)
        
        # 生成最终结果
        final_result = self._query_model(best_image, text)
        
        attack_result = {
            'success': self._check_attack_success(final_result, original_prediction),
            'original_prediction': original_prediction,
            'adversarial_prediction': final_result['prediction'],
            'adversarial_image': best_image,
            'distance': best_distance,
            'queries_used': self.query_count,
            'execution_time': time.time() - start_time,
            'attack_type': 'query_based'
        }
        
        # 更新策略
        self.update_strategy(attack_result)
        
        return attack_result
    
    def _query_model(self, image: Image.Image, text: str) -> Dict[str, Any]:
        """
        查询目标模型
        """
        self.query_count += 1
        return self.target_pipeline.detect(image, text)
    
    def _initialize_population(self, image: Image.Image, size: int) -> List[Image.Image]:
        """
        初始化种群
        """
        population = [image]  # 包含原始图像
        
        for _ in range(size - 1):
            # 添加随机扰动
            perturbed = self._add_random_noise(image)
            population.append(perturbed)
        
        return population
    
    def _add_random_noise(self, image: Image.Image) -> Image.Image:
        """
        添加随机噪声
        """
        import torchvision.transforms as transforms
        
        # 转换为张量
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        
        tensor = to_tensor(image)
        
        # 添加噪声
        noise_scale = self.config.get('attack_budget', 0.1)
        noise = torch.randn_like(tensor) * noise_scale
        perturbed = torch.clamp(tensor + noise, 0, 1)
        
        return to_pil(perturbed)
    
    def _compute_fitness(self, result: Dict[str, Any], 
                        original_prediction: int, 
                        target_label: Optional[int] = None) -> float:
        """
        计算适应度分数
        """
        if target_label is not None:
            # 目标攻击
            target_confidence = result.get('confidence', {}).get(str(target_label), 0.0)
            return target_confidence
        else:
            # 非目标攻击
            original_confidence = result.get('confidence', {}).get(str(original_prediction), 1.0)
            return 1.0 - original_confidence
    
    def _evolve_population(self, population: List[Image.Image], 
                          fitness_scores: List[float]) -> List[Image.Image]:
        """
        进化种群
        """
        # 选择最优个体
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_size = len(population) // 4
        
        new_population = []
        
        # 保留精英
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]])
        
        # 生成新个体
        while len(new_population) < len(population):
            # 选择父母
            parent1_idx = np.random.choice(elite_size)
            parent2_idx = np.random.choice(elite_size)
            
            parent1 = population[sorted_indices[parent1_idx]]
            parent2 = population[sorted_indices[parent2_idx]]
            
            # 交叉和变异
            child = self._crossover_and_mutate(parent1, parent2)
            new_population.append(child)
        
        return new_population
    
    def _crossover_and_mutate(self, parent1: Image.Image, 
                             parent2: Image.Image) -> Image.Image:
        """
        交叉和变异操作
        """
        import torchvision.transforms as transforms
        
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        
        tensor1 = to_tensor(parent1)
        tensor2 = to_tensor(parent2)
        
        # 交叉
        alpha = np.random.random()
        child_tensor = alpha * tensor1 + (1 - alpha) * tensor2
        
        # 变异
        mutation_rate = 0.1
        if np.random.random() < mutation_rate:
            noise = torch.randn_like(child_tensor) * 0.01
            child_tensor = child_tensor + noise
        
        child_tensor = torch.clamp(child_tensor, 0, 1)
        
        return to_pil(child_tensor)
    
    def _compute_distance(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        计算图像距离
        """
        import torchvision.transforms as transforms
        
        to_tensor = transforms.ToTensor()
        
        tensor1 = to_tensor(image1)
        tensor2 = to_tensor(image2)
        
        return torch.norm(tensor1 - tensor2).item()


class TransferBasedAdaptiveAttacker(AdaptiveAttacker):
    """基于迁移的自适应攻击器"""
    
    def __init__(self, target_pipeline: MultiModalDetectionPipeline, 
                 surrogate_models: List[MultiModalDetectionPipeline], 
                 config: Dict[str, Any]):
        super().__init__(target_pipeline, config)
        self.surrogate_models = surrogate_models
        
    def attack(self, image: Union[Image.Image, torch.Tensor], 
              text: str, 
              target_label: Optional[int] = None) -> Dict[str, Any]:
        """
        执行基于迁移的自适应攻击
        """
        start_time = time.time()
        
        # 在代理模型上生成对抗样本
        best_adversarial = None
        best_transferability = 0.0
        
        for surrogate in self.surrogate_models:
            # 使用梯度方法在代理模型上生成对抗样本
            gradient_attacker = GradientBasedAdaptiveAttacker(surrogate, self.config)
            surrogate_result = gradient_attacker.attack(image, text, target_label)
            
            if surrogate_result['success']:
                # 测试在目标模型上的迁移性
                target_result = self.target_pipeline.detect(
                    surrogate_result['adversarial_image'], text
                )
                
                # 计算迁移性分数
                transferability = self._compute_transferability(
                    surrogate_result, target_result
                )
                
                if transferability > best_transferability:
                    best_transferability = transferability
                    best_adversarial = surrogate_result
        
        # 生成最终结果
        if best_adversarial is not None:
            final_result = self.target_pipeline.detect(
                best_adversarial['adversarial_image'], text
            )
            
            original_result = self.target_pipeline.detect(image, text)
            
            attack_result = {
                'success': self._check_attack_success(final_result, original_result['prediction']),
                'original_prediction': original_result['prediction'],
                'adversarial_prediction': final_result['prediction'],
                'adversarial_image': best_adversarial['adversarial_image'],
                'transferability_score': best_transferability,
                'execution_time': time.time() - start_time,
                'attack_type': 'transfer_based'
            }
        else:
            original_result = self.target_pipeline.detect(image, text)
            attack_result = {
                'success': False,
                'original_prediction': original_result['prediction'],
                'execution_time': time.time() - start_time,
                'attack_type': 'transfer_based'
            }
        
        # 更新策略
        self.update_strategy(attack_result)
        
        return attack_result
    
    def _compute_transferability(self, surrogate_result: Dict[str, Any], 
                               target_result: Dict[str, Any]) -> float:
        """
        计算迁移性分数
        """
        # 简单的迁移性度量：预测是否改变
        if (surrogate_result['original_prediction'] != target_result['prediction']):
            return 1.0
        else:
            # 基于置信度变化的软迁移性度量
            original_conf = surrogate_result.get('original_confidence', 0.5)
            target_conf = target_result.get('confidence', {}).get(
                str(surrogate_result['original_prediction']), 0.5
            )
            return max(0, original_conf - target_conf)


class AdaptiveAttackExperimentManager:
    """自适应攻击实验管理器"""
    
    def __init__(self, config: AdaptiveAttackConfig, base_config: Dict[str, Any]):
        """
        初始化实验管理器
        
        Args:
            config: 实验配置
            base_config: 基础系统配置
        """
        self.config = config
        self.base_config = base_config
        self.results = {}
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.metrics_calculator = MetricsCalculator()
        self.visualization_manager = VisualizationManager()
        
    def run_adaptive_attack_experiments(self, test_data: List[Tuple]) -> Dict[str, Any]:
        """
        运行自适应攻击实验
        
        Args:
            test_data: 测试数据
        
        Returns:
            实验结果字典
        """
        logger.info("开始自适应攻击实验")
        start_time = time.time()
        
        # 创建目标检测管道
        target_pipeline = create_detection_pipeline(self.base_config)
        
        # 创建代理模型（用于迁移攻击）
        surrogate_pipelines = self._create_surrogate_models()
        
        all_results = {}
        
        # 1. 基于梯度的攻击
        if self.config.use_gradient_based:
            logger.info("运行基于梯度的自适应攻击")
            gradient_attacker = GradientBasedAdaptiveAttacker(target_pipeline, self.base_config)
            all_results['gradient_based'] = self._run_attack_experiment(
                gradient_attacker, test_data, "gradient_based"
            )
        
        # 2. 基于查询的攻击
        if self.config.use_query_based:
            logger.info("运行基于查询的自适应攻击")
            query_attacker = QueryBasedAdaptiveAttacker(target_pipeline, self.base_config)
            all_results['query_based'] = self._run_attack_experiment(
                query_attacker, test_data, "query_based"
            )
        
        # 3. 基于迁移的攻击
        if self.config.use_transfer_based and surrogate_pipelines:
            logger.info("运行基于迁移的自适应攻击")
            transfer_attacker = TransferBasedAdaptiveAttacker(
                target_pipeline, surrogate_pipelines, self.base_config
            )
            all_results['transfer_based'] = self._run_attack_experiment(
                transfer_attacker, test_data, "transfer_based"
            )
        
        # 4. 集成攻击
        if self.config.use_ensemble_attack:
            logger.info("运行集成自适应攻击")
            all_results['ensemble_attack'] = self._run_ensemble_attack(
                target_pipeline, surrogate_pipelines, test_data
            )
        
        # 保存结果
        self._save_results(all_results)
        
        # 生成可视化
        if self.config.generate_visualizations:
            self._generate_visualizations(all_results)
        
        # 生成报告
        self._generate_attack_report(all_results)
        
        total_time = time.time() - start_time
        logger.info(f"自适应攻击实验完成，总耗时: {total_time:.2f} 秒")
        
        return all_results
    
    def _create_surrogate_models(self) -> List[MultiModalDetectionPipeline]:
        """
        创建代理模型
        """
        surrogate_configs = [
            # 可以使用不同的模型配置作为代理
            self.base_config.copy(),  # 相同配置但不同初始化
        ]
        
        surrogate_pipelines = []
        for config in surrogate_configs:
            try:
                pipeline = create_detection_pipeline(config)
                surrogate_pipelines.append(pipeline)
            except Exception as e:
                logger.warning(f"创建代理模型失败: {e}")
        
        return surrogate_pipelines
    
    def _run_attack_experiment(self, attacker: AdaptiveAttacker, 
                             test_data: List[Tuple], 
                             attack_type: str) -> Dict[str, Any]:
        """
        运行单个攻击实验
        
        Args:
            attacker: 攻击器实例
            test_data: 测试数据
            attack_type: 攻击类型
        
        Returns:
            攻击结果
        """
        results = []
        successful_attacks = 0
        total_samples = min(len(test_data), self.config.num_samples)
        
        for i, (image, text, _) in enumerate(tqdm(test_data[:total_samples], 
                                                  desc=f"{attack_type} attack")):
            try:
                # 执行攻击
                attack_result = attacker.attack(image, text)
                
                # 记录结果
                attack_result['sample_idx'] = i
                results.append(attack_result)
                
                if attack_result['success']:
                    successful_attacks += 1
                
                # 保存攻击样本（如果配置要求）
                if (self.config.save_attack_samples and 
                    attack_result['success'] and 
                    'adversarial_image' in attack_result):
                    self._save_attack_sample(attack_result, i, attack_type)
                
            except Exception as e:
                logger.error(f"攻击样本 {i} 失败: {e}")
                continue
        
        # 计算统计结果
        attack_success_rate = successful_attacks / len(results) if results else 0
        
        experiment_result = {
            'attack_type': attack_type,
            'total_samples': len(results),
            'successful_attacks': successful_attacks,
            'success_rate': attack_success_rate,
            'individual_results': results,
            'statistics': self._compute_attack_statistics(results)
        }
        
        return experiment_result
    
    def _run_ensemble_attack(self, target_pipeline: MultiModalDetectionPipeline,
                           surrogate_pipelines: List[MultiModalDetectionPipeline],
                           test_data: List[Tuple]) -> Dict[str, Any]:
        """
        运行集成攻击实验
        
        Args:
            target_pipeline: 目标管道
            surrogate_pipelines: 代理管道列表
            test_data: 测试数据
        
        Returns:
            集成攻击结果
        """
        results = []
        successful_attacks = 0
        total_samples = min(len(test_data), self.config.num_samples)
        
        for i, (image, text, _) in enumerate(tqdm(test_data[:total_samples], 
                                                  desc="ensemble attack")):
            try:
                # 创建多个攻击器
                attackers = [
                    GradientBasedAdaptiveAttacker(target_pipeline, self.base_config),
                    QueryBasedAdaptiveAttacker(target_pipeline, self.base_config)
                ]
                
                if surrogate_pipelines:
                    attackers.append(
                        TransferBasedAdaptiveAttacker(
                            target_pipeline, surrogate_pipelines, self.base_config
                        )
                    )
                
                # 执行多种攻击
                attack_results = []
                for attacker in attackers:
                    try:
                        result = attacker.attack(image, text)
                        attack_results.append(result)
                    except Exception as e:
                        logger.warning(f"集成攻击中的单个攻击失败: {e}")
                        continue
                
                # 选择最佳攻击结果
                best_result = self._select_best_attack(attack_results)
                
                if best_result:
                    best_result['sample_idx'] = i
                    best_result['attack_type'] = 'ensemble'
                    results.append(best_result)
                    
                    if best_result['success']:
                        successful_attacks += 1
                
            except Exception as e:
                logger.error(f"集成攻击样本 {i} 失败: {e}")
                continue
        
        # 计算统计结果
        attack_success_rate = successful_attacks / len(results) if results else 0
        
        experiment_result = {
            'attack_type': 'ensemble',
            'total_samples': len(results),
            'successful_attacks': successful_attacks,
            'success_rate': attack_success_rate,
            'individual_results': results,
            'statistics': self._compute_attack_statistics(results)
        }
        
        return experiment_result
    
    def _select_best_attack(self, attack_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        从多个攻击结果中选择最佳的
        
        Args:
            attack_results: 攻击结果列表
        
        Returns:
            最佳攻击结果
        """
        if not attack_results:
            return None
        
        # 优先选择成功的攻击
        successful_attacks = [r for r in attack_results if r.get('success', False)]
        
        if successful_attacks:
            # 在成功的攻击中选择扰动最小的
            best_attack = min(successful_attacks, 
                            key=lambda x: x.get('perturbation_norm', float('inf')))
            return best_attack
        else:
            # 如果都不成功，返回第一个
            return attack_results[0]
    
    def _compute_attack_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算攻击统计信息
        
        Args:
            results: 攻击结果列表
        
        Returns:
            统计信息字典
        """
        if not results:
            return {}
        
        # 提取数值指标
        execution_times = [r.get('execution_time', 0) for r in results]
        perturbation_norms = [r.get('perturbation_norm', 0) for r in results if 'perturbation_norm' in r]
        query_counts = [r.get('queries_used', 0) for r in results if 'queries_used' in r]
        
        statistics = {
            'mean_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            'total_execution_time': np.sum(execution_times)
        }
        
        if perturbation_norms:
            statistics.update({
                'mean_perturbation_norm': np.mean(perturbation_norms),
                'std_perturbation_norm': np.std(perturbation_norms),
                'min_perturbation_norm': np.min(perturbation_norms),
                'max_perturbation_norm': np.max(perturbation_norms)
            })
        
        if query_counts:
            statistics.update({
                'mean_queries': np.mean(query_counts),
                'std_queries': np.std(query_counts),
                'total_queries': np.sum(query_counts)
            })
        
        return statistics
    
    def _save_attack_sample(self, attack_result: Dict[str, Any], 
                          sample_idx: int, attack_type: str):
        """
        保存攻击样本
        
        Args:
            attack_result: 攻击结果
            sample_idx: 样本索引
            attack_type: 攻击类型
        """
        if 'adversarial_image' in attack_result:
            sample_dir = self.output_dir / "attack_samples" / attack_type
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存对抗图像
            image_path = sample_dir / f"sample_{sample_idx}_adversarial.png"
            attack_result['adversarial_image'].save(image_path)
    
    def _save_results(self, results: Dict[str, Any]):
        """
        保存实验结果
        
        Args:
            results: 实验结果字典
        """
        # 保存完整结果
        results_file = self.output_dir / "adaptive_attack_results.json"
        
        # 处理不可序列化的对象
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存汇总表格
        summary_data = []
        for attack_type, attack_results in results.items():
            if isinstance(attack_results, dict) and 'statistics' in attack_results:
                row = {
                    'attack_type': attack_type,
                    'success_rate': attack_results.get('success_rate', 0),
                    'total_samples': attack_results.get('total_samples', 0),
                    'successful_attacks': attack_results.get('successful_attacks', 0),
                    **attack_results.get('statistics', {})
                }
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.output_dir / "attack_summary.csv"
            summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"结果已保存到: {results_file}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        将对象转换为可序列化格式
        
        Args:
            obj: 输入对象
        
        Returns:
            可序列化对象
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, Image.Image):
            return f"<PIL.Image {obj.size}>"
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """
        生成可视化图表
        
        Args:
            results: 实验结果字典
        """
        logger.info("生成可视化图表")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 攻击成功率对比
        self._plot_success_rates(results)
        
        # 2. 执行时间对比
        self._plot_execution_times(results)
        
        # 3. 扰动强度分布
        self._plot_perturbation_distributions(results)
        
        # 4. 查询次数分析（针对查询攻击）
        self._plot_query_analysis(results)
        
        logger.info(f"可视化图表已保存到: {self.output_dir}")
    
    def _plot_success_rates(self, results: Dict[str, Any]):
        """
        绘制攻击成功率对比图
        """
        attack_types = []
        success_rates = []
        
        for attack_type, attack_results in results.items():
            if isinstance(attack_results, dict) and 'success_rate' in attack_results:
                attack_types.append(attack_type.replace('_', ' ').title())
                success_rates.append(attack_results['success_rate'])
        
        if attack_types:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(attack_types, success_rates)
            plt.title('Attack Success Rates Comparison')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            
            # 添加数值标签
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'success_rates_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_execution_times(self, results: Dict[str, Any]):
        """
        绘制执行时间对比图
        """
        attack_types = []
        mean_times = []
        std_times = []
        
        for attack_type, attack_results in results.items():
            if (isinstance(attack_results, dict) and 
                'statistics' in attack_results and 
                'mean_execution_time' in attack_results['statistics']):
                
                attack_types.append(attack_type.replace('_', ' ').title())
                mean_times.append(attack_results['statistics']['mean_execution_time'])
                std_times.append(attack_results['statistics'].get('std_execution_time', 0))
        
        if attack_types:
            plt.figure(figsize=(10, 6))
            plt.bar(attack_types, mean_times, yerr=std_times, capsize=5)
            plt.title('Attack Execution Times Comparison')
            plt.ylabel('Execution Time (seconds)')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'execution_times_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_perturbation_distributions(self, results: Dict[str, Any]):
        """
        绘制扰动强度分布图
        """
        plt.figure(figsize=(12, 8))
        
        for i, (attack_type, attack_results) in enumerate(results.items()):
            if (isinstance(attack_results, dict) and 
                'individual_results' in attack_results):
                
                perturbations = []
                for result in attack_results['individual_results']:
                    if 'perturbation_norm' in result:
                        perturbations.append(result['perturbation_norm'])
                
                if perturbations:
                    plt.subplot(2, 2, i + 1)
                    plt.hist(perturbations, bins=20, alpha=0.7)
                    plt.title(f'{attack_type.replace("_", " ").title()} Perturbation Distribution')
                    plt.xlabel('Perturbation Norm')
                    plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'perturbation_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_query_analysis(self, results: Dict[str, Any]):
        """
        绘制查询次数分析图
        """
        query_based_results = results.get('query_based')
        if (query_based_results and 
            isinstance(query_based_results, dict) and 
            'individual_results' in query_based_results):
            
            query_counts = []
            success_flags = []
            
            for result in query_based_results['individual_results']:
                if 'queries_used' in result:
                    query_counts.append(result['queries_used'])
                    success_flags.append(result.get('success', False))
            
            if query_counts:
                plt.figure(figsize=(12, 5))
                
                # 查询次数分布
                plt.subplot(1, 2, 1)
                plt.hist(query_counts, bins=20, alpha=0.7)
                plt.title('Query Count Distribution')
                plt.xlabel('Number of Queries')
                plt.ylabel('Frequency')
                
                # 成功率与查询次数关系
                plt.subplot(1, 2, 2)
                successful_queries = [q for q, s in zip(query_counts, success_flags) if s]
                failed_queries = [q for q, s in zip(query_counts, success_flags) if not s]
                
                plt.hist([successful_queries, failed_queries], 
                        bins=20, alpha=0.7, label=['Successful', 'Failed'])
                plt.title('Query Count vs Success')
                plt.xlabel('Number of Queries')
                plt.ylabel('Frequency')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'query_analysis.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def _generate_attack_report(self, results: Dict[str, Any]):
        """
        生成攻击实验报告
        
        Args:
            results: 实验结果字典
        """
        logger.info("生成攻击实验报告")
        
        report_lines = [
            "# 多模态检测一致性系统自适应攻击实验报告\n\n",
            f"**实验时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**实验配置**: {self.config.experiment_name}\n",
            f"**测试样本数**: {self.config.num_samples}\n",
            f"**攻击预算**: {self.config.attack_budget}\n\n",
            "## 实验概述\n\n",
            "本报告展示了针对多模态检测一致性系统的自适应攻击实验结果，",
            "评估了系统对不同类型攻击的鲁棒性。\n\n"
        ]
        
        # 生成攻击结果汇总
        report_lines.append("## 攻击结果汇总\n\n")
        report_lines.append("| 攻击类型 | 成功率 | 平均执行时间 | 平均扰动强度 | 平均查询次数 |\n")
        report_lines.append("|----------|--------|--------------|--------------|--------------|\n")
        
        for attack_type, attack_results in results.items():
            if isinstance(attack_results, dict):
                success_rate = attack_results.get('success_rate', 0)
                stats = attack_results.get('statistics', {})
                
                mean_time = stats.get('mean_execution_time', 0)
                mean_perturbation = stats.get('mean_perturbation_norm', 'N/A')
                mean_queries = stats.get('mean_queries', 'N/A')
                
                report_lines.append(
                    f"| {attack_type.replace('_', ' ').title()} | "
                    f"{success_rate:.3f} | {mean_time:.3f}s | "
                    f"{mean_perturbation if isinstance(mean_perturbation, str) else f'{mean_perturbation:.4f}'} | "
                    f"{mean_queries if isinstance(mean_queries, str) else f'{mean_queries:.1f}'} |\n"
                )
        
        report_lines.append("\n")
        
        # 生成详细分析
        for attack_type, attack_results in results.items():
            if isinstance(attack_results, dict):
                report_lines.extend(self._generate_attack_analysis(attack_type, attack_results))
        
        # 生成结论和建议
        report_lines.extend(self._generate_attack_conclusions(results))
        
        # 保存报告
        report_file = self.output_dir / "adaptive_attack_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        logger.info(f"攻击实验报告已保存到: {report_file}")
    
    def _generate_attack_analysis(self, attack_type: str, 
                                attack_results: Dict[str, Any]) -> List[str]:
        """
        生成单个攻击类型的详细分析
        
        Args:
            attack_type: 攻击类型
            attack_results: 攻击结果
        
        Returns:
            分析文本行列表
        """
        lines = [
            f"### {attack_type.replace('_', ' ').title()} 攻击分析\n\n"
        ]
        
        success_rate = attack_results.get('success_rate', 0)
        total_samples = attack_results.get('total_samples', 0)
        successful_attacks = attack_results.get('successful_attacks', 0)
        
        lines.append(f"- **总样本数**: {total_samples}\n")
        lines.append(f"- **成功攻击数**: {successful_attacks}\n")
        lines.append(f"- **攻击成功率**: {success_rate:.3f}\n")
        
        stats = attack_results.get('statistics', {})
        if stats:
            lines.append(f"- **平均执行时间**: {stats.get('mean_execution_time', 0):.3f} 秒\n")
            
            if 'mean_perturbation_norm' in stats:
                lines.append(f"- **平均扰动强度**: {stats['mean_perturbation_norm']:.4f}\n")
            
            if 'mean_queries' in stats:
                lines.append(f"- **平均查询次数**: {stats['mean_queries']:.1f}\n")
        
        lines.append("\n")
        
        return lines
    
    def _generate_attack_conclusions(self, results: Dict[str, Any]) -> List[str]:
        """
        生成攻击实验结论
        
        Args:
            results: 所有攻击结果
        
        Returns:
            结论文本行列表
        """
        lines = [
            "## 实验结论\n\n",
            "基于自适应攻击实验的结果，我们得出以下结论:\n\n"
        ]
        
        # 找出最有效的攻击方法
        best_attack = None
        best_success_rate = 0
        
        for attack_type, attack_results in results.items():
            if isinstance(attack_results, dict):
                success_rate = attack_results.get('success_rate', 0)
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_attack = attack_type
        
        if best_attack:
            lines.append(f"- **最有效的攻击方法**: {best_attack.replace('_', ' ').title()} "
                        f"(成功率: {best_success_rate:.3f})\n")
        
        # 系统鲁棒性评估
        overall_success_rate = np.mean([
            r.get('success_rate', 0) for r in results.values() 
            if isinstance(r, dict) and 'success_rate' in r
        ])
        
        if overall_success_rate < 0.3:
            robustness_level = "高"
        elif overall_success_rate < 0.6:
            robustness_level = "中等"
        else:
            robustness_level = "低"
        
        lines.append(f"- **系统整体鲁棒性**: {robustness_level} (平均攻击成功率: {overall_success_rate:.3f})\n")
        
        lines.extend([
            "\n",
            "## 安全建议\n\n",
            "基于攻击实验结果，我们建议:\n\n",
            "1. 加强对梯度信息的保护，防止基于梯度的攻击\n",
            "2. 限制模型查询频率，降低查询攻击的有效性\n",
            "3. 使用对抗训练提高模型鲁棒性\n",
            "4. 实施输入验证和异常检测机制\n",
            "5. 定期更新模型以应对新的攻击方法\n\n",
            "---\n\n",
            f"**报告生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "**作者**: 张昕 (zhang.xin@duke.edu)\n",
            "**学校**: Duke University\n"
        ])
        
        return lines


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行多模态检测一致性系统自适应攻击实验",
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
        default='./results/adaptive_attack',
        help='输出目录'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='测试样本数量'
    )
    
    parser.add_argument(
        '--attack-budget',
        type=float,
        default=0.1,
        help='攻击预算（扰动强度）'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['coco', 'flickr30k'],
        default='coco',
        help='使用的数据集'
    )
    
    parser.add_argument(
        '--attack-types',
        nargs='+',
        choices=['gradient_based', 'query_based', 'transfer_based', 'ensemble_attack'],
        default=['gradient_based', 'query_based', 'transfer_based', 'ensemble_attack'],
        help='要测试的攻击类型'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='不生成可视化图表'
    )
    
    parser.add_argument(
        '--save-samples',
        action='store_true',
        help='保存攻击样本'
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
        
        # 创建攻击实验配置
        attack_config = AdaptiveAttackConfig(
            experiment_name="multimodal_detection_adaptive_attack",
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            attack_budget=args.attack_budget,
            use_gradient_based='gradient_based' in args.attack_types,
            use_query_based='query_based' in args.attack_types,
            use_transfer_based='transfer_based' in args.attack_types,
            use_ensemble_attack='ensemble_attack' in args.attack_types,
            generate_visualizations=not args.no_visualizations,
            save_attack_samples=args.save_samples
        )
        
        # 加载测试数据
        logger.info(f"加载 {args.dataset} 数据集")
        data_loader_manager = DataLoaderManager(base_config)
        
        if args.dataset == 'coco':
            test_data = data_loader_manager.load_coco_data('test')
        elif args.dataset == 'flickr30k':
            test_data = data_loader_manager.load_flickr30k_data('test')
        else:
            raise ValueError(f"不支持的数据集: {args.dataset}")
        
        # 创建实验管理器
        experiment_manager = AdaptiveAttackExperimentManager(attack_config, base_config)
        
        # 运行攻击实验
        logger.info("开始运行自适应攻击实验")
        results = experiment_manager.run_adaptive_attack_experiments(test_data)
        
        # 输出结果摘要
        logger.info("\n=== 自适应攻击实验结果摘要 ===")
        for attack_type, attack_results in results.items():
            if isinstance(attack_results, dict) and 'success_rate' in attack_results:
                logger.info(f"{attack_type}: 成功率 {attack_results['success_rate']:.3f}")
        
        logger.info(f"详细结果已保存到: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("实验被用户中断")
    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())