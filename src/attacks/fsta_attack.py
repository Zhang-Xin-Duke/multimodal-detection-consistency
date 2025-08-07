"""FSTA (Feature Space Targeted Attack) 攻击实现

基于特征空间目标攻击的对抗样本生成方法。
参考论文：Feature Space Targeted Attack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FSTAAttackConfig:
    """FSTA攻击配置
    
    基于参考实现的配置参数
    """
    # 基础设置
    random_seed: int = 42
    device: str = 'cuda'
    
    # FSTA核心参数
    epsilon: float = 0.031  # 扰动幅度 (8/255)
    alpha: float = 0.008    # 步长 (2/255)
    num_iter: int = 20      # 迭代次数
    
    # 特征空间参数
    feature_layer: str = "penultimate"  # 目标特征层
    target_feature_method: str = "centroid"  # 目标特征选择方法
    feature_distance_metric: str = "cosine"  # 特征距离度量
    
    # 目标设置
    targeted: bool = True
    target_selection: str = "random"  # 目标选择策略
    num_targets: int = 10
    
    # 损失权重
    feature_weight: float = 1.0
    output_weight: float = 0.1
    diversity_weight: float = 0.05
    
    # 优化参数
    momentum: float = 0.9
    decay_factor: float = 0.99
    learning_rate: float = 0.01
    
    # 约束参数
    norm_type: str = "inf"  # 范数类型
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    
    # 高级参数
    adaptive_step_size: bool = True
    gradient_clipping: bool = True
    clip_value: float = 1.0
    
    # 评估参数
    batch_size: int = 32
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000


class FSTAAttacker:
    """FSTA攻击器
    
    实现基于特征空间的目标攻击方法
    """
    
    def __init__(self, clip_model, config: FSTAAttackConfig):
        """
        初始化FSTA攻击器
        
        Args:
            clip_model: CLIP模型
            config: 攻击配置
        """
        self.clip_model = clip_model
        self.config = config
        self.device = torch.device(config.device)
        
        # 攻击统计
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'avg_iterations': 0,
            'avg_perturbation_norm': 0.0
        }
        
        # 缓存
        if config.enable_cache:
            self.cache = {}
            self.cache_keys = []
        
        logger.info(f"初始化FSTA攻击器，配置: {config}")
    
    def attack(self, images: torch.Tensor, texts: List[str], 
               target_texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        执行FSTA攻击
        
        Args:
            images: 输入图像 [batch_size, 3, H, W]
            texts: 文本查询列表
            target_texts: 目标文本列表（用于目标攻击）
            
        Returns:
            攻击结果字典
        """
        batch_size = images.size(0)
        
        # 编码文本特征
        with torch.no_grad():
            text_features = self.clip_model.encode_text(texts)
            if target_texts:
                target_features = self.clip_model.encode_text(target_texts)
            else:
                # 随机选择目标特征
                target_features = self._generate_random_targets(text_features)
        
        # 执行攻击
        adv_images = self._fsta_attack(
            images, text_features, target_features
        )
        
        # 计算攻击成功率
        success_rate = self._evaluate_attack_success(
            adv_images, texts, target_texts
        )
        
        # 更新统计信息
        self.attack_stats['total_attacks'] += batch_size
        self.attack_stats['successful_attacks'] += int(success_rate * batch_size)
        
        # 缓存结果
        if self.config.enable_cache:
            self._cache_results(images, adv_images, texts)
        
        return {
            'adversarial_images': adv_images,
            'original_images': images,
            'success_rate': success_rate,
            'perturbation_norm': torch.norm(
                adv_images - images, p=float('inf'), dim=(1,2,3)
            ).mean().item()
        }
    
    def _fsta_attack(self, images: torch.Tensor, text_features: torch.Tensor,
                     target_features: torch.Tensor) -> torch.Tensor:
        """
        执行FSTA攻击的核心逻辑
        
        Args:
            images: 原始图像
            text_features: 文本特征
            target_features: 目标特征
            
        Returns:
            对抗样本
        """
        adv_images = images.clone().detach().to(self.device)
        adv_images.requires_grad_(True)
        
        # 初始化动量
        momentum = torch.zeros_like(adv_images)
        
        # 计算扰动边界
        if self.config.norm_type == "inf":
            delta_min = -self.config.epsilon
            delta_max = self.config.epsilon
        else:
            delta_min = delta_max = self.config.epsilon
        
        # 自适应学习率
        current_lr = self.config.learning_rate
        
        for iteration in range(self.config.num_iter):
            # 前向传播
            image_features = self.clip_model.encode_image(adv_images)
            
            # 计算特征空间损失
            feature_loss = self._compute_feature_loss(
                image_features, text_features, target_features
            )
            
            # 计算输出损失
            output_loss = self._compute_output_loss(
                image_features, target_features
            )
            
            # 计算多样性损失
            diversity_loss = self._compute_diversity_loss(image_features)
            
            # 总损失
            total_loss = (
                self.config.feature_weight * feature_loss +
                self.config.output_weight * output_loss +
                self.config.diversity_weight * diversity_loss
            )
            
            # 反向传播
            total_loss.backward()
            
            # 获取梯度
            grad = adv_images.grad.data
            
            # 梯度裁剪
            if self.config.gradient_clipping:
                grad = torch.clamp(grad, -self.config.clip_value, self.config.clip_value)
            
            # 动量更新
            momentum = self.config.momentum * momentum + grad
            
            # 更新对抗样本
            if self.config.norm_type == "inf":
                adv_images.data = adv_images.data - current_lr * momentum.sign()
                # L∞约束
                delta = torch.clamp(
                    adv_images.data - images, delta_min, delta_max
                )
                adv_images.data = images + delta
            else:
                # L2约束
                adv_images.data = adv_images.data - current_lr * momentum
                delta = adv_images.data - images
                delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
                factor = torch.min(torch.ones_like(delta_norm), 
                                 self.config.epsilon / (delta_norm + 1e-8))
                delta = delta * factor.view(-1, 1, 1, 1)
                adv_images.data = images + delta
            
            # 像素值约束
            adv_images.data = torch.clamp(
                adv_images.data, self.config.clamp_min, self.config.clamp_max
            )
            
            # 清零梯度
            adv_images.grad.zero_()
            
            # 自适应学习率衰减
            if self.config.adaptive_step_size:
                current_lr *= self.config.decay_factor
        
        return adv_images.detach()
    
    def _compute_feature_loss(self, image_features: torch.Tensor,
                             text_features: torch.Tensor,
                             target_features: torch.Tensor) -> torch.Tensor:
        """
        计算特征空间损失
        """
        if self.config.feature_distance_metric == "cosine":
            # 最大化与目标特征的相似度，最小化与原文本特征的相似度
            target_sim = F.cosine_similarity(image_features, target_features, dim=1)
            text_sim = F.cosine_similarity(image_features, text_features, dim=1)
            return -target_sim.mean() + text_sim.mean()
        elif self.config.feature_distance_metric == "euclidean":
            target_dist = torch.norm(image_features - target_features, p=2, dim=1)
            text_dist = torch.norm(image_features - text_features, p=2, dim=1)
            return target_dist.mean() - text_dist.mean()
        else:
            raise ValueError(f"不支持的距离度量: {self.config.feature_distance_metric}")
    
    def _compute_output_loss(self, image_features: torch.Tensor,
                           target_features: torch.Tensor) -> torch.Tensor:
        """
        计算输出损失
        """
        return F.mse_loss(image_features, target_features)
    
    def _compute_diversity_loss(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        计算多样性损失，鼓励生成多样化的对抗样本
        """
        batch_size = image_features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=image_features.device)
        
        # 计算特征之间的相似度矩阵
        sim_matrix = F.cosine_similarity(
            image_features.unsqueeze(1), image_features.unsqueeze(0), dim=2
        )
        
        # 排除对角线元素
        mask = torch.eye(batch_size, device=image_features.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0)
        
        # 返回负的平均相似度（鼓励多样性）
        return sim_matrix.sum() / (batch_size * (batch_size - 1))
    
    def _generate_random_targets(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        生成随机目标特征
        """
        batch_size, feature_dim = text_features.shape
        
        if self.config.target_selection == "random":
            # 生成随机单位向量
            random_features = torch.randn(batch_size, feature_dim, device=self.device)
            random_features = F.normalize(random_features, p=2, dim=1)
            return random_features
        elif self.config.target_selection == "centroid":
            # 使用特征中心点的反方向
            centroid = text_features.mean(dim=0, keepdim=True)
            return -F.normalize(centroid, p=2, dim=1).repeat(batch_size, 1)
        else:
            raise ValueError(f"不支持的目标选择策略: {self.config.target_selection}")
    
    def _evaluate_attack_success(self, adv_images: torch.Tensor, texts: List[str],
                               target_texts: Optional[List[str]] = None) -> float:
        """
        评估攻击成功率
        """
        with torch.no_grad():
            # 编码对抗图像特征
            adv_image_features = self.clip_model.encode_image(adv_images)
            
            # 编码文本特征
            text_features = self.clip_model.encode_text(texts)
            
            # 计算相似度
            similarities = F.cosine_similarity(
                adv_image_features, text_features, dim=1
            )
            
            # 如果是目标攻击，检查是否成功欺骗到目标
            if target_texts:
                target_features = self.clip_model.encode_text(target_texts)
                target_similarities = F.cosine_similarity(
                    adv_image_features, target_features, dim=1
                )
                # 攻击成功：与目标相似度高于与原文本相似度
                success = (target_similarities > similarities).float()
            else:
                # 非目标攻击：相似度显著降低
                success = (similarities < 0.5).float()  # 阈值可调
            
            return success.mean().item()
    
    def _cache_results(self, images: torch.Tensor, adv_images: torch.Tensor,
                      texts: List[str]):
        """
        缓存攻击结果
        """
        if len(self.cache_keys) >= self.config.cache_size:
            # 移除最旧的缓存
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]
        
        # 生成缓存键
        cache_key = hash((tuple(images.flatten().tolist()), tuple(texts)))
        
        # 存储结果
        self.cache[cache_key] = {
            'adversarial_images': adv_images.clone(),
            'original_images': images.clone(),
            'texts': texts.copy()
        }
        self.cache_keys.append(cache_key)
    
    def get_attack_stats(self) -> Dict[str, float]:
        """
        获取攻击统计信息
        """
        stats = self.attack_stats.copy()
        if stats['total_attacks'] > 0:
            stats['success_rate'] = stats['successful_attacks'] / stats['total_attacks']
        else:
            stats['success_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """
        重置攻击统计信息
        """
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'avg_iterations': 0,
            'avg_perturbation_norm': 0.0
        }


def create_fsta_attacker(clip_model, config: Optional[FSTAAttackConfig] = None) -> FSTAAttacker:
    """
    创建FSTA攻击器的工厂函数
    
    Args:
        clip_model: CLIP模型
        config: 攻击配置，如果为None则使用默认配置
        
    Returns:
        FSTA攻击器实例
    """
    if config is None:
        config = FSTAAttackConfig()
    
    return FSTAAttacker(clip_model, config)


class FSTAAttackPresets:
    """
    FSTA攻击预设配置
    """
    
    @staticmethod
    def weak_attack() -> FSTAAttackConfig:
        """弱攻击配置"""
        return FSTAAttackConfig(
            epsilon=0.016,  # 4/255
            num_iter=10,
            learning_rate=0.005
        )
    
    @staticmethod
    def strong_attack() -> FSTAAttackConfig:
        """强攻击配置"""
        return FSTAAttackConfig(
            epsilon=0.047,  # 12/255
            num_iter=40,
            learning_rate=0.02
        )
    
    @staticmethod
    def targeted_attack() -> FSTAAttackConfig:
        """目标攻击配置"""
        return FSTAAttackConfig(
            targeted=True,
            target_selection="centroid",
            feature_weight=2.0,
            num_iter=30
        )
    
    @staticmethod
    def paper_config() -> FSTAAttackConfig:
        """论文中的配置"""
        return FSTAAttackConfig(
            epsilon=0.031,  # 8/255
            alpha=0.008,    # 2/255
            num_iter=20,
            feature_layer="penultimate",
            feature_distance_metric="cosine"
        )