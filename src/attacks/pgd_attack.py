"""PGD图像攻击模块

实现基于PGD (Projected Gradient Descent) 的图像对抗性攻击方法。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from PIL import Image
import torchvision.transforms as transforms

@dataclass
class PGDAttackConfig:
    """PGD攻击配置"""
    # 基础设置
    random_seed: int = 42
    device: str = 'cuda'
    
    # PGD参数
    epsilon: float = 8.0 / 255.0  # 最大扰动幅度
    alpha: float = 2.0 / 255.0    # 步长
    num_steps: int = 10           # 迭代步数
    
    # 攻击目标
    targeted: bool = False        # 是否为目标攻击
    
    # 约束参数
    clip_min: float = 0.0
    clip_max: float = 1.0
    
    # 优化参数
    momentum: float = 0.9         # 动量系数
    use_momentum: bool = True     # 是否使用动量
    
    # 评估参数
    batch_size: int = 32
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000

class PGDAttacker:
    """PGD攻击器"""
    
    def __init__(self, clip_model, config: PGDAttackConfig):
        """
        初始化PGD攻击器
        
        Args:
            clip_model: CLIP模型实例
            config: 攻击配置
        """
        self.clip_model = clip_model
        self.config = config
        self.device = torch.device(config.device)
        
        # 设置随机种子
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # 初始化统计信息
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'average_perturbation': 0.0,
            'average_iterations': 0.0
        }
        
        # 缓存
        self.cache = {} if config.enable_cache else None
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 反归一化
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        logging.info(f"PGD攻击器初始化完成，配置: {config}")
    
    def attack(self, image: Union[Image.Image, torch.Tensor], 
               text: str, 
               target_text: Optional[str] = None) -> Dict[str, Any]:
        """
        执行PGD攻击
        
        Args:
            image: 输入图像
            text: 原始文本
            target_text: 目标文本（目标攻击时使用）
        
        Returns:
            攻击结果字典
        """
        # 检查缓存
        cache_key = self._get_cache_key(image, text, target_text)
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 转换输入格式
        if isinstance(image, Image.Image):
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
        
        # 编码文本
        with torch.no_grad():
            text_features = self.clip_model.encode_text([text])
            if target_text:
                target_features = self.clip_model.encode_text([target_text])
            else:
                target_features = None
        
        # 执行PGD攻击
        adversarial_image, attack_info = self._pgd_attack(
            image_tensor, text_features, target_features
        )
        
        # 计算攻击成功率
        success = self._evaluate_attack_success(
            adversarial_image, text_features, target_features
        )
        
        # 更新统计信息
        self._update_stats(attack_info, success)
        
        result = {
            'adversarial_image': adversarial_image,
            'original_image': image_tensor,
            'perturbation': adversarial_image - image_tensor,
            'success': success,
            'attack_info': attack_info,
            'config': self.config
        }
        
        # 缓存结果
        if self.cache and len(self.cache) < self.config.cache_size:
            self.cache[cache_key] = result
        
        return result
    
    def _pgd_attack(self, image: torch.Tensor, 
                    text_features: torch.Tensor,
                    target_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        执行PGD攻击的核心逻辑
        
        Args:
            image: 输入图像张量
            text_features: 文本特征
            target_features: 目标文本特征（可选）
        
        Returns:
            对抗样本和攻击信息
        """
        # 初始化对抗样本
        adv_image = image.clone().detach()
        
        # 添加随机噪声
        if self.config.num_steps > 1:
            noise = torch.empty_like(adv_image).uniform_(
                -self.config.epsilon, self.config.epsilon
            )
            adv_image = torch.clamp(adv_image + noise, 
                                  self.config.clip_min, self.config.clip_max)
        
        # 初始化动量
        momentum = torch.zeros_like(adv_image) if self.config.use_momentum else None
        
        attack_info = {
            'iterations': 0,
            'loss_history': [],
            'similarity_history': [],
            'perturbation_norm': 0.0
        }
        
        for step in range(self.config.num_steps):
            adv_image.requires_grad_(True)
            
            # 前向传播
            adv_features = self.clip_model.encode_image(adv_image)
            
            # 计算损失
            if self.config.targeted and target_features is not None:
                # 目标攻击：最大化与目标文本的相似度
                loss = -F.cosine_similarity(adv_features, target_features).mean()
            else:
                # 非目标攻击：最小化与原始文本的相似度
                loss = F.cosine_similarity(adv_features, text_features).mean()
            
            # 反向传播
            loss.backward()
            
            # 获取梯度
            grad = adv_image.grad.data
            
            # 应用动量
            if self.config.use_momentum and momentum is not None:
                momentum = self.config.momentum * momentum + grad / torch.norm(grad, p=1)
                grad = momentum
            
            # 更新对抗样本
            if self.config.targeted:
                adv_image = adv_image.detach() - self.config.alpha * grad.sign()
            else:
                adv_image = adv_image.detach() + self.config.alpha * grad.sign()
            
            # 投影到约束集合
            delta = torch.clamp(adv_image - image, 
                              -self.config.epsilon, self.config.epsilon)
            adv_image = torch.clamp(image + delta, 
                                  self.config.clip_min, self.config.clip_max)
            
            # 记录统计信息
            with torch.no_grad():
                current_features = self.clip_model.encode_image(adv_image)
                similarity = F.cosine_similarity(current_features, text_features).mean().item()
                
                attack_info['loss_history'].append(loss.item())
                attack_info['similarity_history'].append(similarity)
            
            attack_info['iterations'] = step + 1
        
        # 计算最终扰动范数
        perturbation = adv_image - image
        attack_info['perturbation_norm'] = torch.norm(perturbation).item()
        
        return adv_image.detach(), attack_info
    
    def _evaluate_attack_success(self, adversarial_image: torch.Tensor,
                               text_features: torch.Tensor,
                               target_features: Optional[torch.Tensor] = None) -> bool:
        """
        评估攻击是否成功
        
        Args:
            adversarial_image: 对抗样本
            text_features: 原始文本特征
            target_features: 目标文本特征
        
        Returns:
            攻击是否成功
        """
        with torch.no_grad():
            adv_features = self.clip_model.encode_image(adversarial_image)
            
            if self.config.targeted and target_features is not None:
                # 目标攻击：检查是否与目标文本更相似
                target_sim = F.cosine_similarity(adv_features, target_features).mean().item()
                original_sim = F.cosine_similarity(adv_features, text_features).mean().item()
                return target_sim > original_sim
            else:
                # 非目标攻击：检查相似度是否显著下降
                similarity = F.cosine_similarity(adv_features, text_features).mean().item()
                return similarity < 0.5  # 阈值可调
    
    def batch_attack(self, images: List[Union[Image.Image, torch.Tensor]], 
                    texts: List[str], 
                    target_texts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        批量攻击
        
        Args:
            images: 图像列表
            texts: 文本列表
            target_texts: 目标文本列表
        
        Returns:
            攻击结果列表
        """
        results = []
        
        for i, (image, text) in enumerate(zip(images, texts)):
            target_text = target_texts[i] if target_texts else None
            result = self.attack(image, text, target_text)
            results.append(result)
        
        return results
    
    def _get_cache_key(self, image: Union[Image.Image, torch.Tensor], 
                      text: str, target_text: Optional[str] = None) -> str:
        """
        生成缓存键
        
        Args:
            image: 输入图像
            text: 文本
            target_text: 目标文本
        
        Returns:
            缓存键
        """
        # 简化的缓存键生成
        image_hash = hash(str(image)) if isinstance(image, Image.Image) else hash(image.data.tobytes())
        text_hash = hash(text)
        target_hash = hash(target_text) if target_text else 0
        
        return f"pgd_{image_hash}_{text_hash}_{target_hash}_{self.config.epsilon}_{self.config.num_steps}"
    
    def _update_stats(self, attack_info: Dict, success: bool):
        """
        更新攻击统计信息
        
        Args:
            attack_info: 攻击信息
            success: 攻击是否成功
        """
        self.attack_stats['total_attacks'] += 1
        if success:
            self.attack_stats['successful_attacks'] += 1
        
        # 更新平均值
        total = self.attack_stats['total_attacks']
        self.attack_stats['average_perturbation'] = (
            (self.attack_stats['average_perturbation'] * (total - 1) + 
             attack_info['perturbation_norm']) / total
        )
        self.attack_stats['average_iterations'] = (
            (self.attack_stats['average_iterations'] * (total - 1) + 
             attack_info['iterations']) / total
        )
    
    def get_attack_stats(self) -> Dict[str, Any]:
        """
        获取攻击统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.attack_stats.copy()
        if stats['total_attacks'] > 0:
            stats['success_rate'] = stats['successful_attacks'] / stats['total_attacks']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'average_perturbation': 0.0,
            'average_iterations': 0.0
        }
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()

def create_pgd_attacker(clip_model, config: Optional[PGDAttackConfig] = None) -> PGDAttacker:
    """
    创建PGD攻击器的工厂函数
    
    Args:
        clip_model: CLIP模型实例
        config: 攻击配置，如果为None则使用默认配置
    
    Returns:
        PGD攻击器实例
    """
    if config is None:
        config = PGDAttackConfig()
    
    return PGDAttacker(clip_model, config)