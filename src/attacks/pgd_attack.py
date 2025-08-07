"""PGD图像攻击模块

实现基于PGD (Projected Gradient Descent) 的图像对抗性攻击方法。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import time
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

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
    
    # 多GPU和批处理优化配置
    enable_multi_gpu: bool = False        # 是否启用多GPU
    gpu_ids: Optional[List[int]] = None   # 指定GPU ID列表，None表示使用所有可用GPU
    batch_size_per_gpu: int = 8           # 每个GPU的批次大小
    num_workers: int = 4                  # 数据加载器工作进程数
    
    # 内存优化配置
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    mixed_precision: bool = False         # 是否使用混合精度
    pin_memory: bool = True               # 是否使用固定内存
    gradient_clip_value: float = 0.0      # 梯度裁剪值，0表示不裁剪

class PGDAttacker:
    """PGD攻击器"""
    
    def __init__(self, clip_model, config: PGDAttackConfig):
        """
        初始化PGD攻击器
        
        Args:
            clip_model: CLIP模型实例
            config: 攻击配置
        """
        self.config = config
        
        # 设置设备和多GPU配置
        self._setup_devices()
        
        # 初始化CLIP模型
        self._initialize_clip_model(clip_model)
        
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
        
        logging.info(f"PGD攻击器初始化完成，设备: {self.device}")
        logging.info(f"多GPU配置: 启用={config.enable_multi_gpu}, GPU数量={len(self.device_ids)}")
        logging.info(f"批处理配置: 总批次大小={config.batch_size}, 每GPU批次大小={config.batch_size_per_gpu}")
    
    def _setup_devices(self):
        """设置设备和多GPU配置"""
        if self.config.enable_multi_gpu and torch.cuda.device_count() > 1:
            if self.config.gpu_ids is None:
                self.device_ids = list(range(torch.cuda.device_count()))
            else:
                self.device_ids = self.config.gpu_ids
            
            self.device = torch.device(f'cuda:{self.device_ids[0]}')
        else:
            self.device_ids = [0] if torch.cuda.is_available() else []
            self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
    
    def _initialize_clip_model(self, clip_model):
        """初始化CLIP模型"""
        self.clip_model = clip_model
        
        # 设置多GPU支持
        if self.config.enable_multi_gpu and len(self.device_ids) > 1:
            if hasattr(self.clip_model, 'model'):
                self.clip_model.model = nn.DataParallel(
                    self.clip_model.model, device_ids=self.device_ids
                )
            else:
                self.clip_model = nn.DataParallel(
                    self.clip_model, device_ids=self.device_ids
                )
        
        # 将模型移动到主设备
        if hasattr(self.clip_model, 'to'):
            self.clip_model = self.clip_model.to(self.device)
    
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
            # 确保文本特征在正确的设备上
            if isinstance(text_features, torch.Tensor):
                text_features = text_features.to(self.device)
            
            if target_text:
                target_features = self.clip_model.encode_text([target_text])
                # 确保目标文本特征在正确的设备上
                if isinstance(target_features, torch.Tensor):
                    target_features = target_features.to(self.device)
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
            
            # 前向传播（启用梯度计算）
            if hasattr(self.clip_model, 'encode_image_tensor'):
                adv_features = self.clip_model.encode_image_tensor(adv_image, requires_grad=True)
            else:
                adv_features = self.clip_model.encode_image(adv_image)
            
            # 确保图像特征在正确的设备上
            if isinstance(adv_features, torch.Tensor):
                adv_features = adv_features.to(self.device)
            
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
                if hasattr(self.clip_model, 'encode_image_tensor'):
                    current_features = self.clip_model.encode_image_tensor(adv_image, requires_grad=False)
                else:
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
            if hasattr(self.clip_model, 'encode_image_tensor'):
                adv_features = self.clip_model.encode_image_tensor(adversarial_image, requires_grad=False)
            else:
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
        真正的批量攻击实现
        
        Args:
            images: 图像列表
            texts: 文本列表
            target_texts: 目标文本列表
        
        Returns:
            攻击结果列表
        """
        logging.info(f"开始批量PGD攻击 {len(images)} 个样本")
        start_time = time.time()
        
        # 预处理所有图像
        image_tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                img_tensor = self.transform(img)
            else:
                img_tensor = img
            image_tensors.append(img_tensor)
        
        # 转换为批量张量
        batch_images = torch.stack(image_tensors)
        
        # 分批处理以适应GPU内存
        all_results = []
        total_batches = (len(images) + self.config.batch_size - 1) // self.config.batch_size
        
        with tqdm(total=total_batches, desc="PGD批量攻击进度") as pbar:
            for batch_idx in range(0, len(images), self.config.batch_size):
                batch_end = min(batch_idx + self.config.batch_size, len(images))
                
                batch_imgs = batch_images[batch_idx:batch_end].to(self.device)
                batch_texts = texts[batch_idx:batch_end]
                batch_targets = target_texts[batch_idx:batch_end] if target_texts else None
                
                # 执行批量攻击
                batch_results = self._batch_pgd_attack(batch_imgs, batch_texts, batch_targets)
                all_results.extend(batch_results)
                
                pbar.update(1)
                pbar.set_postfix({
                    'batch': f'{batch_idx//self.config.batch_size + 1}/{total_batches}',
                    'success_rate': f'{sum(r["success"] for r in batch_results)/len(batch_results):.3f}'
                })
        
        attack_time = time.time() - start_time
        success_count = sum(1 for r in all_results if r['success'])
        
        # 更新统计信息
        self.attack_stats['total_attacks'] += len(images)
        if success_count > 0:
            self.attack_stats['successful_attacks'] += success_count
        
        logging.info(f"批量PGD攻击完成，成功率: {success_count}/{len(images)} ({success_count/len(images):.3f})")
        logging.info(f"总耗时: {attack_time:.2f}秒，平均每样本: {attack_time/len(images):.3f}秒")
        
        return all_results
    
    def _batch_pgd_attack(self, batch_images: torch.Tensor, batch_texts: List[str], 
                         batch_targets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        批量PGD攻击的核心实现
        
        Args:
            batch_images: 批量图像张量 [batch_size, C, H, W]
            batch_texts: 批量文本列表
            batch_targets: 批量目标文本列表（可选）
            
        Returns:
            批量攻击结果
        """
        batch_size = batch_images.size(0)
        results = []
        
        # 编码文本特征
        with torch.no_grad():
            text_features = self.clip_model.encode_text(batch_texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            if batch_targets:
                target_features = self.clip_model.encode_text(batch_targets)
                target_features = target_features / target_features.norm(dim=-1, keepdim=True)
            else:
                target_features = None
        
        # 初始化对抗样本
        adv_images = batch_images.clone().detach()
        
        # 添加随机噪声
        if self.config.num_steps > 1:
            noise = torch.empty_like(adv_images).uniform_(
                -self.config.epsilon, self.config.epsilon
            )
            adv_images = torch.clamp(adv_images + noise, 
                                   self.config.clip_min, self.config.clip_max)
        
        # 初始化动量
        momentum = torch.zeros_like(adv_images) if self.config.use_momentum else None
        
        # 优化器设置
        if self.config.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        
        # 迭代攻击
        for step in range(self.config.num_steps):
            adv_images.requires_grad_(True)
            
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    # 编码图像特征
                    if hasattr(self.clip_model, 'encode_image_tensor'):
                        adv_features = self.clip_model.encode_image_tensor(adv_images, requires_grad=True)
                    else:
                        adv_features = self.clip_model.encode_image(adv_images)
                    
                    adv_features = adv_features / adv_features.norm(dim=-1, keepdim=True)
                    
                    # 计算损失
                    if self.config.targeted and target_features is not None:
                        # 目标攻击：最大化与目标文本的相似度
                        loss = -F.cosine_similarity(adv_features, target_features, dim=-1).mean()
                    else:
                        # 非目标攻击：最小化与原始文本的相似度
                        loss = F.cosine_similarity(adv_features, text_features, dim=-1).mean()
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(lambda: None)  # 手动梯度更新
                scaler.update()
            else:
                # 编码图像特征
                if hasattr(self.clip_model, 'encode_image_tensor'):
                    adv_features = self.clip_model.encode_image_tensor(adv_images, requires_grad=True)
                else:
                    adv_features = self.clip_model.encode_image(adv_images)
                
                adv_features = adv_features / adv_features.norm(dim=-1, keepdim=True)
                
                # 计算损失
                if self.config.targeted and target_features is not None:
                    # 目标攻击：最大化与目标文本的相似度
                    loss = -F.cosine_similarity(adv_features, target_features, dim=-1).mean()
                else:
                    # 非目标攻击：最小化与原始文本的相似度
                    loss = F.cosine_similarity(adv_features, text_features, dim=-1).mean()
                
                # 反向传播
                loss.backward()
            
            # 获取梯度
            grad = adv_images.grad.data
            
            # 梯度裁剪
            if self.config.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_([adv_images], self.config.gradient_clip_value)
            
            # 应用动量
            if self.config.use_momentum and momentum is not None:
                momentum = self.config.momentum * momentum + grad / torch.norm(grad, p=1, dim=(1,2,3), keepdim=True)
                grad = momentum
            
            # 更新对抗样本
            with torch.no_grad():
                if self.config.targeted:
                    adv_images.data = adv_images.data - self.config.alpha * grad.sign()
                else:
                    adv_images.data = adv_images.data + self.config.alpha * grad.sign()
                
                # 投影到约束集合
                delta = torch.clamp(adv_images.data - batch_images, 
                                  -self.config.epsilon, self.config.epsilon)
                adv_images.data = torch.clamp(batch_images + delta, 
                                            self.config.clip_min, self.config.clip_max)
                
                # 清零梯度
                adv_images.grad.zero_()
        
        # 评估攻击结果
        with torch.no_grad():
            final_features = self.clip_model.encode_image(adv_images)
            final_features = final_features / final_features.norm(dim=-1, keepdim=True)
            
            for i in range(batch_size):
                # 计算攻击成功率
                if self.config.targeted and target_features is not None:
                    similarity = F.cosine_similarity(
                        final_features[i:i+1], target_features[i:i+1], dim=-1
                    ).item()
                    success = similarity > 0.5  # 目标攻击成功阈值
                else:
                    similarity = F.cosine_similarity(
                        final_features[i:i+1], text_features[i:i+1], dim=-1
                    ).item()
                    success = similarity < 0.3  # 非目标攻击成功阈值
                
                # 计算扰动强度
                perturbation = (adv_images[i] - batch_images[i]).abs()
                perturbation_norm = perturbation.max().item()  # L∞范数
                
                attack_info = {
                    'iterations': self.config.num_steps,
                    'final_similarity': similarity,
                    'perturbation_norm': perturbation_norm,
                    'targeted': self.config.targeted
                }
                
                results.append({
                    'adversarial_image': adv_images[i].cpu(),
                    'original_image': batch_images[i].cpu(),
                    'perturbation': (adv_images[i] - batch_images[i]).cpu(),
                    'success': success,
                    'attack_info': attack_info,
                    'config': self.config
                })
        
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
        if isinstance(image, Image.Image):
            image_hash = hash(str(image))
        else:
            # 对于张量，使用其形状和部分值来生成哈希
            tensor_info = f"{image.shape}_{image.device}_{image.sum().item()}"
            image_hash = hash(tensor_info)
        
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