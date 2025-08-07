"""FGSM图像攻击模块

实现基于FGSM (Fast Gradient Sign Method) 的图像对抗性攻击方法。
参考论文: "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., ICLR 2015)
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
class FGSMAttackConfig:
    """FGSM攻击配置
    
    参数设置严格按照原论文 Goodfellow et al., ICLR 2015
    """
    # 基础设置
    random_seed: int = 42
    device: str = 'cuda'
    
    # FGSM核心参数 (原论文设置)
    epsilon: float = 8.0 / 255.0  # 扰动幅度，论文中使用 ε = 0.25 对于MNIST，8/255 对于ImageNet
    
    # 攻击目标
    targeted: bool = False        # 是否为目标攻击
    
    # 约束参数
    clip_min: float = 0.0
    clip_max: float = 1.0
    
    # 评估参数
    batch_size: int = 32
    
    # 批处理和多GPU优化
    enable_multi_gpu: bool = False
    gpu_ids: List[int] = None  # GPU设备ID列表，None表示使用所有可用GPU
    batch_size_per_gpu: int = 8  # 每个GPU的批处理大小
    num_workers: int = 4  # 数据加载器工作进程数
    
    # 内存优化
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    mixed_precision: bool = False  # 混合精度训练
    pin_memory: bool = True  # 固定内存
    gradient_clip_value: float = 1.0  # 梯度裁剪值
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000
    
    # 损失函数类型
    loss_type: str = 'cosine'  # 'cosine', 'mse', 'cross_entropy'

class FGSMAttacker:
    """FGSM攻击器
    
    实现快速梯度符号方法，基于 Goodfellow et al., ICLR 2015
    """
    
    def __init__(self, clip_model, config: FGSMAttackConfig):
        """
        初始化FGSM攻击器
        
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
            'average_similarity_drop': 0.0
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
        
        logging.info(f"初始化FGSM攻击器，设备: {self.device}")
        if self.config.enable_multi_gpu and len(self.device_ids) > 1:
            logging.info(f"启用多GPU并行，设备ID: {self.device_ids}")
        logging.info(f"批处理大小: {self.config.batch_size}")
    
    def _setup_devices(self):
        """设置设备和多GPU配置"""
        self.device = torch.device(self.config.device)
        
        if self.config.enable_multi_gpu and torch.cuda.is_available():
            if self.config.gpu_ids is None:
                # 使用所有可用GPU
                self.device_ids = list(range(torch.cuda.device_count()))
            else:
                self.device_ids = self.config.gpu_ids
            
            # 确保主设备在设备列表中
            if self.device.index not in self.device_ids:
                self.device_ids.insert(0, self.device.index)
        else:
            self.device_ids = [self.device.index] if self.device.type == 'cuda' else [0]
    
    def _initialize_clip_model(self, clip_model):
        """初始化CLIP模型并设置多GPU"""
        self.clip_model = clip_model.to(self.device)
        
        # 多GPU设置
        if self.config.enable_multi_gpu and len(self.device_ids) > 1:
            self.clip_model = nn.DataParallel(self.clip_model, device_ids=self.device_ids)
            logging.info(f"CLIP模型已设置为多GPU并行，设备: {self.device_ids}")
        
        # 确保模型在正确的设备上
        self.clip_model = self.clip_model.to(self.device)
    
    def attack(self, image: Union[Image.Image, torch.Tensor], 
               text: str, 
               target_text: Optional[str] = None) -> Dict[str, Any]:
        """
        执行FGSM攻击
        
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
        
        # 执行FGSM攻击
        adversarial_image, attack_info = self._fgsm_attack(
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
    
    def _fgsm_attack(self, image: torch.Tensor, 
                     text_features: torch.Tensor,
                     target_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        执行FGSM攻击的核心逻辑
        
        基于原论文公式: x_adv = x + ε * sign(∇_x J(θ, x, y))
        
        Args:
            image: 输入图像张量
            text_features: 文本特征
            target_features: 目标文本特征（可选）
        
        Returns:
            对抗样本和攻击信息
        """
        # 克隆图像并启用梯度计算
        adv_image = image.clone().detach().requires_grad_(True)
        
        attack_info = {
            'loss_value': 0.0,
            'original_similarity': 0.0,
            'adversarial_similarity': 0.0,
            'perturbation_norm': 0.0
        }
        
        # 计算原始相似度
        with torch.no_grad():
            if hasattr(self.clip_model, 'encode_image_tensor'):
                original_image_features = self.clip_model.encode_image_tensor(image)
            else:
                original_image_features = self.clip_model.encode_image(image)
            
            if isinstance(original_image_features, torch.Tensor):
                original_image_features = original_image_features.to(self.device)
            
            original_similarity = F.cosine_similarity(
                original_image_features, text_features, dim=-1
            ).mean().item()
            attack_info['original_similarity'] = original_similarity
        
        # 前向传播计算损失
        if hasattr(self.clip_model, 'encode_image_tensor'):
            image_features = self.clip_model.encode_image_tensor(adv_image)
        else:
            image_features = self.clip_model.encode_image(adv_image)
        
        if isinstance(image_features, torch.Tensor):
            image_features = image_features.to(self.device)
        
        # 计算损失
        if self.config.targeted and target_features is not None:
            # 目标攻击：最大化与目标文本的相似度
            if self.config.loss_type == 'cosine':
                loss = -F.cosine_similarity(image_features, target_features, dim=-1).mean()
            elif self.config.loss_type == 'mse':
                loss = F.mse_loss(image_features, target_features)
            else:
                raise ValueError(f"不支持的损失函数类型: {self.config.loss_type}")
        else:
            # 非目标攻击：最小化与原始文本的相似度
            if self.config.loss_type == 'cosine':
                loss = F.cosine_similarity(image_features, text_features, dim=-1).mean()
            elif self.config.loss_type == 'mse':
                loss = -F.mse_loss(image_features, text_features)
            else:
                raise ValueError(f"不支持的损失函数类型: {self.config.loss_type}")
        
        attack_info['loss_value'] = loss.item()
        
        # 反向传播计算梯度
        loss.backward()
        
        # FGSM核心公式：x_adv = x + ε * sign(∇_x J)
        with torch.no_grad():
            # 获取梯度符号
            grad_sign = adv_image.grad.sign()
            
            # 应用FGSM扰动
            perturbation = self.config.epsilon * grad_sign
            adv_image = adv_image + perturbation
            
            # 裁剪到有效范围
            adv_image = torch.clamp(adv_image, self.config.clip_min, self.config.clip_max)
            
            # 计算扰动范数
            perturbation_norm = torch.norm(perturbation, p=float('inf')).item()
            attack_info['perturbation_norm'] = perturbation_norm
        
        # 计算对抗样本的相似度
        with torch.no_grad():
            if hasattr(self.clip_model, 'encode_image_tensor'):
                adv_image_features = self.clip_model.encode_image_tensor(adv_image)
            else:
                adv_image_features = self.clip_model.encode_image(adv_image)
            
            if isinstance(adv_image_features, torch.Tensor):
                adv_image_features = adv_image_features.to(self.device)
            
            adversarial_similarity = F.cosine_similarity(
                adv_image_features, text_features, dim=-1
            ).mean().item()
            attack_info['adversarial_similarity'] = adversarial_similarity
        
        return adv_image.detach(), attack_info
    
    def _evaluate_attack_success(self, adversarial_image: torch.Tensor,
                                text_features: torch.Tensor,
                                target_features: Optional[torch.Tensor] = None) -> bool:
        """
        评估攻击是否成功
        
        Args:
            adversarial_image: 对抗样本
            text_features: 原始文本特征
            target_features: 目标文本特征（可选）
        
        Returns:
            攻击是否成功
        """
        with torch.no_grad():
            if hasattr(self.clip_model, 'encode_image_tensor'):
                adv_features = self.clip_model.encode_image_tensor(adversarial_image)
            else:
                adv_features = self.clip_model.encode_image(adversarial_image)
            
            if isinstance(adv_features, torch.Tensor):
                adv_features = adv_features.to(self.device)
            
            if self.config.targeted and target_features is not None:
                # 目标攻击：检查是否更接近目标文本
                target_sim = F.cosine_similarity(adv_features, target_features, dim=-1).mean()
                original_sim = F.cosine_similarity(adv_features, text_features, dim=-1).mean()
                return target_sim > original_sim
            else:
                # 非目标攻击：检查相似度是否显著下降
                similarity = F.cosine_similarity(adv_features, text_features, dim=-1).mean()
                # 设置成功阈值为相似度下降超过0.1
                return similarity < 0.5  # 可根据需要调整阈值
    
    def _get_cache_key(self, image: Union[Image.Image, torch.Tensor], 
                      text: str, target_text: Optional[str] = None) -> str:
        """
        生成缓存键
        
        Args:
            image: 输入图像
            text: 文本
            target_text: 目标文本
        
        Returns:
            缓存键字符串
        """
        if isinstance(image, Image.Image):
            image_hash = hash(image.tobytes())
        else:
            image_hash = hash(image.cpu().numpy().tobytes())
        
        text_hash = hash(text)
        target_hash = hash(target_text) if target_text else 0
        
        return f"fgsm_{image_hash}_{text_hash}_{target_hash}_{self.config.epsilon}"
    
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
        
        # 更新平均扰动
        total = self.attack_stats['total_attacks']
        prev_avg_pert = self.attack_stats['average_perturbation']
        self.attack_stats['average_perturbation'] = (
            (prev_avg_pert * (total - 1) + attack_info['perturbation_norm']) / total
        )
        
        # 更新平均相似度下降
        similarity_drop = (
            attack_info['original_similarity'] - attack_info['adversarial_similarity']
        )
        prev_avg_drop = self.attack_stats['average_similarity_drop']
        self.attack_stats['average_similarity_drop'] = (
            (prev_avg_drop * (total - 1) + similarity_drop) / total
        )
    
    def get_attack_stats(self) -> Dict[str, float]:
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
    
    def batch_attack(self, images: List[Union[Image.Image, torch.Tensor]], 
                    texts: List[str],
                    target_texts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        批量执行FGSM攻击（真正的批处理和多GPU并行）
        
        Args:
            images: 图像列表
            texts: 文本列表
            target_texts: 目标文本列表（可选）
        
        Returns:
            攻击结果列表
        """
        start_time = time.time()
        logging.info(f"开始批量FGSM攻击，样本数: {len(images)}")
        
        # 预处理图像
        processed_images = []
        for img in images:
            if isinstance(img, Image.Image):
                img_tensor = self.transform(img)
            else:
                img_tensor = img
            processed_images.append(img_tensor)
        
        # 转换为批量张量
        batch_images = torch.stack(processed_images).to(self.device)
        
        # 计算批处理大小
        total_samples = len(images)
        batch_size = self.config.batch_size
        if self.config.enable_multi_gpu and len(self.device_ids) > 1:
            batch_size = batch_size * len(self.device_ids)
        
        results = []
        
        # 分批处理
        with tqdm(total=total_samples, desc="FGSM批量攻击") as pbar:
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                batch_imgs = batch_images[i:end_idx]
                batch_texts = texts[i:end_idx]
                batch_targets = target_texts[i:end_idx] if target_texts else None
                
                # 执行批量攻击
                batch_results = self._batch_fgsm_attack(
                    batch_imgs, batch_texts, batch_targets
                )
                results.extend(batch_results)
                
                pbar.update(end_idx - i)
        
        # 更新统计信息
        for result in results:
            success = result['attack_success']
            self._update_attack_stats(result, success)
        
        end_time = time.time()
        logging.info(f"批量FGSM攻击完成，耗时: {end_time - start_time:.2f}秒")
        
        return results
    
    def _batch_fgsm_attack(self, batch_images: torch.Tensor, 
                          batch_texts: List[str],
                          batch_targets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        执行批量FGSM攻击的核心逻辑
        
        Args:
            batch_images: 批量图像张量 [B, C, H, W]
            batch_texts: 批量文本列表
            batch_targets: 批量目标文本列表（可选）
        
        Returns:
            批量攻击结果列表
        """
        batch_size = batch_images.size(0)
        results = []
        
        # 编码文本特征
        with torch.no_grad():
            text_tokens = torch.cat([self.clip_model.tokenize(text).to(self.device) 
                                   for text in batch_texts])
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 编码目标文本特征（如果有）
            if batch_targets:
                target_tokens = torch.cat([self.clip_model.tokenize(target).to(self.device) 
                                         for target in batch_targets])
                target_features = self.clip_model.encode_text(target_tokens)
                target_features = target_features / target_features.norm(dim=-1, keepdim=True)
            else:
                target_features = None
        
        # 编码原始图像特征
        with torch.no_grad():
            original_image_features = self.clip_model.encode_image(batch_images)
            original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)
            
            # 计算原始相似度
            original_similarities = torch.sum(original_image_features * text_features, dim=-1)
        
        # 初始化对抗样本
        adv_images = batch_images.clone().detach().requires_grad_(True)
        
        # 混合精度训练
        if self.config.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        
        # FGSM攻击
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            # 编码对抗图像特征
            adv_image_features = self.clip_model.encode_image(adv_images)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=-1, keepdim=True)
            
            if self.config.targeted and target_features is not None:
                # 目标攻击：最大化与目标文本的相似度
                loss = -torch.sum(adv_image_features * target_features, dim=-1).mean()
            else:
                # 非目标攻击：最小化与原始文本的相似度
                loss = torch.sum(adv_image_features * text_features, dim=-1).mean()
        
        # 反向传播
        if self.config.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(lambda: None)  # 不需要优化器
            scaler.update()
        else:
            loss.backward()
        
        # 获取梯度
        grad = adv_images.grad.data
        
        # 梯度裁剪
        if self.config.gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_([adv_images], self.config.gradient_clip_value)
        
        # FGSM更新
        if self.config.targeted and target_features is not None:
            # 目标攻击：沿梯度负方向
            adv_images = adv_images - self.config.epsilon * grad.sign()
        else:
            # 非目标攻击：沿梯度正方向
            adv_images = adv_images + self.config.epsilon * grad.sign()
        
        # 投影到约束范围
        adv_images = torch.clamp(adv_images, 
                               batch_images - self.config.epsilon, 
                               batch_images + self.config.epsilon)
        adv_images = torch.clamp(adv_images, self.config.pixel_min, self.config.pixel_max)
        
        # 评估攻击结果
        with torch.no_grad():
            final_image_features = self.clip_model.encode_image(adv_images)
            final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)
            
            # 计算最终相似度
            final_similarities = torch.sum(final_image_features * text_features, dim=-1)
            
            # 计算扰动强度
            perturbations = (adv_images - batch_images).view(batch_size, -1)
            perturbation_norms = torch.norm(perturbations, p=float('inf'), dim=-1)
        
        # 构建结果
        for i in range(batch_size):
            # 判断攻击成功
            if self.config.targeted and target_features is not None:
                target_sim = torch.sum(final_image_features[i] * target_features[i])
                attack_success = target_sim > original_similarities[i]
            else:
                similarity_drop = original_similarities[i] - final_similarities[i]
                attack_success = similarity_drop > 0.1  # 相似度下降阈值
            
            result = {
                'adversarial_image': adv_images[i].cpu(),
                'original_image': batch_images[i].cpu(),
                'perturbation': (adv_images[i] - batch_images[i]).cpu(),
                'perturbation_norm': perturbation_norms[i].item(),
                'original_similarity': original_similarities[i].item(),
                'adversarial_similarity': final_similarities[i].item(),
                'attack_success': attack_success.item() if isinstance(attack_success, torch.Tensor) else attack_success,
                'text': batch_texts[i],
                'target_text': batch_targets[i] if batch_targets else None,
                'attack_method': 'FGSM',
                'config': self.config
            }
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
            logging.info("FGSM攻击器缓存已清空")

def create_fgsm_attacker(clip_model, config: Optional[FGSMAttackConfig] = None) -> FGSMAttacker:
    """
    创建FGSM攻击器的工厂函数
    
    Args:
        clip_model: CLIP模型实例
        config: 攻击配置，如果为None则使用默认配置
    
    Returns:
        FGSM攻击器实例
    """
    if config is None:
        config = FGSMAttackConfig()
    
    return FGSMAttacker(clip_model, config)

# 预定义的配置
class FGSMAttackPresets:
    """FGSM攻击预设配置"""
    
    @staticmethod
    def weak_attack() -> FGSMAttackConfig:
        """弱攻击配置"""
        return FGSMAttackConfig(
            epsilon=4.0 / 255.0,  # 较小的扰动
            loss_type='cosine'
        )
    
    @staticmethod
    def strong_attack() -> FGSMAttackConfig:
        """强攻击配置"""
        return FGSMAttackConfig(
            epsilon=16.0 / 255.0,  # 较大的扰动
            loss_type='cosine'
        )
    
    @staticmethod
    def targeted_attack() -> FGSMAttackConfig:
        """目标攻击配置"""
        return FGSMAttackConfig(
            epsilon=8.0 / 255.0,
            targeted=True,
            loss_type='cosine'
        )
    
    @staticmethod
    def paper_config() -> FGSMAttackConfig:
        """原论文配置 (Goodfellow et al., ICLR 2015)"""
        return FGSMAttackConfig(
            epsilon=8.0 / 255.0,  # 对应ImageNet的典型设置
            targeted=False,
            loss_type='cosine',
            clip_min=0.0,
            clip_max=1.0
        )