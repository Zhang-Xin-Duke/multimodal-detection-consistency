"""SMA (Semantic Misalignment Attack) 攻击实现

基于语义错位攻击的对抗样本生成方法。
参考论文：Adversarial Illusions in Multi-Modal Embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SMAAttackConfig:
    """SMA攻击配置
    
    基于参考实现的配置参数
    """
    # 基础设置
    random_seed: int = 42
    device: str = 'cuda'
    
    # SMA核心参数
    epsilon: float = 0.031  # 扰动幅度 (8/255)
    alpha: float = 0.008    # 步长 (2/255)
    num_iter: int = 15      # 迭代次数
    
    # 语义错位参数
    semantic_weight: float = 2.0      # 语义损失权重
    perceptual_weight: float = 0.5    # 感知损失权重
    diversity_weight: float = 0.1     # 多样性损失权重
    
    # 目标设置
    targeted: bool = True
    target_selection: str = "semantic"  # 目标选择策略
    semantic_shift_strength: float = 1.5  # 语义偏移强度
    
    # 优化参数
    momentum: float = 0.9
    decay_factor: float = 0.95
    learning_rate: float = 0.01
    gamma_epochs: int = 5  # 学习率衰减周期
    
    # 约束参数
    norm_type: str = "inf"  # 范数类型
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    
    # JPEG压缩模拟
    jpeg_compression: bool = False
    jpeg_quality: int = 95
    
    # 高级参数
    adaptive_step_size: bool = True
    gradient_clipping: bool = True
    clip_value: float = 1.0
    
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


class SMAAttacker:
    """SMA攻击器
    
    实现基于语义错位的对抗攻击方法
    """
    
    def __init__(self, clip_model, config: SMAAttackConfig):
        """
        初始化SMA攻击器
        
        Args:
            clip_model: CLIP模型
            config: 攻击配置
        """
        self.config = config
        
        # 设置设备和多GPU配置
        self._setup_devices()
        
        # 初始化CLIP模型
        self._initialize_clip_model(clip_model)
        
        # 攻击统计
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'semantic_misalignment_score': 0.0,
            'avg_perturbation_norm': 0.0
        }
        
        # 缓存
        if config.enable_cache:
            self.cache = {}
            self.cache_keys = []
        
        logger.info(f"初始化SMA攻击器，设备: {self.device}")
        if self.config.enable_multi_gpu and len(self.device_ids) > 1:
            logger.info(f"启用多GPU并行，设备ID: {self.device_ids}")
        logger.info(f"批处理大小: {self.config.batch_size}")
    
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
            logger.info(f"CLIP模型已设置为多GPU并行，设备: {self.device_ids}")
        
        # 确保模型在正确的设备上
        self.clip_model = self.clip_model.to(self.device)
    
    def attack(self, images: torch.Tensor, texts: List[str], 
               target_semantics: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        执行SMA攻击
        
        Args:
            images: 输入图像 [batch_size, 3, H, W]
            texts: 文本查询列表
            target_semantics: 目标语义列表（用于语义错位）
            
        Returns:
            攻击结果字典
        """
        batch_size = images.size(0)
        
        # 编码文本特征
        with torch.no_grad():
            text_features = self.clip_model.encode_text(texts)
            if target_semantics:
                target_features = self.clip_model.encode_text(target_semantics)
            else:
                # 生成语义错位目标
                target_features = self._generate_semantic_targets(text_features, texts)
        
        # 执行攻击
        adv_images = self._sma_attack(
            images, text_features, target_features, texts
        )
        
        # 计算攻击成功率和语义错位分数
        success_rate, semantic_score = self._evaluate_attack_success(
            adv_images, texts, target_semantics
        )
        
        # 更新统计信息
        self.attack_stats['total_attacks'] += batch_size
        self.attack_stats['successful_attacks'] += int(success_rate * batch_size)
        self.attack_stats['semantic_misalignment_score'] += semantic_score
        
        # 缓存结果
        if self.config.enable_cache:
            self._cache_results(images, adv_images, texts)
        
        return {
            'adversarial_images': adv_images,
            'original_images': images,
            'success_rate': success_rate,
            'semantic_misalignment_score': semantic_score,
            'perturbation_norm': torch.norm(
                adv_images - images, p=float('inf'), dim=(1,2,3)
            ).mean().item()
        }
    
    def _sma_attack(self, images: torch.Tensor, text_features: torch.Tensor,
                    target_features: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        执行SMA攻击的核心逻辑
        
        Args:
            images: 原始图像
            text_features: 文本特征
            target_features: 目标特征
            texts: 文本列表
            
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
        
        pbar = tqdm(range(self.config.num_iter), desc="SMA Attack")
        for iteration in pbar:
            # JPEG压缩模拟（如果启用）
            if self.config.jpeg_compression:
                adv_images_processed = self._apply_jpeg_compression(adv_images)
            else:
                adv_images_processed = adv_images
            
            # 前向传播
            image_features = self.clip_model.encode_image(adv_images_processed)
            
            # 计算语义错位损失
            semantic_loss = self._compute_semantic_misalignment_loss(
                image_features, text_features, target_features
            )
            
            # 计算感知损失
            perceptual_loss = self._compute_perceptual_loss(
                adv_images, images
            )
            
            # 计算多样性损失
            diversity_loss = self._compute_diversity_loss(image_features)
            
            # 总损失
            total_loss = (
                self.config.semantic_weight * semantic_loss +
                self.config.perceptual_weight * perceptual_loss +
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
            
            # 学习率衰减
            if (iteration + 1) % self.config.gamma_epochs == 0:
                current_lr *= self.config.decay_factor
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss.item(),
                'lr': current_lr,
                'semantic': semantic_loss.item()
            })
        
        return adv_images.detach()
    
    def _compute_semantic_misalignment_loss(self, image_features: torch.Tensor,
                                          text_features: torch.Tensor,
                                          target_features: torch.Tensor) -> torch.Tensor:
        """
        计算语义错位损失
        
        目标：使图像特征与目标语义高度相似，但与原文本语义不匹配
        """
        # 最大化与目标语义的相似度
        target_similarity = F.cosine_similarity(image_features, target_features, dim=1)
        
        # 最小化与原文本的相似度
        text_similarity = F.cosine_similarity(image_features, text_features, dim=1)
        
        # 语义错位损失：鼓励高目标相似度，低原文本相似度
        semantic_loss = -target_similarity.mean() + text_similarity.mean()
        
        # 添加语义偏移强度
        semantic_shift = self.config.semantic_shift_strength * (
            target_similarity - text_similarity
        ).mean()
        
        return semantic_loss - semantic_shift
    
    def _compute_perceptual_loss(self, adv_images: torch.Tensor,
                               orig_images: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失，保持视觉质量
        """
        # 简单的L2感知损失
        perceptual_loss = F.mse_loss(adv_images, orig_images)
        
        # 可以扩展为更复杂的感知损失（如VGG特征损失）
        return perceptual_loss
    
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
    
    def _generate_semantic_targets(self, text_features: torch.Tensor, 
                                 texts: List[str]) -> torch.Tensor:
        """
        生成语义错位目标
        
        基于原文本生成语义上不匹配但特征空间相近的目标
        """
        batch_size, feature_dim = text_features.shape
        
        if self.config.target_selection == "semantic":
            # 生成语义相反的目标
            # 简单策略：在特征空间中生成正交方向
            orthogonal_features = torch.randn_like(text_features)
            
            # 使用Gram-Schmidt正交化
            for i in range(batch_size):
                # 计算与原特征的投影
                projection = torch.dot(orthogonal_features[i], text_features[i]) / \
                           torch.norm(text_features[i])**2 * text_features[i]
                # 减去投影得到正交分量
                orthogonal_features[i] = orthogonal_features[i] - projection
                # 归一化
                orthogonal_features[i] = F.normalize(orthogonal_features[i], p=2, dim=0)
            
            return orthogonal_features
        
        elif self.config.target_selection == "random":
            # 生成随机目标
            random_features = torch.randn(batch_size, feature_dim, device=self.device)
            return F.normalize(random_features, p=2, dim=1)
        
        elif self.config.target_selection == "adversarial":
            # 生成对抗性目标（与原特征相反）
            return -F.normalize(text_features, p=2, dim=1)
        
        else:
            raise ValueError(f"不支持的目标选择策略: {self.config.target_selection}")
    
    def _apply_jpeg_compression(self, images: torch.Tensor) -> torch.Tensor:
        """
        应用JPEG压缩模拟
        
        这是一个简化的JPEG压缩模拟，实际应用中可能需要更复杂的实现
        """
        # 简单的量化模拟
        quantization_factor = (100 - self.config.jpeg_quality) / 100.0
        noise = torch.randn_like(images) * quantization_factor * 0.01
        return torch.clamp(images + noise, 0, 1)
    
    def _evaluate_attack_success(self, adv_images: torch.Tensor, texts: List[str],
                               target_semantics: Optional[List[str]] = None) -> Tuple[float, float]:
        """
        评估攻击成功率和语义错位分数
        
        Returns:
            (success_rate, semantic_misalignment_score)
        """
        with torch.no_grad():
            # 编码对抗图像特征
            adv_image_features = self.clip_model.encode_image(adv_images)
            
            # 编码文本特征
            text_features = self.clip_model.encode_text(texts)
            
            # 计算与原文本的相似度
            text_similarities = F.cosine_similarity(
                adv_image_features, text_features, dim=1
            )
            
            # 计算语义错位分数
            if target_semantics:
                target_features = self.clip_model.encode_text(target_semantics)
                target_similarities = F.cosine_similarity(
                    adv_image_features, target_features, dim=1
                )
                
                # 语义错位分数：目标相似度高，原文本相似度低
                semantic_score = (target_similarities - text_similarities).mean().item()
                
                # 攻击成功：目标相似度 > 原文本相似度
                success = (target_similarities > text_similarities).float()
            else:
                # 无目标攻击：原文本相似度显著降低
                semantic_score = -text_similarities.mean().item()
                success = (text_similarities < 0.3).float()  # 阈值可调
            
            return success.mean().item(), semantic_score
    
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
            stats['avg_semantic_score'] = stats['semantic_misalignment_score'] / stats['total_attacks']
        else:
            stats['success_rate'] = 0.0
            stats['avg_semantic_score'] = 0.0
        return stats
    
    def reset_stats(self):
        """
        重置攻击统计信息
        """
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'semantic_misalignment_score': 0.0,
            'avg_perturbation_norm': 0.0
        }
    
    def batch_attack(self, images: List[Union[torch.Tensor, 'PIL.Image.Image']], 
                    texts: List[str],
                    target_semantics: Optional[List[str]] = None) -> List[Dict[str, torch.Tensor]]:
        """真正的批量攻击实现
        
        Args:
            images: 图像列表
            texts: 文本列表
            target_semantics: 目标语义列表
            
        Returns:
            攻击结果列表
        """
        logger.info(f"开始批量SMA攻击 {len(images)} 个样本")
        start_time = time.time()
        
        # 预处理所有图像
        image_tensors = []
        for img in images:
            if hasattr(img, 'convert'):  # PIL Image
                # 简单的预处理，实际应用中可能需要更复杂的变换
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img)
            else:
                img_tensor = img
            image_tensors.append(img_tensor)
        
        # 转换为批量张量
        batch_images = torch.stack(image_tensors)
        
        # 分批处理以适应GPU内存
        all_results = []
        total_batches = (len(images) + self.config.batch_size - 1) // self.config.batch_size
        
        with tqdm(total=total_batches, desc="SMA批量攻击进度") as pbar:
            for batch_idx in range(0, len(images), self.config.batch_size):
                batch_end = min(batch_idx + self.config.batch_size, len(images))
                
                batch_imgs = batch_images[batch_idx:batch_end].to(self.device)
                batch_texts = texts[batch_idx:batch_end]
                batch_targets = target_semantics[batch_idx:batch_end] if target_semantics else None
                
                # 执行批量攻击
                batch_results = self._batch_sma_attack(batch_imgs, batch_texts, batch_targets)
                all_results.extend(batch_results)
                
                pbar.update(1)
                pbar.set_postfix({
                    'batch': f'{batch_idx//self.config.batch_size + 1}/{total_batches}',
                    'success_rate': f'{sum(r["success_rate"] for r in batch_results)/len(batch_results):.3f}'
                })
        
        attack_time = time.time() - start_time
        avg_success_rate = sum(r['success_rate'] for r in all_results) / len(all_results)
        
        # 更新统计信息
        self.attack_stats['total_attacks'] += len(images)
        self.attack_stats['successful_attacks'] += sum(r['success_rate'] * len(images) for r in all_results)
        
        logger.info(f"批量SMA攻击完成，平均成功率: {avg_success_rate:.3f}")
        logger.info(f"总耗时: {attack_time:.2f}秒，平均每样本: {attack_time/len(images):.3f}秒")
        
        return all_results
    
    def _batch_sma_attack(self, batch_images: torch.Tensor, batch_texts: List[str], 
                         batch_targets: Optional[List[str]] = None) -> List[Dict[str, torch.Tensor]]:
        """批量SMA攻击的核心实现
        
        Args:
            batch_images: 批量图像张量 [batch_size, C, H, W]
            batch_texts: 批量文本列表
            batch_targets: 批量目标语义列表
            
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
                # 生成语义错位目标
                target_features = self._generate_semantic_targets(text_features, batch_texts)
        
        # 初始化对抗样本
        adv_images = batch_images.clone().detach().to(self.device)
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
        
        # 混合精度支持
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
        
        # 迭代优化
        for iteration in range(self.config.num_iter):
            # 梯度清零
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            
            # JPEG压缩模拟（如果启用）
            if self.config.jpeg_compression:
                adv_images_processed = self._apply_jpeg_compression(adv_images)
            else:
                adv_images_processed = adv_images
            
            # 前向传播
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    image_features = self.clip_model.encode_image(adv_images_processed)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # 计算语义错位损失
                    semantic_loss = self._compute_semantic_misalignment_loss(
                        image_features, text_features, target_features
                    )
                    
                    # 计算感知损失
                    perceptual_loss = self._compute_perceptual_loss(
                        adv_images, batch_images
                    )
                    
                    # 计算多样性损失
                    diversity_loss = self._compute_diversity_loss(image_features)
                    
                    # 总损失
                    total_loss = (
                        self.config.semantic_weight * semantic_loss +
                        self.config.perceptual_weight * perceptual_loss +
                        self.config.diversity_weight * diversity_loss
                    )
                
                # 反向传播
                scaler.scale(total_loss).backward()
                
                # 梯度裁剪
                if self.config.gradient_clipping:
                    scaler.unscale_(torch.optim.SGD([adv_images], lr=1.0))
                    torch.nn.utils.clip_grad_norm_([adv_images], self.config.gradient_clip_value)
                
                # 获取梯度
                grad = adv_images.grad.data
                scaler.update()
            else:
                image_features = self.clip_model.encode_image(adv_images_processed)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 计算语义错位损失
                semantic_loss = self._compute_semantic_misalignment_loss(
                    image_features, text_features, target_features
                )
                
                # 计算感知损失
                perceptual_loss = self._compute_perceptual_loss(
                    adv_images, batch_images
                )
                
                # 计算多样性损失
                diversity_loss = self._compute_diversity_loss(image_features)
                
                # 总损失
                total_loss = (
                    self.config.semantic_weight * semantic_loss +
                    self.config.perceptual_weight * perceptual_loss +
                    self.config.diversity_weight * diversity_loss
                )
                
                # 反向传播
                total_loss.backward()
                
                # 获取梯度
                grad = adv_images.grad.data
                
                # 梯度裁剪
                if self.config.gradient_clipping:
                    grad = torch.clamp(grad, -self.config.gradient_clip_value, self.config.gradient_clip_value)
            
            # 动量更新
            momentum = self.config.momentum * momentum + grad
            
            # 更新对抗样本
            if self.config.norm_type == "inf":
                adv_images.data = adv_images.data - current_lr * momentum.sign()
                # L∞约束
                delta = torch.clamp(
                    adv_images.data - batch_images, delta_min, delta_max
                )
                adv_images.data = batch_images + delta
            else:
                # L2约束
                adv_images.data = adv_images.data - current_lr * momentum
                delta = adv_images.data - batch_images
                delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
                factor = torch.min(torch.ones_like(delta_norm), 
                                 self.config.epsilon / (delta_norm + 1e-8))
                delta = delta * factor.view(-1, 1, 1, 1)
                adv_images.data = batch_images + delta
            
            # 像素值约束
            adv_images.data = torch.clamp(
                adv_images.data, self.config.clamp_min, self.config.clamp_max
            )
            
            # 学习率衰减
            if self.config.adaptive_step_size and (iteration + 1) % self.config.gamma_epochs == 0:
                current_lr *= self.config.decay_factor
        
        # 评估攻击结果
        with torch.no_grad():
            final_image_features = self.clip_model.encode_image(adv_images)
            final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)
            
            # 计算成功率和语义错位分数
            text_similarities = F.cosine_similarity(
                final_image_features, text_features, dim=1
            )
            
            if batch_targets:
                target_similarities = F.cosine_similarity(
                    final_image_features, target_features, dim=1
                )
                semantic_scores = (target_similarities - text_similarities).cpu().numpy()
                success_flags = (target_similarities > text_similarities).cpu().numpy()
            else:
                semantic_scores = -text_similarities.cpu().numpy()
                success_flags = (text_similarities < 0.3).cpu().numpy()
            
            # 计算扰动强度
            perturbation_norms = torch.norm(
                adv_images - batch_images, p=float('inf'), dim=(1,2,3)
            ).cpu().numpy()
        
        # 构建结果
        for i in range(batch_size):
            results.append({
                'adversarial_images': adv_images[i].cpu(),
                'original_images': batch_images[i].cpu(),
                'success_rate': float(success_flags[i]),
                'semantic_misalignment_score': float(semantic_scores[i]),
                'perturbation_norm': float(perturbation_norms[i]),
                'text': batch_texts[i],
                'target_semantic': batch_targets[i] if batch_targets else None
            })
        
        return results


def create_sma_attacker(clip_model, config: Optional[SMAAttackConfig] = None) -> SMAAttacker:
    """
    创建SMA攻击器的工厂函数
    
    Args:
        clip_model: CLIP模型
        config: 攻击配置，如果为None则使用默认配置
        
    Returns:
        SMA攻击器实例
    """
    if config is None:
        config = SMAAttackConfig()
    
    return SMAAttacker(clip_model, config)


class SMAAttackPresets:
    """
    SMA攻击预设配置
    """
    
    @staticmethod
    def weak_attack() -> SMAAttackConfig:
        """弱攻击配置"""
        return SMAAttackConfig(
            epsilon=0.016,  # 4/255
            num_iter=10,
            semantic_weight=1.0,
            learning_rate=0.005
        )
    
    @staticmethod
    def strong_attack() -> SMAAttackConfig:
        """强攻击配置"""
        return SMAAttackConfig(
            epsilon=0.047,  # 12/255
            num_iter=25,
            semantic_weight=3.0,
            learning_rate=0.02
        )
    
    @staticmethod
    def semantic_targeted_attack() -> SMAAttackConfig:
        """语义目标攻击配置"""
        return SMAAttackConfig(
            targeted=True,
            target_selection="semantic",
            semantic_weight=2.5,
            semantic_shift_strength=2.0,
            num_iter=20
        )
    
    @staticmethod
    def paper_config() -> SMAAttackConfig:
        """论文中的配置"""
        return SMAAttackConfig(
            epsilon=0.031,  # 8/255
            alpha=0.008,    # 2/255
            num_iter=15,
            semantic_weight=2.0,
            target_selection="semantic",
            jpeg_compression=False
        )
    
    @staticmethod
    def jpeg_robust_attack() -> SMAAttackConfig:
        """JPEG鲁棒攻击配置"""
        return SMAAttackConfig(
            epsilon=0.039,  # 10/255
            num_iter=20,
            semantic_weight=2.0,
            jpeg_compression=True,
            jpeg_quality=85
        )