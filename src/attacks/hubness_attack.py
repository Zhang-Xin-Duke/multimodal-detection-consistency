"""Hubness攻击模块

基于《Adversarial Hubness in Multi-Modal Retrieval》论文的复现实现。
论文作者: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov
论文链接: https://arxiv.org/pdf/2412.14113
GitHub: https://github.com/Tingwei-Zhang/adv_hub

核心思想:
1. 利用高维向量空间中的hubness现象
2. 将任意图像转化为对抗性hub
3. 使单个对抗样本能够被大量不同查询检索到
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
from pathlib import Path
from tqdm import tqdm
import json
import time
from PIL import Image
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import random
import hashlib
from ..models.clip_model import CLIPModel, CLIPConfig
from ..utils.metrics import SimilarityMetrics
import warnings
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class HubnessAttackConfig:
    """Hubness攻击配置 - 基于原论文《Adversarial Hubness in Multi-Modal Retrieval》"""
    # 模型配置
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"
    
    # 核心攻击参数 (基于原论文)
    epsilon: float = 16.0 / 255.0  # L∞扰动强度，论文标准设置
    num_iterations: int = 500  # 优化迭代次数
    step_size: float = 0.02  # 梯度步长
    
    # Hubness核心参数
    k_neighbors: int = 10  # k-近邻数量，用于hubness计算
    num_target_queries: int = 100  # 目标查询数量，论文中使用100个随机查询
    hubness_weight: float = 1.0  # hubness损失权重
    success_threshold: float = 0.84  # 攻击成功阈值，论文中21000/25000=0.84
    
    # 优化参数
    learning_rate: float = 0.02  # 学习率
    momentum: float = 0.9  # 动量
    weight_decay: float = 1e-4  # 权重衰减
    
    # 攻击模式
    attack_mode: str = "universal"  # universal: 通用攻击, targeted: 针对性攻击
    target_concepts: List[str] = None  # 针对性攻击的目标概念
    
    # 约束参数
    norm_constraint: str = "linf"  # L∞范数约束
    clamp_min: float = 0.0  # 图像像素最小值
    clamp_max: float = 1.0  # 图像像素最大值
    
    # 随机化参数
    random_start: bool = True  # 随机初始化扰动
    random_seed: int = 42  # 随机种子
    
    # 批处理和多GPU优化配置
    enable_multi_gpu: bool = True  # 启用多GPU并行
    gpu_ids: List[int] = None  # GPU ID列表，None表示使用所有可用GPU
    batch_size: int = 64  # 总批次大小
    batch_size_per_gpu: int = 16  # 每个GPU的批次大小
    num_workers: int = 4  # 数据加载器工作进程数
    
    # 内存优化
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    mixed_precision: bool = True  # 混合精度训练
    pin_memory: bool = True  # 固定内存
    
    # 缓存配置
    enable_cache: bool = True  # 启用结果缓存
    cache_size: int = 1000  # 缓存大小
    
    # 数据集配置
    dataset_size: int = 25000  # 测试数据集大小（论文中使用25K）
    query_pool_size: int = 1000  # 查询池大小
    
    def __post_init__(self):
        """初始化后处理"""
        if self.target_concepts is None:
            self.target_concepts = []
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HubnessAttackConfig':
        """从字典创建配置"""
        hubness_config = config_dict.get('attacks', {}).get('hubness', {})
        
        return cls(
            clip_model=hubness_config.get('clip_model', "openai/clip-vit-base-patch32"),
            epsilon=hubness_config.get('epsilon', 16.0 / 255.0),
            num_iterations=hubness_config.get('num_iterations', 500),
            step_size=hubness_config.get('step_size', 0.02),
            k_neighbors=hubness_config.get('k_neighbors', 10),
            num_target_queries=hubness_config.get('num_target_queries', 100),
            hubness_weight=hubness_config.get('hubness_weight', 1.0),
            success_threshold=hubness_config.get('success_threshold', 0.84),
            learning_rate=hubness_config.get('learning_rate', 0.02),
            momentum=hubness_config.get('momentum', 0.9),
            weight_decay=hubness_config.get('weight_decay', 1e-4),
            attack_mode=hubness_config.get('attack_mode', "universal"),
            target_concepts=hubness_config.get('target_concepts', []),
            norm_constraint=hubness_config.get('norm_constraint', "linf"),
            clamp_min=hubness_config.get('clamp_min', 0.0),
            clamp_max=hubness_config.get('clamp_max', 1.0),
            random_start=hubness_config.get('random_start', True),
            random_seed=hubness_config.get('random_seed', 42),
            enable_cache=hubness_config.get('enable_cache', True),
            cache_size=hubness_config.get('cache_size', 1000),
            dataset_size=hubness_config.get('dataset_size', 25000),
            query_pool_size=hubness_config.get('query_pool_size', 1000)
        )


class HubnessAttack:
    """Hubness攻击实现 - 基于原论文《Adversarial Hubness in Multi-Modal Retrieval》
    
    该攻击利用高维向量空间中的hubness现象，将任意图像转化为对抗性hub，
    使其能够被大量不同的文本查询检索到。
    """
    
    def __init__(self, config: HubnessAttackConfig):
        """初始化Hubness攻击器
        
        Args:
            config: 攻击配置
        """
        self.config = config
        
        # 设置设备和多GPU
        self._setup_devices()
        
        # 初始化CLIP模型
        self.clip_model = None
        self._initialize_clip_model()
        
        # 攻击统计
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'average_iterations': 0,
            'average_hubness_score': 0.0,
            'average_attack_time': 0.0
        }
        
        # 缓存
        self.cache = {} if config.enable_cache else None
        
        # 设置随机种子
        self._set_random_seed(config.random_seed)
        
        # 参考数据库
        self.text_features = None
        self.image_features = None
        
        logger.info(f"Hubness攻击器初始化完成，设备: {self.device}")
        logger.info(f"多GPU配置: 启用={config.enable_multi_gpu}, GPU数量={len(self.device_ids)}")
        logger.info(f"批处理配置: 总批次大小={config.batch_size}, 每GPU批次大小={config.batch_size_per_gpu}")
    
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
    
    def build_reference_database(self, reference_images: List[Image.Image], reference_texts: List[str]):
        """构建参考数据库
        
        Args:
            reference_images: 参考图像列表
            reference_texts: 参考文本列表
        """
        logger.info(f"开始构建参考数据库，包含 {len(reference_images)} 个图像和 {len(reference_texts)} 个文本")
        
        # 编码参考图像
        self.image_features = self.clip_model.encode_image(reference_images)
        
        # 编码参考文本
        self.text_features = self.clip_model.encode_text(reference_texts)
        
        logger.info(f"参考数据库构建完成，图像特征: {self.image_features.shape}, 文本特征: {self.text_features.shape}")
    
    def batch_attack(self, images: List[Union[torch.Tensor, Image.Image]], texts: List[str]) -> List[Dict[str, Any]]:
        """真正的批量攻击实现
        
        Args:
            images: 图像列表
            texts: 文本列表
            
        Returns:
            攻击结果列表
        """
        logger.info(f"开始批量攻击 {len(images)} 个样本")
        start_time = time.time()
        
        # 预处理所有图像
        image_tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                img_tensor = self.clip_model.preprocess(img)
            else:
                img_tensor = img
            image_tensors.append(img_tensor)
        
        # 转换为批量张量
        batch_images = torch.stack(image_tensors)
        
        # 分批处理以适应GPU内存
        all_results = []
        total_batches = (len(images) + self.config.batch_size - 1) // self.config.batch_size
        
        with tqdm(total=total_batches, desc="批量攻击进度") as pbar:
            for batch_idx in range(0, len(images), self.config.batch_size):
                batch_end = min(batch_idx + self.config.batch_size, len(images))
                
                batch_imgs = batch_images[batch_idx:batch_end].to(self.device)
                batch_texts = texts[batch_idx:batch_end]
                
                # 执行批量攻击
                batch_results = self._batch_attack_core(batch_imgs, batch_texts)
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
        self.attack_stats['average_attack_time'] = (
            (self.attack_stats['average_attack_time'] * (self.attack_stats['total_attacks'] - len(images)) + attack_time) / 
            self.attack_stats['total_attacks']
        )
        
        logger.info(f"批量攻击完成，成功率: {success_count}/{len(images)} ({success_count/len(images):.3f})")
        logger.info(f"总耗时: {attack_time:.2f}秒，平均每样本: {attack_time/len(images):.3f}秒")
        
        return all_results
    
    def _batch_attack_core(self, batch_images: torch.Tensor, batch_texts: List[str]) -> List[Dict[str, Any]]:
        """批量攻击的核心实现
        
        Args:
            batch_images: 批量图像张量 [batch_size, C, H, W]
            batch_texts: 批量文本列表
            
        Returns:
            批量攻击结果
        """
        batch_size = batch_images.size(0)
        results = []
        
        # 编码文本特征
        with torch.no_grad():
            text_features = self.clip_model.encode_text(batch_texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 为每个样本生成随机查询
        batch_queries = []
        for _ in range(batch_size):
            queries = self._generate_random_queries(self.config.num_target_queries)
            batch_queries.append(queries)
        
        # 编码所有查询
        all_queries = [q for queries in batch_queries for q in queries]
        with torch.no_grad():
            query_features = self.clip_model.encode_text(all_queries)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)
        
        # 重新组织查询特征
        query_features_list = []
        start_idx = 0
        for queries in batch_queries:
            end_idx = start_idx + len(queries)
            query_features_list.append(query_features[start_idx:end_idx])
            start_idx = end_idx
        
        # 初始化对抗样本
        adv_images = batch_images.clone().detach().requires_grad_(True)
        
        # 优化器设置
        if self.config.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        
        # 迭代攻击
        for iteration in range(self.config.num_iterations):
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    # 编码图像特征
                    image_features = self.clip_model.encode_image(adv_images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # 计算hubness损失
                    total_loss = 0
                    for i in range(batch_size):
                        img_feat = image_features[i:i+1]
                        query_feat = query_features_list[i]
                        
                        # 计算相似度
                        similarities = torch.mm(img_feat, query_feat.t())
                        
                        # Hubness损失：最大化平均相似度
                        hubness_loss = -similarities.mean()
                        total_loss += hubness_loss
                    
                    total_loss = total_loss / batch_size
                
                # 反向传播
                scaler.scale(total_loss).backward()
                scaler.step(lambda: None)  # 手动梯度更新
                scaler.update()
            else:
                # 编码图像特征
                image_features = self.clip_model.encode_image(adv_images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 计算hubness损失
                total_loss = 0
                for i in range(batch_size):
                    img_feat = image_features[i:i+1]
                    query_feat = query_features_list[i]
                    
                    # 计算相似度
                    similarities = torch.mm(img_feat, query_feat.t())
                    
                    # Hubness损失：最大化平均相似度
                    hubness_loss = -similarities.mean()
                    total_loss += hubness_loss
                
                total_loss = total_loss / batch_size
                
                # 反向传播
                total_loss.backward()
            
            # 梯度上升更新
            with torch.no_grad():
                grad = adv_images.grad.data
                
                # 梯度裁剪
                if self.config.gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_([adv_images], self.config.gradient_clip_value)
                
                # 更新对抗样本
                if self.config.norm_constraint == 'linf':
                    adv_images.data += self.config.step_size * grad.sign()
                    # L∞约束
                    delta = torch.clamp(adv_images.data - batch_images, -self.config.epsilon, self.config.epsilon)
                    adv_images.data = torch.clamp(batch_images + delta, 0, 1)
                elif self.config.norm_constraint == 'l2':
                    grad_norm = grad.view(batch_size, -1).norm(dim=1, keepdim=True)
                    grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
                    adv_images.data += self.config.step_size * grad_normalized
                    # L2约束
                    delta = adv_images.data - batch_images
                    delta_norm = delta.view(batch_size, -1).norm(dim=1, keepdim=True)
                    delta = delta / (delta_norm.view(-1, 1, 1, 1) + 1e-8) * torch.clamp(delta_norm, max=self.config.epsilon).view(-1, 1, 1, 1)
                    adv_images.data = torch.clamp(batch_images + delta, 0, 1)
                
                # 清零梯度
                adv_images.grad.zero_()
        
        # 评估攻击结果
        with torch.no_grad():
            final_image_features = self.clip_model.encode_image(adv_images)
            final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)
            
            for i in range(batch_size):
                img_feat = final_image_features[i:i+1]
                query_feat = query_features_list[i]
                
                # 计算最终hubness分数
                similarities = torch.mm(img_feat, query_feat.t())
                hubness_score = similarities.mean().item()
                
                # 计算扰动强度
                perturbation = (adv_images[i] - batch_images[i]).abs()
                if self.config.norm_constraint == 'linf':
                    perturbation_strength = perturbation.max().item()
                else:
                    perturbation_strength = perturbation.view(-1).norm().item()
                
                # 判断攻击成功
                success = hubness_score > self.config.success_threshold
                
                results.append({
                    'success': success,
                    'hubness_score': hubness_score,
                    'perturbation_strength': perturbation_strength,
                    'adversarial_image': adv_images[i].cpu(),
                    'original_image': batch_images[i].cpu(),
                    'target_queries': batch_queries[i],
                    'iterations': self.config.num_iterations
                })
        
        return results
    
    def _set_random_seed(self, seed: int):
        """设置随机种子以确保可复现性"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _initialize_clip_model(self):
        """初始化CLIP模型"""
        try:
            clip_config = CLIPConfig(
                model_name=self.config.clip_model,
                device=self.device
            )
            self.clip_model = CLIPModel(clip_config)
            
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
            
            logger.info(f"CLIP模型加载成功: {self.config.clip_model}")
        except Exception as e:
            logger.error(f"CLIP模型加载失败: {e}")
            raise
    
    def compute_hubness(self, image_features: torch.Tensor, 
                       text_features: torch.Tensor, 
                       k: int = 10) -> float:
        """计算hubness分数 - 基于原论文算法
        
        Args:
            image_features: 图像特征 [num_images, feature_dim]
            text_features: 文本特征 [num_texts, feature_dim] 
            k: k-近邻数量
            
        Returns:
            hubness分数 (表示图像被检索为top-1的查询比例)
        """
        # 确保两个特征张量在同一设备上
        if image_features.device != text_features.device:
            text_features = text_features.to(image_features.device)
        
        # 计算相似度矩阵
        similarities = F.cosine_similarity(
            text_features.unsqueeze(1),  # [num_texts, 1, feature_dim]
            image_features.unsqueeze(0),  # [1, num_images, feature_dim]
            dim=2
        )  # [num_texts, num_images]
        
        # 对每个查询找到最相似的图像（top-1）
        _, top_1_indices = torch.topk(similarities, 1, dim=1)  # [num_texts, 1]
        
        # 统计目标图像（第一个图像）被选为top-1的次数
        target_image_idx = 0  # 我们关心的是第一个图像（对抗样本）
        top_1_count = (top_1_indices.squeeze() == target_image_idx).sum().item()
        
        # 计算hubness分数：被检索为top-1的查询比例
        hubness_score = top_1_count / text_features.size(0)
        
        return hubness_score
    
    def create_adversarial_hub(self, image: Union[torch.Tensor, Image.Image], 
                              text_queries: List[str]) -> Dict[str, Any]:
        """创建对抗性hub - 基于原论文算法
        
        Args:
            image: 输入图像
            text_queries: 文本查询列表
            
        Returns:
            攻击结果字典
        """
        start_time = time.time()
        
        # 缓存检查
        if self.cache is not None:
            cache_key = self._generate_cache_key(image, text_queries)
            if cache_key in self.cache:
                logger.info("使用缓存结果")
                return self.cache[cache_key]
        
        # 输入转换
        if isinstance(image, Image.Image):
            # 使用CLIP模型的预处理器
            image_tensor = self.clip_model.preprocess(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
        
        # 编码文本查询
        text_features = self.clip_model.encode_text(text_queries)
        
        # 确保text_features在正确的设备上
        if text_features.device != self.device:
            text_features = text_features.to(self.device)
        
        # 执行攻击
        result = self._perform_attack(image_tensor, text_features, text_queries)
        
        # 更新统计
        attack_time = time.time() - start_time
        self._update_attack_stats(result, attack_time)
        
        # 缓存结果
        if self.cache is not None and len(self.cache) < self.config.cache_size:
            self.cache[cache_key] = result
        
        return result
    
    def _perform_attack(self, image_tensor: torch.Tensor, 
                       text_features: torch.Tensor,
                       text_queries: List[str]) -> Dict[str, Any]:
        """执行攻击的核心逻辑"""
        original_image = image_tensor.clone().detach()
        
        # 初始化对抗样本
        adv_image = original_image.clone().detach().requires_grad_(True)
        
        # 随机初始化扰动
        if self.config.random_start:
            noise = torch.randn_like(adv_image) * self.config.epsilon * 0.1
            adv_image = adv_image + noise
            adv_image = torch.clamp(adv_image, 
                                  original_image - self.config.epsilon,
                                  original_image + self.config.epsilon)
            adv_image = torch.clamp(adv_image, self.config.clamp_min, self.config.clamp_max)
            adv_image = adv_image.detach().requires_grad_(True)
        
        # 初始化扰动
        perturbation = torch.zeros_like(original_image, requires_grad=True)
        
        # 随机初始化扰动
        if self.config.random_start:
            with torch.no_grad():
                perturbation.data.uniform_(-self.config.epsilon, self.config.epsilon)
        
        best_loss = float('inf')
        best_image = original_image.clone()
        
        # 优化循环 - 基于论文的梯度上升算法
        for iteration in range(self.config.num_iterations):
            # 前向传播
            current_image = original_image + perturbation
            current_image = torch.clamp(current_image, self.config.clamp_min, self.config.clamp_max)
            
            # 编码图像（需要梯度计算）
            image_features = self.clip_model.encode_image_tensor(current_image, requires_grad=True)
            
            # 计算损失
            loss = self._compute_hubness_loss(image_features, text_features)
            
            # 反向传播
            loss.backward()
            
            # 获取梯度
            grad = perturbation.grad.data
            
            # 梯度上升（最小化负相似度，即最大化相似度）
            perturbation.data = perturbation.data - self.config.step_size * grad.sign()
            
            # 应用L∞范数约束
            perturbation.data = torch.clamp(
                perturbation.data, 
                -self.config.epsilon, 
                self.config.epsilon
            )
            
            # 确保图像在有效范围内
            perturbation.data = torch.clamp(
                original_image + perturbation.data,
                self.config.clamp_min,
                self.config.clamp_max
            ) - original_image
            
            # 清零梯度
            perturbation.grad.zero_()
            
            # 更新最佳结果（损失越小越好，因为是负相似度）
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_image = current_image.clone().detach()
            
            # 早停检查和日志
            if iteration % 50 == 0:
                current_hubness = self.compute_hubness(
                    image_features.unsqueeze(0), 
                    text_features, 
                    self.config.k_neighbors
                )
                logger.debug(f"迭代 {iteration}, 损失: {loss.item():.4f}, Hubness: {current_hubness:.4f}")
        
        # 计算最终hubness分数
        with torch.no_grad():
            final_features = self.clip_model.encode_image_tensor(best_image, requires_grad=False)
            hubness_score = self.compute_hubness(
                final_features.unsqueeze(0), 
                text_features, 
                self.config.k_neighbors
            )
        
        # 计算扰动强度
        perturbation = best_image - original_image
        perturbation_norm = torch.norm(perturbation, p=float('inf')).item()
        
        return {
            'adversarial_image': best_image.squeeze(0).cpu(),
            'original_image': original_image.squeeze(0).cpu(),
            'perturbation': perturbation.squeeze(0).cpu(),
            'hubness_score': hubness_score,
            'perturbation_norm': perturbation_norm,
            'final_loss': best_loss,
            'iterations': self.config.num_iterations,
            'success': hubness_score > self.config.success_threshold,
            'text_queries': text_queries
        }
    
    def _compute_hubness_loss(self, image_features: torch.Tensor, 
                             text_features: torch.Tensor) -> torch.Tensor:
        """计算hubness损失函数
        
        基于原论文的损失函数设计，目标是最大化图像与所有目标查询的相似度
        """
        # 确保两个特征张量在同一设备上
        if image_features.device != text_features.device:
            text_features = text_features.to(image_features.device)
        
        # 确保特征已归一化
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 计算图像特征与所有文本特征的相似度
        similarities = torch.mm(image_features, text_features.t())  # [1, num_texts]
        
        # 损失：负平均相似度（最大化相似度以增加hubness）
        loss = -similarities.mean()
        
        return loss
    
    def _generate_cache_key(self, image: Union[torch.Tensor, Image.Image], 
                           text_queries: List[str]) -> str:
        """生成缓存键"""
        # 简化的缓存键生成
        text_hash = hashlib.md5(''.join(text_queries).encode()).hexdigest()[:8]
        return f"hubness_{text_hash}"
    
    def _update_attack_stats(self, result: Dict[str, Any], attack_time: float):
        """更新攻击统计信息"""
        self.attack_stats['total_attacks'] += 1
        if result['success']:
            self.attack_stats['successful_attacks'] += 1
        
        # 更新平均值
        total = self.attack_stats['total_attacks']
        self.attack_stats['average_iterations'] = (
            (self.attack_stats['average_iterations'] * (total - 1) + result['iterations']) / total
        )
        self.attack_stats['average_hubness_score'] = (
            (self.attack_stats['average_hubness_score'] * (total - 1) + result['hubness_score']) / total
        )
        self.attack_stats['average_attack_time'] = (
            (self.attack_stats['average_attack_time'] * (total - 1) + attack_time) / total
        )
    
    def attack(self, image: Union[torch.Tensor, Image.Image], 
              text: str = None) -> Dict[str, Any]:
        """执行Hubness攻击
        
        Args:
            image: 输入图像
            text: 可选的文本查询（如果不提供，使用随机查询）
            
        Returns:
            攻击结果
        """
        # 图像预处理
        if isinstance(image, Image.Image):
            # 使用CLIP模型的预处理器
            image_tensor = self.clip_model.preprocess(image)
        else:
            image_tensor = image
        
        # 生成或使用文本查询
        if text is None:
            text_queries = self._generate_random_queries()
        else:
            text_queries = [text]
        
        # 执行攻击
        return self.create_adversarial_hub(image_tensor, text_queries)
    
    def attack_single(self, image: Union[torch.Tensor, Image.Image], 
                     text: str) -> Dict[str, Any]:
        """对单个图像-文本对执行攻击
        
        Args:
            image: 输入图像
            text: 文本查询
            
        Returns:
            攻击结果，包含success, hubness, similarity_change, iterations等字段
        """
        try:
            # 执行攻击
            result = self.attack(image, text)
            
            # 转换结果格式以匹配run_experiments.py的期望
            return {
                'success': result.get('success', False),
                'hubness': result.get('hubness_score', 0.0),
                'similarity_change': result.get('perturbation_norm', 0.0),
                'iterations': result.get('iterations', 0),
                'adversarial_image': result.get('adversarial_image'),
                'original_image': result.get('original_image'),
                'perturbation': result.get('perturbation'),
                'final_loss': result.get('final_loss', float('inf'))
            }
        except Exception as e:
            logger.error(f"单样本攻击失败: {e}")
            return {
                'success': False,
                'hubness': 0.0,
                'similarity_change': 0.0,
                'iterations': 0,
                'error': str(e)
            }
    
    def _generate_random_queries(self) -> List[str]:
        """生成随机文本查询"""
        # 简化的随机查询生成
        common_queries = [
            "a photo of a cat", "a dog playing", "a beautiful landscape",
            "a person walking", "a car on the road", "a bird flying",
            "a flower in the garden", "a building in the city", "food on a plate",
            "a sunset over the ocean"
        ]
        
        num_queries = min(self.config.num_target_queries, len(common_queries))
        return random.sample(common_queries, num_queries)
    
    def get_attack_stats(self) -> Dict[str, Any]:
        """获取攻击统计信息"""
        stats = self.attack_stats.copy()
        if stats['total_attacks'] > 0:
            stats['success_rate'] = stats['successful_attacks'] / stats['total_attacks']
        else:
            stats['success_rate'] = 0.0
        return stats


class HubnessAttackPresets:
    """Hubness攻击预设配置"""
    
    @staticmethod
    def weak_attack() -> HubnessAttackConfig:
        """弱攻击配置"""
        return HubnessAttackConfig(
            epsilon=8.0 / 255.0,
            num_iterations=100,
            learning_rate=0.01,
            k_neighbors=5,
            num_target_queries=50
        )
    
    @staticmethod
    def strong_attack() -> HubnessAttackConfig:
        """强攻击配置"""
        return HubnessAttackConfig(
            epsilon=32.0 / 255.0,
            num_iterations=1000,
            learning_rate=0.05,
            k_neighbors=20,
            num_target_queries=200
        )
    
    @staticmethod
    def targeted_attack(target_concepts: List[str]) -> HubnessAttackConfig:
        """针对性攻击配置"""
        return HubnessAttackConfig(
            attack_mode="targeted",
            target_concepts=target_concepts,
            epsilon=16.0 / 255.0,
            num_iterations=500,
            learning_rate=0.02,
            k_neighbors=10,
            num_target_queries=100
        )
    
    @staticmethod
    def paper_standard() -> HubnessAttackConfig:
        """论文标准配置"""
        return HubnessAttackConfig(
            epsilon=16.0 / 255.0,
            num_iterations=500,
            learning_rate=0.02,
            k_neighbors=10,
            num_target_queries=100,
            dataset_size=25000,
            query_pool_size=1000
        )


def create_hubness_attacker(config: Optional[HubnessAttackConfig] = None) -> HubnessAttack:
    """创建Hubness攻击器实例
    
    Args:
        config: 攻击配置，如果为None则使用默认配置
        
    Returns:
        HubnessAttack实例
    """
    if config is None:
        config = HubnessAttackConfig()
    
    return HubnessAttack(config)