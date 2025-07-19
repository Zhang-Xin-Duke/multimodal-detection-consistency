"""Hubness攻击模块

实现基于hubness的对抗性攻击方法。
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
from PIL import Image
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import random
from src.models import CLIPModel, CLIPConfig
from src.utils.metrics import SimilarityMetrics
import warnings

logger = logging.getLogger(__name__)


@dataclass
class HubnessAttackConfig:
    """Hubness攻击配置"""
    # 模型配置
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"
    
    # 攻击参数
    epsilon: float = 0.1  # 扰动强度
    num_iterations: int = 100  # 迭代次数
    step_size: float = 0.01  # 步长
    
    # Hubness参数
    k_neighbors: int = 10  # 近邻数量
    hubness_threshold: float = 0.8  # Hubness阈值
    target_hubness: float = 0.9  # 目标hubness值
    
    # 优化参数
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # 约束参数
    norm_constraint: str = "l2"  # l1, l2, linf
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    
    # 目标设置
    attack_mode: str = "targeted"  # targeted, untargeted
    target_similarity: float = 0.9  # 目标相似性
    
    # 搜索参数
    random_start: bool = True
    num_restarts: int = 1
    early_stopping: bool = True
    patience: int = 10
    
    # 评估参数
    success_threshold: float = 0.8
    consistency_weight: float = 0.5
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000


class HubnessAttacker:
    """Hubness攻击器"""
    
    def __init__(self, config: HubnessAttackConfig):
        """
        初始化Hubness攻击器
        
        Args:
            config: 攻击配置
        """
        self.config = config
        
        # 初始化CLIP模型
        self.clip_model = self._initialize_clip_model()
        
        # 相似性计算器
        self.similarity_metrics = SimilarityMetrics()
        
        # 近邻搜索器
        self.nn_searcher = None
        
        # 缓存
        self.attack_cache = {}
        self.hubness_cache = {}
        
        # 统计信息
        self.attack_stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'failed_attacks': 0,
            'total_time': 0.0,
            'cache_hits': 0
        }
        
        logger.info("Hubness攻击器初始化完成")
    
    def _initialize_clip_model(self) -> CLIPModel:
        """
        初始化CLIP模型
        
        Returns:
            CLIP模型实例
        """
        try:
            clip_config = CLIPConfig(
                model_name=self.config.clip_model,
                device=self.config.device
            )
            
            clip_model = CLIPModel(clip_config)
            logger.info(f"CLIP模型初始化完成: {self.config.clip_model}")
            
            return clip_model
            
        except Exception as e:
            logger.error(f"CLIP模型初始化失败: {e}")
            raise
    
    def build_reference_database(self, images: List[Image.Image], 
                                texts: List[str]):
        """
        构建参考数据库
        
        Args:
            images: 参考图像列表
            texts: 参考文本列表
        """
        try:
            logger.info(f"构建参考数据库: {len(images)} 张图像, {len(texts)} 个文本")
            
            # 编码图像和文本
            image_features = []
            text_features = []
            
            for image in images:
                features = self.clip_model.encode_image(image)
                image_features.append(features)
            
            for text in texts:
                features = self.clip_model.encode_text(text)
                text_features.append(features)
            
            self.image_features = np.array(image_features)
            self.text_features = np.array(text_features)
            
            # 构建近邻搜索器
            all_features = np.vstack([self.image_features, self.text_features])
            self.nn_searcher = NearestNeighbors(
                n_neighbors=self.config.k_neighbors + 1,
                metric='cosine'
            )
            self.nn_searcher.fit(all_features)
            
            logger.info("参考数据库构建完成")
            
        except Exception as e:
            logger.error(f"构建参考数据库失败: {e}")
            raise
    
    def compute_hubness(self, features: np.ndarray) -> float:
        """
        计算hubness值
        
        Args:
            features: 特征向量
            
        Returns:
            hubness值
        """
        try:
            if self.nn_searcher is None:
                raise ValueError("参考数据库未构建")
            
            # 查找最近邻
            distances, indices = self.nn_searcher.kneighbors(
                features.reshape(1, -1)
            )
            
            # 计算hubness（被作为近邻的频率）
            # 这里简化为与近邻的平均相似性
            similarities = 1 - distances[0][1:]  # 排除自身
            hubness = np.mean(similarities)
            
            return float(hubness)
            
        except Exception as e:
            logger.error(f"计算hubness失败: {e}")
            return 0.0
    
    def create_adversarial_hub(self, image: Union[Image.Image, torch.Tensor], 
                              text: str, 
                              target_text: Optional[str] = None) -> Dict[str, Any]:
        """
        创建对抗性hub
        
        Args:
            image: 输入图像
            text: 输入文本
            target_text: 目标文本（用于targeted攻击）
            
        Returns:
            攻击结果字典
        """
        try:
            start_time = time.time()
            
            # 检查缓存
            cache_key = self._get_cache_key(image, text, target_text)
            if self.config.enable_cache and cache_key in self.attack_cache:
                self.attack_stats['cache_hits'] += 1
                return self.attack_cache[cache_key]
            
            # 转换输入
            if isinstance(image, Image.Image):
                image_tensor = self.clip_model.preprocess_image(image)
            else:
                image_tensor = image
            
            image_tensor = image_tensor.to(self.config.device)
            image_tensor.requires_grad_(True)
            
            # 编码原始文本
            original_text_features = self.clip_model.encode_text(text)
            
            # 编码目标文本（如果有）
            target_text_features = None
            if target_text:
                target_text_features = self.clip_model.encode_text(target_text)
            
            # 执行攻击
            best_result = None
            best_success = False
            
            for restart in range(self.config.num_restarts):
                result = self._single_attack_iteration(
                    image_tensor.clone(),
                    original_text_features,
                    target_text_features,
                    text
                )
                
                if result['success'] and (not best_success or 
                    result['hubness'] > best_result['hubness']):
                    best_result = result
                    best_success = True
                elif not best_success:
                    best_result = result
            
            # 构建最终结果
            final_result = {
                'success': best_result['success'],
                'adversarial_image': best_result['adversarial_image'],
                'original_image': image,
                'perturbation': best_result['perturbation'],
                'hubness': best_result['hubness'],
                'similarity_change': best_result['similarity_change'],
                'iterations': best_result['iterations'],
                'attack_time': time.time() - start_time,
                'config': self.config.__dict__
            }
            
            # 缓存结果
            if self.config.enable_cache:
                if len(self.attack_cache) >= self.config.cache_size:
                    # 清理最旧的缓存项
                    oldest_key = next(iter(self.attack_cache))
                    del self.attack_cache[oldest_key]
                
                self.attack_cache[cache_key] = final_result
            
            # 更新统计信息
            self.attack_stats['total_attacks'] += 1
            if final_result['success']:
                self.attack_stats['successful_attacks'] += 1
            else:
                self.attack_stats['failed_attacks'] += 1
            self.attack_stats['total_time'] += final_result['attack_time']
            
            logger.debug(f"Hubness攻击完成: {'成功' if final_result['success'] else '失败'} "
                        f"(hubness: {final_result['hubness']:.3f})")
            
            return final_result
            
        except Exception as e:
            logger.error(f"创建对抗性hub失败: {e}")
            return {
                'success': False,
                'adversarial_image': image,
                'original_image': image,
                'perturbation': None,
                'hubness': 0.0,
                'similarity_change': 0.0,
                'iterations': 0,
                'attack_time': 0.0,
                'error': str(e)
            }
    
    def _single_attack_iteration(self, image_tensor: torch.Tensor, 
                                original_text_features: np.ndarray,
                                target_text_features: Optional[np.ndarray],
                                text: str) -> Dict[str, Any]:
        """
        单次攻击迭代
        
        Args:
            image_tensor: 图像张量
            original_text_features: 原始文本特征
            target_text_features: 目标文本特征
            text: 原始文本
            
        Returns:
            攻击结果
        """
        try:
            # 随机初始化扰动
            if self.config.random_start:
                noise = torch.randn_like(image_tensor) * self.config.epsilon * 0.1
                image_tensor = image_tensor + noise
                image_tensor = torch.clamp(image_tensor, self.config.clamp_min, self.config.clamp_max)
            
            original_image = image_tensor.clone()
            
            # 优化器
            optimizer = torch.optim.SGD(
                [image_tensor], 
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
            
            best_hubness = 0.0
            best_image = image_tensor.clone()
            patience_counter = 0
            
            for iteration in range(self.config.num_iterations):
                optimizer.zero_grad()
                
                # 编码当前图像
                current_image_features = self.clip_model.encode_image_tensor(image_tensor)
                
                # 计算hubness损失
                hubness_loss = self._compute_hubness_loss(current_image_features)
                
                # 计算相似性损失
                similarity_loss = self._compute_similarity_loss(
                    current_image_features,
                    original_text_features,
                    target_text_features
                )
                
                # 总损失
                total_loss = hubness_loss + self.config.consistency_weight * similarity_loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                # 应用约束
                with torch.no_grad():
                    # 扰动约束
                    perturbation = image_tensor - original_image
                    perturbation = self._apply_norm_constraint(perturbation)
                    image_tensor.data = original_image + perturbation
                    
                    # 像素值约束
                    image_tensor.data = torch.clamp(
                        image_tensor.data, 
                        self.config.clamp_min, 
                        self.config.clamp_max
                    )
                
                # 评估当前结果
                current_hubness = self.compute_hubness(
                    current_image_features.detach().cpu().numpy()
                )
                
                if current_hubness > best_hubness:
                    best_hubness = current_hubness
                    best_image = image_tensor.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 早停
                if (self.config.early_stopping and 
                    patience_counter >= self.config.patience):
                    break
                
                # 成功条件检查
                if current_hubness >= self.config.target_hubness:
                    break
            
            # 计算最终结果
            final_perturbation = best_image - original_image
            final_hubness = self.compute_hubness(
                self.clip_model.encode_image_tensor(best_image).detach().cpu().numpy()
            )
            
            # 计算相似性变化
            original_similarity = self.clip_model.compute_similarity(
                text, 
                self._tensor_to_image(original_image)
            )
            adversarial_similarity = self.clip_model.compute_similarity(
                text, 
                self._tensor_to_image(best_image)
            )
            similarity_change = adversarial_similarity - original_similarity
            
            success = final_hubness >= self.config.hubness_threshold
            
            return {
                'success': success,
                'adversarial_image': self._tensor_to_image(best_image),
                'perturbation': final_perturbation.detach().cpu().numpy(),
                'hubness': final_hubness,
                'similarity_change': similarity_change,
                'iterations': iteration + 1
            }
            
        except Exception as e:
            logger.error(f"攻击迭代失败: {e}")
            return {
                'success': False,
                'adversarial_image': self._tensor_to_image(image_tensor),
                'perturbation': None,
                'hubness': 0.0,
                'similarity_change': 0.0,
                'iterations': 0
            }
    
    def _compute_hubness_loss(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        计算hubness损失
        
        Args:
            image_features: 图像特征
            
        Returns:
            hubness损失
        """
        try:
            # 简化的hubness损失：最大化与参考特征的相似性
            if hasattr(self, 'image_features') and hasattr(self, 'text_features'):
                ref_features = torch.tensor(
                    np.vstack([self.image_features, self.text_features]),
                    device=image_features.device,
                    dtype=image_features.dtype
                )
                
                # 计算相似性
                similarities = F.cosine_similarity(
                    image_features.unsqueeze(0), 
                    ref_features, 
                    dim=1
                )
                
                # hubness损失：负的平均相似性（最大化相似性）
                hubness_loss = -similarities.mean()
            else:
                # 如果没有参考数据库，使用简单的正则化
                hubness_loss = -torch.norm(image_features)
            
            return hubness_loss
            
        except Exception as e:
            logger.error(f"计算hubness损失失败: {e}")
            return torch.tensor(0.0, device=image_features.device)
    
    def _compute_similarity_loss(self, image_features: torch.Tensor,
                                original_text_features: np.ndarray,
                                target_text_features: Optional[np.ndarray]) -> torch.Tensor:
        """
        计算相似性损失
        
        Args:
            image_features: 图像特征
            original_text_features: 原始文本特征
            target_text_features: 目标文本特征
            
        Returns:
            相似性损失
        """
        try:
            original_text_tensor = torch.tensor(
                original_text_features,
                device=image_features.device,
                dtype=image_features.dtype
            )
            
            if self.config.attack_mode == "targeted" and target_text_features is not None:
                # Targeted攻击：最大化与目标文本的相似性
                target_text_tensor = torch.tensor(
                    target_text_features,
                    device=image_features.device,
                    dtype=image_features.dtype
                )
                
                target_similarity = F.cosine_similarity(
                    image_features, target_text_tensor, dim=0
                )
                
                similarity_loss = -target_similarity  # 最大化相似性
            else:
                # Untargeted攻击：最小化与原始文本的相似性
                original_similarity = F.cosine_similarity(
                    image_features, original_text_tensor, dim=0
                )
                
                similarity_loss = original_similarity  # 最小化相似性
            
            return similarity_loss
            
        except Exception as e:
            logger.error(f"计算相似性损失失败: {e}")
            return torch.tensor(0.0, device=image_features.device)
    
    def _apply_norm_constraint(self, perturbation: torch.Tensor) -> torch.Tensor:
        """
        应用范数约束
        
        Args:
            perturbation: 扰动张量
            
        Returns:
            约束后的扰动
        """
        try:
            if self.config.norm_constraint == "l2":
                norm = torch.norm(perturbation)
                if norm > self.config.epsilon:
                    perturbation = perturbation * (self.config.epsilon / norm)
            elif self.config.norm_constraint == "linf":
                perturbation = torch.clamp(
                    perturbation, 
                    -self.config.epsilon, 
                    self.config.epsilon
                )
            elif self.config.norm_constraint == "l1":
                # L1约束的投影（简化版本）
                flat_perturbation = perturbation.flatten()
                l1_norm = torch.sum(torch.abs(flat_perturbation))
                if l1_norm > self.config.epsilon:
                    perturbation = perturbation * (self.config.epsilon / l1_norm)
            
            return perturbation
            
        except Exception as e:
            logger.error(f"应用范数约束失败: {e}")
            return perturbation
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        将张量转换为PIL图像
        
        Args:
            tensor: 图像张量
            
        Returns:
            PIL图像
        """
        try:
            # 假设tensor是[C, H, W]格式，值在[0, 1]范围内
            tensor = tensor.detach().cpu()
            
            # 转换为numpy数组
            if tensor.dim() == 4:  # [B, C, H, W]
                tensor = tensor.squeeze(0)
            
            # 转换为[H, W, C]格式
            if tensor.shape[0] == 3:  # [C, H, W]
                tensor = tensor.permute(1, 2, 0)
            
            # 转换为0-255范围
            array = (tensor.numpy() * 255).astype(np.uint8)
            
            # 创建PIL图像
            image = Image.fromarray(array)
            
            return image
            
        except Exception as e:
            logger.error(f"张量转图像失败: {e}")
            # 返回空白图像
            return Image.new('RGB', (224, 224), color='white')
    
    def _get_cache_key(self, image: Union[Image.Image, torch.Tensor], 
                      text: str, target_text: Optional[str]) -> str:
        """
        生成缓存键
        
        Args:
            image: 输入图像
            text: 输入文本
            target_text: 目标文本
            
        Returns:
            缓存键
        """
        # 简化的缓存键生成
        text_hash = hash(text)
        target_hash = hash(target_text) if target_text else 0
        
        # 对于图像，使用简单的哈希
        if isinstance(image, torch.Tensor):
            image_hash = hash(image.cpu().numpy().tobytes())
        else:
            image_array = np.array(image)
            image_hash = hash(image_array.tobytes())
        
        config_hash = hash(str(self.config.__dict__))
        
        return f"{text_hash}_{target_hash}_{image_hash}_{config_hash}"
    
    def evaluate_attack_success(self, original_image: Union[Image.Image, torch.Tensor],
                               adversarial_image: Union[Image.Image, torch.Tensor],
                               text: str) -> Dict[str, Any]:
        """
        评估攻击成功率
        
        Args:
            original_image: 原始图像
            adversarial_image: 对抗性图像
            text: 文本
            
        Returns:
            评估结果
        """
        try:
            # 计算原始相似性
            original_similarity = self.clip_model.compute_similarity(text, original_image)
            
            # 计算对抗性相似性
            adversarial_similarity = self.clip_model.compute_similarity(text, adversarial_image)
            
            # 计算hubness
            adversarial_features = self.clip_model.encode_image(adversarial_image)
            hubness = self.compute_hubness(adversarial_features)
            
            # 计算扰动大小
            if isinstance(original_image, Image.Image):
                orig_array = np.array(original_image)
            else:
                orig_array = original_image.cpu().numpy()
            
            if isinstance(adversarial_image, Image.Image):
                adv_array = np.array(adversarial_image)
            else:
                adv_array = adversarial_image.cpu().numpy()
            
            perturbation_l2 = np.linalg.norm(adv_array - orig_array)
            perturbation_linf = np.max(np.abs(adv_array - orig_array))
            
            # 判断攻击成功
            success = (
                hubness >= self.config.success_threshold and
                abs(adversarial_similarity - original_similarity) >= 0.1
            )
            
            return {
                'success': success,
                'original_similarity': float(original_similarity),
                'adversarial_similarity': float(adversarial_similarity),
                'similarity_change': float(adversarial_similarity - original_similarity),
                'hubness': float(hubness),
                'perturbation_l2': float(perturbation_l2),
                'perturbation_linf': float(perturbation_linf)
            }
            
        except Exception as e:
            logger.error(f"评估攻击成功率失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
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
            result = self.create_adversarial_hub(image, text, target_text)
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """
        清理缓存
        """
        self.attack_cache.clear()
        self.hubness_cache.clear()
        logger.info("Hubness攻击器缓存已清理")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'attack_stats': self.attack_stats.copy(),
            'cache_size': len(self.attack_cache),
            'config': {
                'epsilon': self.config.epsilon,
                'num_iterations': self.config.num_iterations,
                'k_neighbors': self.config.k_neighbors,
                'hubness_threshold': self.config.hubness_threshold,
                'attack_mode': self.config.attack_mode
            }
        }


class AdaptiveHubnessAttacker(HubnessAttacker):
    """自适应Hubness攻击器"""
    
    def __init__(self, config: HubnessAttackConfig):
        super().__init__(config)
        
        # 自适应参数
        self.adaptation_history = []
        self.success_rates = []
        
        logger.info("自适应Hubness攻击器初始化完成")
    
    def adaptive_attack(self, image: Union[Image.Image, torch.Tensor], 
                       text: str, 
                       adaptation_steps: int = 5) -> Dict[str, Any]:
        """
        自适应攻击
        
        Args:
            image: 输入图像
            text: 输入文本
            adaptation_steps: 自适应步数
            
        Returns:
            攻击结果
        """
        try:
            best_result = None
            best_success = False
            
            for step in range(adaptation_steps):
                # 根据历史调整参数
                self._adapt_parameters(step)
                
                # 执行攻击
                result = self.create_adversarial_hub(image, text)
                
                # 记录结果
                self.adaptation_history.append({
                    'step': step,
                    'config': self.config.__dict__.copy(),
                    'success': result['success'],
                    'hubness': result['hubness']
                })
                
                if result['success'] and (not best_success or 
                    result['hubness'] > best_result['hubness']):
                    best_result = result
                    best_success = True
                elif not best_success:
                    best_result = result
                
                # 早停条件
                if result['success'] and result['hubness'] >= self.config.target_hubness:
                    break
            
            # 更新成功率
            recent_successes = [h['success'] for h in self.adaptation_history[-10:]]
            success_rate = sum(recent_successes) / len(recent_successes)
            self.success_rates.append(success_rate)
            
            best_result['adaptation_steps'] = step + 1
            best_result['final_success_rate'] = success_rate
            
            return best_result
            
        except Exception as e:
            logger.error(f"自适应攻击失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _adapt_parameters(self, step: int):
        """
        自适应调整参数
        
        Args:
            step: 当前步数
        """
        try:
            if not self.adaptation_history:
                return
            
            # 分析最近的成功率
            recent_history = self.adaptation_history[-5:]
            recent_success_rate = sum(h['success'] for h in recent_history) / len(recent_history)
            
            # 根据成功率调整参数
            if recent_success_rate < 0.3:
                # 成功率低，增加攻击强度
                self.config.epsilon = min(self.config.epsilon * 1.1, 0.5)
                self.config.num_iterations = min(self.config.num_iterations + 10, 200)
                self.config.learning_rate = min(self.config.learning_rate * 1.1, 0.01)
            elif recent_success_rate > 0.8:
                # 成功率高，减少攻击强度以提高隐蔽性
                self.config.epsilon = max(self.config.epsilon * 0.9, 0.01)
                self.config.num_iterations = max(self.config.num_iterations - 5, 20)
                self.config.learning_rate = max(self.config.learning_rate * 0.9, 0.0001)
            
            # 调整hubness相关参数
            recent_hubness = [h['hubness'] for h in recent_history if h['hubness'] > 0]
            if recent_hubness:
                avg_hubness = np.mean(recent_hubness)
                if avg_hubness < self.config.hubness_threshold:
                    self.config.target_hubness = min(self.config.target_hubness + 0.05, 1.0)
                else:
                    self.config.target_hubness = max(self.config.target_hubness - 0.02, 0.5)
            
            logger.debug(f"参数自适应调整 (步骤 {step}): epsilon={self.config.epsilon:.3f}, "
                        f"iterations={self.config.num_iterations}, lr={self.config.learning_rate:.4f}")
            
        except Exception as e:
            logger.error(f"参数自适应调整失败: {e}")


def create_hubness_attacker(config: Optional[HubnessAttackConfig] = None, 
                           adaptive: bool = False) -> Union[HubnessAttacker, AdaptiveHubnessAttacker]:
    """
    创建Hubness攻击器实例
    
    Args:
        config: 攻击配置
        adaptive: 是否使用自适应攻击器
        
    Returns:
        Hubness攻击器实例
    """
    if config is None:
        config = HubnessAttackConfig()
    
    if adaptive:
        return AdaptiveHubnessAttacker(config)
    else:
        return HubnessAttacker(config)