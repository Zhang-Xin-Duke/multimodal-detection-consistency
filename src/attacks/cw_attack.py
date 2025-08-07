"""C&W图像攻击模块

实现基于C&W (Carlini & Wagner) 的图像对抗性攻击方法。
参考论文: "Towards Evaluating the Robustness of Neural Networks" (Carlini & Wagner, S&P 2017)
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
class CWAttackConfig:
    """C&W攻击配置
    
    参数设置严格按照原论文 Carlini & Wagner, S&P 2017
    """
    # 基础设置
    random_seed: int = 42
    device: str = 'cuda'
    
    # C&W核心参数 (原论文设置)
    c: float = 1.0                    # 正则化参数，论文中建议从1开始
    kappa: float = 0.0                # 置信度参数，论文中使用0
    max_iterations: int = 1000        # 最大迭代次数，论文中使用1000
    learning_rate: float = 0.01       # 学习率，论文中使用0.01
    
    # 二分搜索参数
    binary_search_steps: int = 9      # 二分搜索步数，论文中使用9
    initial_const: float = 1e-3       # 初始常数
    
    # 攻击目标
    targeted: bool = False            # 是否为目标攻击
    
    # 约束参数
    clip_min: float = 0.0
    clip_max: float = 1.0
    
    # 优化参数
    abort_early: bool = True          # 是否提前终止
    
    # 评估参数
    batch_size: int = 32
    
    # 缓存配置
    enable_cache: bool = True
    
    # 批处理和多GPU优化
    enable_multi_gpu: bool = False    # 是否启用多GPU并行
    gpu_ids: Optional[List[int]] = None  # GPU设备ID列表，None表示使用所有可用GPU
    batch_size_per_gpu: int = 8       # 每个GPU的批处理大小
    num_workers: int = 4              # 数据加载器工作进程数
    
    # 内存优化
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    mixed_precision: bool = True      # 是否使用混合精度训练
    pin_memory: bool = True           # 是否使用固定内存
    gradient_clip_value: float = 1.0  # 梯度裁剪阈值
    cache_size: int = 1000
    
    # 损失函数类型
    loss_type: str = 'cosine'         # 'cosine', 'mse'
    
    # 优化器类型
    optimizer_type: str = 'adam'      # 'adam', 'sgd'

class CWAttacker:
    """C&W攻击器
    
    实现Carlini & Wagner攻击方法，基于 Carlini & Wagner, S&P 2017
    """
    
    def __init__(self, clip_model, config: CWAttackConfig):
        """
        初始化C&W攻击器
        
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
            'average_iterations': 0.0,
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
        
        logging.info(f"初始化C&W攻击器，设备: {self.device}")
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
        
        logging.info(f"C&W攻击器初始化完成，配置: {config}")
    
    def attack(self, image: Union[Image.Image, torch.Tensor], 
               text: str, 
               target_text: Optional[str] = None) -> Dict[str, Any]:
        """
        执行C&W攻击
        
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
        
        # 计算原始相似度
        with torch.no_grad():
            if hasattr(self.clip_model, 'encode_image_tensor'):
                original_image_features = self.clip_model.encode_image_tensor(image_tensor)
            else:
                original_image_features = self.clip_model.encode_image(image_tensor)
            
            if isinstance(original_image_features, torch.Tensor):
                original_image_features = original_image_features.to(self.device)
            
            original_similarity = torch.nn.functional.cosine_similarity(
                original_image_features, text_features, dim=-1
            ).mean().item()
        
        # 执行C&W攻击
        adversarial_image, attack_info = self._cw_attack(
            image_tensor, text_features, target_features
        )
        
        # 计算对抗相似度
        with torch.no_grad():
            if hasattr(self.clip_model, 'encode_image_tensor'):
                adversarial_image_features = self.clip_model.encode_image_tensor(adversarial_image)
            else:
                adversarial_image_features = self.clip_model.encode_image(adversarial_image)
            
            if isinstance(adversarial_image_features, torch.Tensor):
                adversarial_image_features = adversarial_image_features.to(self.device)
            
            adversarial_similarity = torch.nn.functional.cosine_similarity(
                adversarial_image_features, text_features, dim=-1
            ).mean().item()
        
        # 添加相似度信息到攻击信息中
        attack_info['original_similarity'] = original_similarity
        attack_info['adversarial_similarity'] = adversarial_similarity
        
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
    
    def _cw_attack(self, image: torch.Tensor, 
                   text_features: torch.Tensor,
                   target_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        执行C&W攻击的核心逻辑
        
        基于原论文的优化目标:
        minimize ||δ||_p + c * f(x + δ)
        其中 f(x + δ) 是攻击目标函数
        
        Args:
            image: 输入图像张量
            text_features: 文本特征
            target_features: 目标文本特征（可选）
        
        Returns:
            对抗样本和攻击信息
        """
        batch_size = image.shape[0]
        
        # 初始化变量
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.ones(batch_size, device=self.device) * 1e10
        const = torch.ones(batch_size, device=self.device) * self.config.initial_const
        
        best_l2 = torch.ones(batch_size, device=self.device) * 1e10
        best_attack = image.clone()
        
        attack_info = {
            'iterations': 0,
            'loss_history': [],
            'similarity_history': [],
            'perturbation_norm': 0.0,
            'binary_search_steps': 0
        }
        
        # 二分搜索找到最优的c值
        for binary_step in range(self.config.binary_search_steps):
            attack_info['binary_search_steps'] = binary_step + 1
            
            # 对每个c值进行优化
            adv_image, step_info = self._optimize_cw(
                image, text_features, target_features, const
            )
            
            # 更新攻击信息
            attack_info['iterations'] += step_info['iterations']
            attack_info['loss_history'].extend(step_info['loss_history'])
            attack_info['similarity_history'].extend(step_info['similarity_history'])
            
            # 计算L2距离
            l2_dist = torch.norm((adv_image - image).view(batch_size, -1), p=2, dim=1)
            
            # 检查攻击是否成功
            success_mask = self._check_success_batch(adv_image, text_features, target_features)
            
            # 更新最佳攻击
            for i in range(batch_size):
                if success_mask[i] and l2_dist[i] < best_l2[i]:
                    best_l2[i] = l2_dist[i]
                    best_attack[i] = adv_image[i]
                
                # 更新二分搜索边界
                if success_mask[i]:
                    upper_bound[i] = const[i]
                else:
                    lower_bound[i] = const[i]
                
                # 更新c值
                if upper_bound[i] < 1e9:
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    const[i] = lower_bound[i] * 10
        
        # 计算最终扰动范数
        final_perturbation = best_attack - image
        attack_info['perturbation_norm'] = torch.norm(
            final_perturbation.view(batch_size, -1), p=2, dim=1
        ).mean().item()
        
        return best_attack, attack_info
    
    def _optimize_cw(self, image: torch.Tensor, 
                     text_features: torch.Tensor,
                     target_features: Optional[torch.Tensor],
                     const: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        对给定的c值进行C&W优化
        
        Args:
            image: 输入图像
            text_features: 文本特征
            target_features: 目标文本特征
            const: 正则化常数
        
        Returns:
            优化后的对抗样本和信息
        """
        # 使用tanh空间进行优化（确保像素值在[0,1]范围内）
        # w = arctanh(2x - 1), x = (tanh(w) + 1) / 2
        w = torch.arctanh((image * 2 - 1) * 0.999999)  # 避免数值问题
        w = w.clone().detach().requires_grad_(True)
        
        # 设置优化器
        if self.config.optimizer_type == 'adam':
            optimizer = torch.optim.Adam([w], lr=self.config.learning_rate)
        else:
            optimizer = torch.optim.SGD([w], lr=self.config.learning_rate)
        
        step_info = {
            'iterations': 0,
            'loss_history': [],
            'similarity_history': []
        }
        
        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()
            
            # 将w转换回图像空间
            adv_image = (torch.tanh(w) + 1) / 2
            
            # 计算图像特征
            if hasattr(self.clip_model, 'encode_image_tensor'):
                image_features = self.clip_model.encode_image_tensor(adv_image)
            else:
                image_features = self.clip_model.encode_image(adv_image)
            
            if isinstance(image_features, torch.Tensor):
                image_features = image_features.to(self.device)
            
            # 计算攻击目标损失
            if self.config.targeted and target_features is not None:
                # 目标攻击：最大化与目标文本的相似度
                if self.config.loss_type == 'cosine':
                    attack_loss = -F.cosine_similarity(
                        image_features, target_features, dim=-1
                    ).mean()
                else:
                    attack_loss = F.mse_loss(image_features, target_features)
            else:
                # 非目标攻击：最小化与原始文本的相似度
                if self.config.loss_type == 'cosine':
                    attack_loss = F.cosine_similarity(
                        image_features, text_features, dim=-1
                    ).mean()
                else:
                    attack_loss = -F.mse_loss(image_features, text_features)
            
            # 添加kappa项（置信度）
            attack_loss = torch.clamp(attack_loss - self.config.kappa, min=0)
            
            # 计算L2距离损失
            l2_loss = torch.norm((adv_image - image).view(image.shape[0], -1), p=2, dim=1)
            
            # 总损失：L2距离 + c * 攻击损失
            total_loss = l2_loss.mean() + const.mean() * attack_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 记录信息
            step_info['iterations'] = iteration + 1
            step_info['loss_history'].append(total_loss.item())
            
            # 计算当前相似度
            with torch.no_grad():
                current_similarity = F.cosine_similarity(
                    image_features, text_features, dim=-1
                ).mean().item()
                step_info['similarity_history'].append(current_similarity)
            
            # 提前终止条件
            if self.config.abort_early and iteration > 100:
                if len(step_info['loss_history']) > 10:
                    recent_losses = step_info['loss_history'][-10:]
                    if max(recent_losses) - min(recent_losses) < 1e-6:
                        break
        
        # 返回最终的对抗样本
        with torch.no_grad():
            final_adv_image = (torch.tanh(w) + 1) / 2
        
        return final_adv_image, step_info
    
    def _check_success_batch(self, adversarial_images: torch.Tensor,
                            text_features: torch.Tensor,
                            target_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        批量检查攻击是否成功
        
        Args:
            adversarial_images: 对抗样本批次
            text_features: 原始文本特征
            target_features: 目标文本特征（可选）
        
        Returns:
            成功掩码张量
        """
        with torch.no_grad():
            if hasattr(self.clip_model, 'encode_image_tensor'):
                adv_features = self.clip_model.encode_image_tensor(adversarial_images)
            else:
                adv_features = self.clip_model.encode_image(adversarial_images)
            
            if isinstance(adv_features, torch.Tensor):
                adv_features = adv_features.to(self.device)
            
            if self.config.targeted and target_features is not None:
                # 目标攻击：检查是否更接近目标文本
                target_sim = F.cosine_similarity(adv_features, target_features, dim=-1)
                original_sim = F.cosine_similarity(adv_features, text_features, dim=-1)
                return target_sim > original_sim
            else:
                # 非目标攻击：检查相似度是否显著下降
                similarity = F.cosine_similarity(adv_features, text_features, dim=-1)
                return similarity < 0.5  # 可根据需要调整阈值
    
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
        success_mask = self._check_success_batch(
            adversarial_image, text_features, target_features
        )
        return success_mask.any().item()
    
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
        
        return f"cw_{image_hash}_{text_hash}_{target_hash}_{self.config.c}_{self.config.kappa}"
    
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
        
        # 更新平均迭代次数
        prev_avg_iter = self.attack_stats['average_iterations']
        self.attack_stats['average_iterations'] = (
            (prev_avg_iter * (total - 1) + attack_info['iterations']) / total
        )
        
        # 更新平均相似度下降
        if attack_info['similarity_history']:
            original_sim = attack_info['similarity_history'][0]
            final_sim = attack_info['similarity_history'][-1]
            similarity_drop = original_sim - final_sim
            
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
        批量执行C&W攻击（真正的批处理和多GPU并行）
        
        Args:
            images: 图像列表
            texts: 文本列表
            target_texts: 目标文本列表（可选）
        
        Returns:
            攻击结果列表
        """
        start_time = time.time()
        logging.info(f"开始批量C&W攻击，样本数: {len(images)}")
        
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
        with tqdm(total=total_samples, desc="C&W批量攻击") as pbar:
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                batch_imgs = batch_images[i:end_idx]
                batch_texts = texts[i:end_idx]
                batch_targets = target_texts[i:end_idx] if target_texts else None
                
                # 执行批量攻击
                batch_results = self._batch_cw_attack(
                    batch_imgs, batch_texts, batch_targets
                )
                results.extend(batch_results)
                
                pbar.update(end_idx - i)
        
        # 更新统计信息
        for result in results:
            success = result['attack_success']
            self._update_attack_stats(result, success)
        
        end_time = time.time()
        logging.info(f"批量C&W攻击完成，耗时: {end_time - start_time:.2f}秒")
        
        return results
    
    def _batch_cw_attack(self, batch_images: torch.Tensor, 
                        batch_texts: List[str],
                        batch_targets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        执行批量C&W攻击的核心逻辑
        
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
        
        # 初始化C&W攻击参数
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.ones(batch_size, device=self.device) * 1e10
        const = torch.ones(batch_size, device=self.device) * self.config.initial_const
        
        best_l2 = torch.ones(batch_size, device=self.device) * 1e10
        best_attack = batch_images.clone()
        
        # 混合精度训练
        if self.config.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        
        # 二分搜索找到最优的c值
        for binary_step in range(self.config.binary_search_steps):
            # 初始化对抗样本（使用arctanh变换）
            w = torch.zeros_like(batch_images, requires_grad=True)
            
            # 优化器
            if self.config.optimizer_type == 'adam':
                optimizer = torch.optim.Adam([w], lr=self.config.learning_rate)
            else:
                optimizer = torch.optim.SGD([w], lr=self.config.learning_rate)
            
            # 迭代优化
            for iteration in range(self.config.max_iterations):
                optimizer.zero_grad()
                
                # 将w转换为对抗样本
                adv_images = 0.5 * (torch.tanh(w) + 1)
                adv_images = torch.clamp(adv_images, self.config.clip_min, self.config.clip_max)
                
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    # 编码对抗图像特征
                    adv_image_features = self.clip_model.encode_image(adv_images)
                    adv_image_features = adv_image_features / adv_image_features.norm(dim=-1, keepdim=True)
                    
                    # 计算L2距离
                    l2_dist = torch.norm((adv_images - batch_images).view(batch_size, -1), p=2, dim=-1)
                    
                    # 计算攻击损失
                    if self.config.targeted and target_features is not None:
                        # 目标攻击：最大化与目标文本的相似度
                        attack_loss = -torch.sum(adv_image_features * target_features, dim=-1)
                    else:
                        # 非目标攻击：最小化与原始文本的相似度
                        attack_loss = torch.sum(adv_image_features * text_features, dim=-1)
                    
                    # C&W损失函数
                    f_loss = torch.clamp(attack_loss - self.config.kappa, min=0)
                    total_loss = l2_dist + const * f_loss
                    loss = total_loss.mean()
                
                # 反向传播
                if self.config.mixed_precision:
                    scaler.scale(loss).backward()
                    if self.config.gradient_clip_value > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_([w], self.config.gradient_clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.config.gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_([w], self.config.gradient_clip_value)
                    optimizer.step()
                
                # 检查是否找到更好的攻击
                with torch.no_grad():
                    success_mask = self._check_success_batch(adv_images, text_features, target_features)
                    better_mask = (l2_dist < best_l2) & success_mask
                    
                    best_l2[better_mask] = l2_dist[better_mask]
                    best_attack[better_mask] = adv_images[better_mask]
                
                # 提前终止检查
                if self.config.abort_early and iteration > 100:
                    if torch.all(success_mask):
                        break
            
            # 更新二分搜索边界
            with torch.no_grad():
                success_mask = self._check_success_batch(best_attack, text_features, target_features)
                
                # 成功的样本：降低const
                upper_bound[success_mask] = torch.min(upper_bound[success_mask], const[success_mask])
                
                # 失败的样本：提高const
                fail_mask = ~success_mask
                lower_bound[fail_mask] = torch.max(lower_bound[fail_mask], const[fail_mask])
                
                # 更新const
                finite_upper = upper_bound < 1e9
                const[finite_upper] = (lower_bound[finite_upper] + upper_bound[finite_upper]) / 2
                const[~finite_upper] = const[~finite_upper] * 10
        
        # 评估最终攻击结果
        with torch.no_grad():
            final_image_features = self.clip_model.encode_image(best_attack)
            final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)
            
            # 计算最终相似度
            final_similarities = torch.sum(final_image_features * text_features, dim=-1)
            
            # 计算扰动强度
            perturbations = (best_attack - batch_images).view(batch_size, -1)
            perturbation_norms = torch.norm(perturbations, p=2, dim=-1)
        
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
                'adversarial_image': best_attack[i].cpu(),
                'original_image': batch_images[i].cpu(),
                'perturbation': (best_attack[i] - batch_images[i]).cpu(),
                'perturbation_norm': perturbation_norms[i].item(),
                'original_similarity': original_similarities[i].item(),
                'adversarial_similarity': final_similarities[i].item(),
                'attack_success': attack_success.item() if isinstance(attack_success, torch.Tensor) else attack_success,
                'text': batch_texts[i],
                'target_text': batch_targets[i] if batch_targets else None,
                'attack_method': 'C&W',
                'iterations': self.config.max_iterations,
                'config': self.config
            }
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
            logging.info("C&W攻击器缓存已清空")

def create_cw_attacker(clip_model, config: Optional[CWAttackConfig] = None) -> CWAttacker:
    """
    创建C&W攻击器的工厂函数
    
    Args:
        clip_model: CLIP模型实例
        config: 攻击配置，如果为None则使用默认配置
    
    Returns:
        C&W攻击器实例
    """
    if config is None:
        config = CWAttackConfig()
    
    return CWAttacker(clip_model, config)

# 预定义的配置
class CWAttackPresets:
    """C&W攻击预设配置"""
    
    @staticmethod
    def weak_attack() -> CWAttackConfig:
        """弱攻击配置"""
        return CWAttackConfig(
            c=0.1,
            max_iterations=500,
            learning_rate=0.005
        )
    
    @staticmethod
    def strong_attack() -> CWAttackConfig:
        """强攻击配置"""
        return CWAttackConfig(
            c=10.0,
            max_iterations=2000,
            learning_rate=0.02
        )
    
    @staticmethod
    def targeted_attack() -> CWAttackConfig:
        """目标攻击配置"""
        return CWAttackConfig(
            c=1.0,
            targeted=True,
            max_iterations=1000,
            learning_rate=0.01
        )
    
    @staticmethod
    def paper_config() -> CWAttackConfig:
        """原论文配置 (Carlini & Wagner, S&P 2017)"""
        return CWAttackConfig(
            c=1.0,
            kappa=0.0,
            max_iterations=1000,
            learning_rate=0.01,
            binary_search_steps=9,
            initial_const=1e-3,
            targeted=False,
            abort_early=True,
            optimizer_type='adam'
        )
    
    @staticmethod
    def fast_config() -> CWAttackConfig:
        """快速配置（用于测试）"""
        return CWAttackConfig(
            c=1.0,
            max_iterations=100,
            learning_rate=0.05,
            binary_search_steps=3
        )