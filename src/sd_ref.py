"""
Stable Diffusion参考生成模块

使用Stable Diffusion模型生成参考图像，用于一致性检测。
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import hashlib
from PIL import Image
import time
from .models import StableDiffusionModel, StableDiffusionConfig
from .text_augment import TextAugmenter, TextAugmentConfig
import random

logger = logging.getLogger(__name__)


@dataclass
class SDReferenceConfig:
    """SD参考生成配置"""
    # SD模型配置
    sd_model: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"
    torch_dtype: str = "float16"
    
    # 生成参数
    num_images_per_prompt: int = 3
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    
    # 变体生成
    use_text_variants: bool = True
    num_text_variants: int = 3
    variant_methods: List[str] = None
    
    # 种子控制
    use_fixed_seeds: bool = True
    seed_range: Tuple[int, int] = (0, 10000)
    
    # 质量控制
    enable_safety_checker: bool = False
    filter_low_quality: bool = True
    quality_threshold: float = 0.5
    
    # 缓存配置
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    
    # 批处理
    batch_size: int = 4
    
    def __post_init__(self):
        if self.variant_methods is None:
            self.variant_methods = ['synonym', 'paraphrase']


class SDReferenceGenerator:
    """SD参考图像生成器"""
    
    def __init__(self, config: SDReferenceConfig):
        """
        初始化SD参考生成器
        
        Args:
            config: SD参考生成配置
        """
        self.config = config
        
        # 初始化SD模型
        self.sd_model = self._initialize_sd_model()
        
        # 初始化文本增强器（如果需要）
        self.text_augmenter = None
        if self.config.use_text_variants:
            self.text_augmenter = self._initialize_text_augmenter()
        
        # 缓存
        self.generation_cache = {}
        self.prompt_cache = {}
        
        # 统计信息
        self.generation_stats = {
            'total_generated': 0,
            'cache_hits': 0,
            'generation_time': 0.0
        }
        
        logger.info("SD参考生成器初始化完成")
    
    def _initialize_sd_model(self) -> StableDiffusionModel:
        """
        初始化Stable Diffusion模型
        
        Returns:
            SD模型实例
        """
        try:
            sd_config = StableDiffusionConfig(
                model_name=self.config.sd_model,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=self.config.height,
                width=self.config.width,
                safety_checker=self.config.enable_safety_checker
            )
            
            sd_model = StableDiffusionModel(sd_config)
            logger.info(f"SD模型初始化完成: {self.config.sd_model}")
            
            return sd_model
            
        except Exception as e:
            logger.error(f"SD模型初始化失败: {e}")
            raise
    
    def _initialize_text_augmenter(self) -> TextAugmenter:
        """
        初始化文本增强器
        
        Returns:
            文本增强器实例
        """
        try:
            augment_config = TextAugmentConfig(
                num_variants=self.config.num_text_variants,
                similarity_threshold=0.8
            )
            
            text_augmenter = TextAugmenter(augment_config)
            logger.info("文本增强器初始化完成")
            
            return text_augmenter
            
        except Exception as e:
            logger.warning(f"文本增强器初始化失败: {e}")
            return None
    
    def generate_reference_images(self, prompt: str, 
                                 num_images: Optional[int] = None,
                                 use_variants: Optional[bool] = None,
                                 seeds: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        生成参考图像
        
        Args:
            prompt: 文本提示
            num_images: 生成图像数量
            use_variants: 是否使用文本变体
            seeds: 指定的种子列表
            
        Returns:
            生成结果字典
        """
        try:
            num_images = num_images or self.config.num_images_per_prompt
            use_variants = use_variants if use_variants is not None else self.config.use_text_variants
            
            # 检查缓存
            cache_key = self._get_cache_key(prompt, num_images, use_variants, seeds)
            if self.config.enable_cache and cache_key in self.generation_cache:
                self.generation_stats['cache_hits'] += 1
                return self.generation_cache[cache_key]
            
            start_time = time.time()
            
            # 准备提示列表
            prompts = [prompt]
            if use_variants and self.text_augmenter is not None:
                variants = self.text_augmenter.generate_variants(
                    prompt, 
                    methods=self.config.variant_methods
                )
                prompts.extend(variants[:self.config.num_text_variants])
            
            # 生成种子
            if seeds is None:
                seeds = self._generate_seeds(num_images * len(prompts))
            
            # 生成图像
            all_images = []
            all_prompts_used = []
            all_seeds_used = []
            
            seed_idx = 0
            for prompt_text in prompts:
                for _ in range(num_images):
                    if seed_idx < len(seeds):
                        seed = seeds[seed_idx]
                    else:
                        seed = random.randint(*self.config.seed_range)
                    
                    # 生成单张图像
                    images = self.sd_model.generate_image(
                        prompt=prompt_text,
                        num_images=1,
                        seed=seed,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        height=self.config.height,
                        width=self.config.width
                    )
                    
                    if images:
                        all_images.extend(images)
                        all_prompts_used.append(prompt_text)
                        all_seeds_used.append(seed)
                    
                    seed_idx += 1
            
            # 质量过滤
            if self.config.filter_low_quality:
                filtered_results = self._filter_low_quality_images(
                    all_images, all_prompts_used, all_seeds_used
                )
                all_images, all_prompts_used, all_seeds_used = filtered_results
            
            # 构建结果
            result = {
                'images': all_images,
                'prompts': all_prompts_used,
                'seeds': all_seeds_used,
                'original_prompt': prompt,
                'generation_time': time.time() - start_time,
                'num_generated': len(all_images)
            }
            
            # 缓存结果
            if self.config.enable_cache:
                self.generation_cache[cache_key] = result
            
            # 更新统计信息
            self.generation_stats['total_generated'] += len(all_images)
            self.generation_stats['generation_time'] += result['generation_time']
            
            logger.debug(f"参考图像生成完成: {prompt} -> {len(all_images)}张图像")
            return result
            
        except Exception as e:
            logger.error(f"生成参考图像失败: {e}")
            return {
                'images': [],
                'prompts': [],
                'seeds': [],
                'original_prompt': prompt,
                'generation_time': 0.0,
                'num_generated': 0,
                'error': str(e)
            }
    
    def _get_cache_key(self, prompt: str, num_images: int, 
                      use_variants: bool, seeds: Optional[List[int]]) -> str:
        """
        生成缓存键
        
        Args:
            prompt: 文本提示
            num_images: 图像数量
            use_variants: 是否使用变体
            seeds: 种子列表
            
        Returns:
            缓存键
        """
        key_data = {
            'prompt': prompt,
            'num_images': num_images,
            'use_variants': use_variants,
            'seeds': seeds,
            'config': {
                'sd_model': self.config.sd_model,
                'num_inference_steps': self.config.num_inference_steps,
                'guidance_scale': self.config.guidance_scale,
                'height': self.config.height,
                'width': self.config.width
            }
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _generate_seeds(self, num_seeds: int) -> List[int]:
        """
        生成种子列表
        
        Args:
            num_seeds: 种子数量
            
        Returns:
            种子列表
        """
        if self.config.use_fixed_seeds:
            # 使用固定种子范围
            seeds = list(range(
                self.config.seed_range[0], 
                min(self.config.seed_range[0] + num_seeds, self.config.seed_range[1])
            ))
            
            # 如果需要更多种子，随机生成
            while len(seeds) < num_seeds:
                seed = random.randint(*self.config.seed_range)
                if seed not in seeds:
                    seeds.append(seed)
        else:
            # 随机生成种子
            seeds = [random.randint(*self.config.seed_range) for _ in range(num_seeds)]
        
        return seeds
    
    def _filter_low_quality_images(self, images: List[Image.Image], 
                                  prompts: List[str], 
                                  seeds: List[int]) -> Tuple[List[Image.Image], List[str], List[int]]:
        """
        过滤低质量图像
        
        Args:
            images: 图像列表
            prompts: 提示列表
            seeds: 种子列表
            
        Returns:
            过滤后的(图像, 提示, 种子)列表
        """
        try:
            # 简单的质量评估（可以扩展为更复杂的方法）
            filtered_images = []
            filtered_prompts = []
            filtered_seeds = []
            
            for i, image in enumerate(images):
                # 基本质量检查
                if self._assess_image_quality(image) >= self.config.quality_threshold:
                    filtered_images.append(image)
                    filtered_prompts.append(prompts[i])
                    filtered_seeds.append(seeds[i])
            
            logger.debug(f"质量过滤: {len(images)} -> {len(filtered_images)}张图像")
            return filtered_images, filtered_prompts, filtered_seeds
            
        except Exception as e:
            logger.warning(f"质量过滤失败: {e}")
            return images, prompts, seeds
    
    def _assess_image_quality(self, image: Image.Image) -> float:
        """
        评估图像质量
        
        Args:
            image: PIL图像
            
        Returns:
            质量分数 (0-1)
        """
        try:
            # 简单的质量评估指标
            img_array = np.array(image)
            
            # 1. 检查图像是否为空白
            if img_array.std() < 10:
                return 0.0
            
            # 2. 检查图像对比度
            contrast = img_array.std() / 255.0
            
            # 3. 检查图像亮度分布
            brightness = img_array.mean() / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # 4. 检查颜色丰富度
            if len(img_array.shape) == 3:
                color_variance = np.var(img_array, axis=(0, 1)).mean()
                color_score = min(color_variance / 1000.0, 1.0)
            else:
                color_score = 0.5
            
            # 综合评分
            quality_score = (contrast * 0.4 + brightness_score * 0.3 + color_score * 0.3)
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"图像质量评估失败: {e}")
            return 0.5
    
    def batch_generate_reference_images(self, prompts: List[str], 
                                       num_images_per_prompt: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        批量生成参考图像
        
        Args:
            prompts: 提示列表
            num_images_per_prompt: 每个提示生成的图像数量
            
        Returns:
            生成结果列表
        """
        results = []
        
        for prompt in prompts:
            result = self.generate_reference_images(
                prompt, 
                num_images=num_images_per_prompt
            )
            results.append(result)
        
        return results
    
    def generate_reference_vectors(self, prompt: str, 
                                  num_images: Optional[int] = None) -> np.ndarray:
        """
        生成参考向量（图像特征）
        
        Args:
            prompt: 文本提示
            num_images: 生成图像数量
            
        Returns:
            参考向量矩阵
        """
        try:
            # 生成参考图像
            result = self.generate_reference_images(prompt, num_images)
            images = result['images']
            
            if not images:
                return np.array([])
            
            # 使用SD模型的VAE编码器提取特征
            reference_vectors = []
            
            for image in images:
                # 编码图像为潜在向量
                latent = self.sd_model.encode_image(image)
                # 展平为一维向量
                vector = latent.flatten()
                reference_vectors.append(vector)
            
            reference_vectors = np.array(reference_vectors)
            
            logger.debug(f"参考向量生成完成: {prompt} -> {reference_vectors.shape}")
            return reference_vectors
            
        except Exception as e:
            logger.error(f"生成参考向量失败: {e}")
            return np.array([])
    
    def save_generated_images(self, result: Dict[str, Any], 
                             save_dir: str, prefix: str = "ref") -> List[str]:
        """
        保存生成的图像
        
        Args:
            result: 生成结果
            save_dir: 保存目录
            prefix: 文件名前缀
            
        Returns:
            保存的文件路径列表
        """
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            saved_paths = []
            images = result['images']
            seeds = result['seeds']
            
            for i, (image, seed) in enumerate(zip(images, seeds)):
                filename = f"{prefix}_{i:04d}_seed_{seed}.png"
                save_path = save_dir / filename
                
                self.sd_model.save_image(image, save_path)
                saved_paths.append(str(save_path))
            
            logger.info(f"图像已保存: {len(saved_paths)}张图像到 {save_dir}")
            return saved_paths
            
        except Exception as e:
            logger.error(f"保存图像失败: {e}")
            return []
    
    def load_reference_images(self, image_paths: List[str]) -> List[Image.Image]:
        """
        加载参考图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            图像列表
        """
        images = []
        
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                images.append(image)
            except Exception as e:
                logger.warning(f"加载图像失败 {path}: {e}")
        
        return images
    
    def clear_cache(self):
        """
        清理缓存
        """
        self.generation_cache.clear()
        self.prompt_cache.clear()
        logger.info("SD生成缓存已清理")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'generation_stats': self.generation_stats.copy(),
            'cache_size': len(self.generation_cache),
            'config': {
                'sd_model': self.config.sd_model,
                'num_images_per_prompt': self.config.num_images_per_prompt,
                'use_text_variants': self.config.use_text_variants,
                'num_text_variants': self.config.num_text_variants,
                'enable_cache': self.config.enable_cache
            }
        }
    
    def update_config(self, **kwargs):
        """
        更新配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"配置已更新: {key} = {value}")
            else:
                logger.warning(f"未知配置参数: {key}")


def create_sd_reference_generator(config: Optional[SDReferenceConfig] = None) -> SDReferenceGenerator:
    """
    创建SD参考生成器实例
    
    Args:
        config: SD参考生成配置
        
    Returns:
        SD参考生成器实例
    """
    if config is None:
        config = SDReferenceConfig()
    
    return SDReferenceGenerator(config)