"""生成参考图像模块

该模块使用Stable Diffusion模型根据文本描述生成参考图像，
用于与输入图像进行一致性比较，检测对抗攻击。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

@dataclass
class GenerativeConfig:
    """生成参考配置"""
    generation_count: int = 3
    image_size: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    seed: Optional[int] = None
    negative_prompt: str = "blurry, low quality, distorted, deformed"
    use_safety_checker: bool = True
    batch_size: int = 1
    device: str = "cuda"

class GenerativeReferenceGenerator:
    """生成参考图像生成器
    
    使用Stable Diffusion模型生成与文本描述相符的参考图像，
    这些图像用于与输入图像进行跨模态一致性比较。
    """
    
    def __init__(self, 
                 sd_model,
                 clip_model,
                 config: GenerativeConfig = None):
        """
        初始化生成参考生成器
        
        Args:
            sd_model: Stable Diffusion模型实例
            clip_model: CLIP模型实例（用于特征提取）
            config: 生成配置
        """
        self.sd_model = sd_model
        self.clip_model = clip_model
        self.config = config or GenerativeConfig()
        self.device = torch.device(self.config.device)
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 生成统计
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_generation_time': 0.0
        }
        
        logger.info("生成参考生成器初始化完成")
    
    def generate_references(self, text: str) -> List[torch.Tensor]:
        """生成参考图像
        
        Args:
            text: 文本描述
            
        Returns:
            生成的参考图像张量列表
        """
        import time
        start_time = time.time()
        
        generated_images = []
        
        try:
            # 预处理文本
            processed_text = self._preprocess_text(text)
            
            # 生成多个图像
            for i in range(self.config.generation_count):
                try:
                    # 设置随机种子（如果指定）
                    if self.config.seed is not None:
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(self.config.seed + i)
                    else:
                        generator = None
                    
                    # 生成图像
                    generated_image = self._generate_single_image(
                        prompt=processed_text,
                        generator=generator
                    )
                    
                    if generated_image is not None:
                        # 转换为张量
                        image_tensor = self._pil_to_tensor(generated_image)
                        generated_images.append(image_tensor)
                        
                        self.generation_stats['successful_generations'] += 1
                    else:
                        self.generation_stats['failed_generations'] += 1
                        
                except Exception as e:
                    logger.warning(f"生成第 {i+1} 张图像失败: {e}")
                    self.generation_stats['failed_generations'] += 1
                    continue
            
            self.generation_stats['total_generations'] += self.config.generation_count
            
            # 更新平均生成时间
            generation_time = time.time() - start_time
            self._update_average_time(generation_time)
            
            logger.debug(f"为文本 '{text[:50]}...' 成功生成 {len(generated_images)} 张参考图像")
            
        except Exception as e:
            logger.error(f"生成参考图像时出错: {e}")
            generated_images = []
        
        return generated_images
    
    def _generate_single_image(self, 
                              prompt: str, 
                              generator: Optional[torch.Generator] = None) -> Optional[Image.Image]:
        """生成单张图像"""
        try:
            # 调用Stable Diffusion模型
            result = self.sd_model.generate(
                prompt=prompt,
                negative_prompt=self.config.negative_prompt,
                height=self.config.image_size,
                width=self.config.image_size,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator
            )
            
            # 提取图像
            if hasattr(result, 'images') and result.images:
                return result.images[0]
            elif isinstance(result, list) and result:
                return result[0]
            else:
                return result
                
        except Exception as e:
            logger.warning(f"Stable Diffusion生成失败: {e}")
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本提示"""
        # 清理文本
        processed = text.strip()
        
        # 添加质量提升词
        quality_enhancers = [
            "high quality",
            "detailed",
            "realistic",
            "professional photography"
        ]
        
        # 检查是否已包含质量词
        has_quality_words = any(word in processed.lower() for word in quality_enhancers)
        
        if not has_quality_words:
            processed = f"{processed}, high quality, detailed"
        
        # 限制长度
        if len(processed) > 200:  # Stable Diffusion的提示词长度限制
            processed = processed[:200].rsplit(' ', 1)[0]  # 在单词边界截断
        
        return processed
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为张量"""
        # 确保图像是RGB格式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 应用变换
        tensor = self.image_transform(pil_image)
        
        return tensor
    
    def _update_average_time(self, new_time: float):
        """更新平均生成时间"""
        total_gens = self.generation_stats['total_generations']
        if total_gens > 0:
            current_avg = self.generation_stats['average_generation_time']
            # 增量更新平均值
            self.generation_stats['average_generation_time'] = (
                (current_avg * (total_gens - self.config.generation_count) + new_time) / total_gens
            )
    
    def batch_generate_references(self, texts: List[str]) -> List[List[torch.Tensor]]:
        """批量生成参考图像
        
        Args:
            texts: 文本描述列表
            
        Returns:
            每个文本对应的参考图像列表
        """
        all_references = []
        
        for text in texts:
            references = self.generate_references(text)
            all_references.append(references)
        
        return all_references
    
    def compute_generation_consistency(self, 
                                     text: str, 
                                     generated_images: List[torch.Tensor]) -> Dict[str, float]:
        """计算生成图像的一致性
        
        Args:
            text: 原始文本
            generated_images: 生成的图像列表
            
        Returns:
            一致性指标
        """
        if not generated_images:
            return {'text_image_consistency': 0.0, 'inter_image_consistency': 0.0}
        
        # 编码文本
        text_features = self.clip_model.encode_text([text])
        
        # 编码所有生成图像
        image_features_list = []
        for img in generated_images:
            img_features = self.clip_model.encode_image(img.unsqueeze(0))
            image_features_list.append(img_features)
        
        # 计算文本-图像一致性
        text_image_similarities = []
        for img_features in image_features_list:
            similarity = torch.cosine_similarity(text_features, img_features, dim=-1).item()
            text_image_similarities.append(similarity)
        
        text_image_consistency = float(np.mean(text_image_similarities))
        
        # 计算图像间一致性
        inter_image_similarities = []
        for i in range(len(image_features_list)):
            for j in range(i + 1, len(image_features_list)):
                similarity = torch.cosine_similarity(
                    image_features_list[i], 
                    image_features_list[j], 
                    dim=-1
                ).item()
                inter_image_similarities.append(similarity)
        
        inter_image_consistency = float(np.mean(inter_image_similarities)) if inter_image_similarities else 1.0
        
        return {
            'text_image_consistency': text_image_consistency,
            'inter_image_consistency': inter_image_consistency,
            'text_image_std': float(np.std(text_image_similarities)),
            'generation_count': len(generated_images)
        }
    
    def evaluate_generation_quality(self, 
                                  text: str, 
                                  generated_images: List[torch.Tensor],
                                  reference_image: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """评估生成质量
        
        Args:
            text: 原始文本
            generated_images: 生成的图像
            reference_image: 参考图像（可选）
            
        Returns:
            质量评估结果
        """
        if not generated_images:
            return {'message': '无生成图像可评估'}
        
        evaluation = {
            'generation_count': len(generated_images),
            'consistency_metrics': self.compute_generation_consistency(text, generated_images)
        }
        
        # 如果有参考图像，计算与参考图像的相似度
        if reference_image is not None:
            ref_similarities = []
            ref_features = self.clip_model.encode_image(reference_image.unsqueeze(0))
            
            for gen_img in generated_images:
                gen_features = self.clip_model.encode_image(gen_img.unsqueeze(0))
                similarity = torch.cosine_similarity(ref_features, gen_features, dim=-1).item()
                ref_similarities.append(similarity)
            
            evaluation['reference_similarity'] = {
                'mean': float(np.mean(ref_similarities)),
                'std': float(np.std(ref_similarities)),
                'max': float(np.max(ref_similarities)),
                'min': float(np.min(ref_similarities))
            }
        
        # 计算多样性分数
        diversity_score = self._compute_diversity_score(generated_images)
        evaluation['diversity_score'] = diversity_score
        
        # 计算综合质量分数
        quality_score = self._compute_quality_score(evaluation)
        evaluation['overall_quality'] = quality_score
        
        return evaluation
    
    def _compute_diversity_score(self, images: List[torch.Tensor]) -> float:
        """计算图像多样性分数"""
        if len(images) < 2:
            return 0.0
        
        # 编码所有图像
        features_list = []
        for img in images:
            features = self.clip_model.encode_image(img.unsqueeze(0))
            features_list.append(features)
        
        # 计算两两相似度
        similarities = []
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                sim = torch.cosine_similarity(features_list[i], features_list[j], dim=-1).item()
                similarities.append(sim)
        
        # 多样性 = 1 - 平均相似度
        diversity = 1.0 - np.mean(similarities)
        return float(np.clip(diversity, 0.0, 1.0))
    
    def _compute_quality_score(self, evaluation: Dict[str, Any]) -> float:
        """计算综合质量分数"""
        consistency_metrics = evaluation.get('consistency_metrics', {})
        
        # 文本-图像一致性权重
        text_image_score = consistency_metrics.get('text_image_consistency', 0.0)
        
        # 图像间一致性权重（适度一致性更好）
        inter_image_consistency = consistency_metrics.get('inter_image_consistency', 0.0)
        inter_image_score = 1.0 - abs(inter_image_consistency - 0.7)  # 理想值0.7
        
        # 多样性权重
        diversity_score = evaluation.get('diversity_score', 0.0)
        
        # 加权综合
        quality_score = (
            0.5 * text_image_score +
            0.3 * inter_image_score +
            0.2 * diversity_score
        )
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    def save_generated_images(self, 
                            images: List[torch.Tensor], 
                            save_dir: str, 
                            prefix: str = "generated") -> List[str]:
        """保存生成的图像
        
        Args:
            images: 图像张量列表
            save_dir: 保存目录
            prefix: 文件名前缀
            
        Returns:
            保存的文件路径列表
        """
        import os
        from pathlib import Path
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        # 反归一化变换
        denormalize = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                               std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage()
        ])
        
        for i, img_tensor in enumerate(images):
            try:
                # 反归一化并转换为PIL图像
                pil_image = denormalize(img_tensor)
                
                # 保存图像
                filename = f"{prefix}_{i:03d}.png"
                filepath = save_path / filename
                pil_image.save(filepath)
                
                saved_paths.append(str(filepath))
                
            except Exception as e:
                logger.warning(f"保存图像 {i} 失败: {e}")
                continue
        
        logger.info(f"已保存 {len(saved_paths)} 张图像到 {save_dir}")
        return saved_paths
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取生成器统计信息"""
        stats = self.generation_stats.copy()
        
        # 计算成功率
        total = stats['total_generations']
        if total > 0:
            stats['success_rate'] = stats['successful_generations'] / total
            stats['failure_rate'] = stats['failed_generations'] / total
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # 添加配置信息
        stats['config'] = {
            'generation_count': self.config.generation_count,
            'image_size': self.config.image_size,
            'guidance_scale': self.config.guidance_scale,
            'num_inference_steps': self.config.num_inference_steps
        }
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_generation_time': 0.0
        }
        logger.info("生成器统计信息已重置")
    
    def update_config(self, new_config: GenerativeConfig):
        """更新配置"""
        self.config = new_config
        logger.info(f"生成器配置已更新: {new_config}")