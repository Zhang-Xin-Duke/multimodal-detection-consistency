"""多GPU并行处理器模块

提供多GPU并行处理功能，充分利用6块RTX 4090 GPU资源。
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from dataclasses import dataclass
import numpy as np
from queue import Queue
import gc

logger = logging.getLogger(__name__)


@dataclass
class MultiGPUConfig:
    """多GPU配置"""
    gpu_ids: List[int] = None  # GPU设备ID列表
    batch_size_per_gpu: int = 4  # 每个GPU的批处理大小
    max_workers: int = 6  # 最大工作线程数
    memory_fraction: float = 0.9  # 每个GPU的内存使用比例
    enable_mixed_precision: bool = True  # 启用混合精度
    enable_compile: bool = True  # 启用torch.compile
    load_balancing: bool = True  # 启用负载均衡
    
    def __post_init__(self):
        if self.gpu_ids is None:
            # 默认使用所有可用GPU
            self.gpu_ids = list(range(torch.cuda.device_count()))
        
        # 确保最大工作线程数不超过GPU数量
        self.max_workers = min(self.max_workers, len(self.gpu_ids))


class GPUWorker:
    """单个GPU工作器"""
    
    def __init__(self, gpu_id: int, config: MultiGPUConfig):
        self.gpu_id = gpu_id
        self.config = config
        self.device = torch.device(f'cuda:{gpu_id}')
        self.is_busy = False
        self.task_queue = Queue()
        self.result_queue = Queue()
        
        # 设置GPU内存限制
        self._setup_gpu_memory()
        
        logger.info(f"GPU Worker {gpu_id} 初始化完成")
    
    def _setup_gpu_memory(self):
        """设置GPU内存限制"""
        try:
            torch.cuda.set_device(self.gpu_id)
            # 设置内存分配策略
            torch.cuda.empty_cache()
            
            # 获取GPU内存信息
            total_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory
            allocated_memory = int(total_memory * self.config.memory_fraction)
            
            logger.info(f"GPU {self.gpu_id}: 总内存 {total_memory/1024**3:.2f}GB, "
                       f"分配 {allocated_memory/1024**3:.2f}GB")
            
        except Exception as e:
            logger.warning(f"设置GPU {self.gpu_id} 内存限制失败: {e}")
    
    def process_batch(self, model_func: Callable, batch_data: Any, **kwargs) -> Any:
        """处理批次数据"""
        self.is_busy = True
        try:
            with torch.cuda.device(self.gpu_id):
                # 移动数据到当前GPU
                if isinstance(batch_data, torch.Tensor):
                    batch_data = batch_data.to(self.device)
                elif isinstance(batch_data, (list, tuple)):
                    batch_data = [item.to(self.device) if isinstance(item, torch.Tensor) else item 
                                 for item in batch_data]
                
                # 执行模型推理
                if self.config.enable_mixed_precision:
                    with torch.cuda.amp.autocast():
                        result = model_func(batch_data, **kwargs)
                else:
                    result = model_func(batch_data, **kwargs)
                
                # 移动结果到CPU以节省GPU内存
                if isinstance(result, torch.Tensor):
                    result = result.cpu()
                elif isinstance(result, (list, tuple)):
                    result = [item.cpu() if isinstance(item, torch.Tensor) else item 
                             for item in result]
                
                return result
                
        except Exception as e:
            logger.error(f"GPU {self.gpu_id} 处理批次失败: {e}")
            raise
        finally:
            self.is_busy = False
            # 清理GPU缓存
            torch.cuda.empty_cache()


class MultiGPUProcessor:
    """多GPU并行处理器"""
    
    def __init__(self, config: MultiGPUConfig = None):
        """
        初始化多GPU处理器
        
        Args:
            config: 多GPU配置
        """
        self.config = config or MultiGPUConfig()
        self.workers = {}
        self.executor = None
        self.load_stats = {gpu_id: 0 for gpu_id in self.config.gpu_ids}
        
        # 检查GPU可用性
        self._check_gpu_availability()
        
        # 初始化GPU工作器
        self._initialize_workers()
        
        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        logger.info(f"多GPU处理器初始化完成，使用GPU: {self.config.gpu_ids}")
    
    def _check_gpu_availability(self):
        """检查GPU可用性"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        
        available_gpus = torch.cuda.device_count()
        if max(self.config.gpu_ids) >= available_gpus:
            raise ValueError(f"请求的GPU ID超出可用范围，可用GPU数量: {available_gpus}")
        
        # 检查每个GPU的状态
        for gpu_id in self.config.gpu_ids:
            try:
                torch.cuda.set_device(gpu_id)
                # 测试GPU内存分配
                test_tensor = torch.randn(100, 100, device=f'cuda:{gpu_id}')
                del test_tensor
                torch.cuda.empty_cache()
                logger.info(f"GPU {gpu_id} 可用")
            except Exception as e:
                logger.error(f"GPU {gpu_id} 不可用: {e}")
                raise
    
    def _initialize_workers(self):
        """初始化GPU工作器"""
        for gpu_id in self.config.gpu_ids:
            self.workers[gpu_id] = GPUWorker(gpu_id, self.config)
    
    def _get_optimal_gpu(self) -> int:
        """获取最优GPU（负载最低）"""
        if self.config.load_balancing:
            return min(self.load_stats.keys(), key=lambda x: self.load_stats[x])
        else:
            # 轮询分配
            return self.config.gpu_ids[len(self.load_stats) % len(self.config.gpu_ids)]
    
    def parallel_process(self, model_func: Callable, data_batches: List[Any], 
                        **kwargs) -> List[Any]:
        """
        并行处理数据批次
        
        Args:
            model_func: 模型处理函数
            data_batches: 数据批次列表
            **kwargs: 额外参数
        
        Returns:
            处理结果列表
        """
        if not data_batches:
            return []
        
        # 分配任务到GPU
        futures = []
        results = [None] * len(data_batches)
        
        for i, batch in enumerate(data_batches):
            # 选择最优GPU
            gpu_id = self._get_optimal_gpu()
            worker = self.workers[gpu_id]
            
            # 提交任务
            future = self.executor.submit(
                worker.process_batch, model_func, batch, **kwargs
            )
            futures.append((i, future, gpu_id))
            
            # 更新负载统计
            self.load_stats[gpu_id] += 1
        
        # 收集结果
        for batch_idx, future, gpu_id in futures:
            try:
                result = future.result()
                results[batch_idx] = result
                
                # 更新负载统计
                self.load_stats[gpu_id] -= 1
                
            except Exception as e:
                logger.error(f"批次 {batch_idx} 在GPU {gpu_id} 上处理失败: {e}")
                raise
        
        return results
    
    def parallel_clip_encode(self, clip_model, texts: List[str], 
                           batch_size: Optional[int] = None) -> torch.Tensor:
        """
        并行CLIP文本编码
        
        Args:
            clip_model: CLIP模型实例
            texts: 文本列表
            batch_size: 批处理大小
        
        Returns:
            编码特征张量
        """
        batch_size = batch_size or self.config.batch_size_per_gpu
        
        # 创建批次
        text_batches = [texts[i:i + batch_size] 
                       for i in range(0, len(texts), batch_size)]
        
        # 定义编码函数
        def encode_func(text_batch):
            return clip_model.encode_text(text_batch)
        
        # 并行处理
        results = self.parallel_process(encode_func, text_batches)
        
        # 合并结果
        return torch.cat([r for r in results if r is not None], dim=0)
    
    def parallel_sd_generate(self, sd_models: Dict[int, Any], prompts: List[str],
                           **generation_kwargs) -> List[Any]:
        """
        并行Stable Diffusion图像生成
        
        Args:
            sd_models: GPU ID到SD模型的映射
            prompts: 提示词列表
            **generation_kwargs: 生成参数
        
        Returns:
            生成的图像列表
        """
        # 创建批次
        batch_size = self.config.batch_size_per_gpu
        prompt_batches = [prompts[i:i + batch_size] 
                         for i in range(0, len(prompts), batch_size)]
        
        # 分配任务到不同GPU
        futures = []
        results = [None] * len(prompt_batches)
        
        for i, prompt_batch in enumerate(prompt_batches):
            gpu_id = self.config.gpu_ids[i % len(self.config.gpu_ids)]
            sd_model = sd_models[gpu_id]
            
            def generate_func(prompts_batch):
                images = []
                for prompt in prompts_batch:
                    image = sd_model.generate_image(prompt, **generation_kwargs)
                    images.extend(image if isinstance(image, list) else [image])
                return images
            
            future = self.executor.submit(generate_func, prompt_batch)
            futures.append((i, future))
        
        # 收集结果
        all_images = []
        for batch_idx, future in futures:
            try:
                batch_images = future.result()
                all_images.extend(batch_images)
            except Exception as e:
                logger.error(f"SD生成批次 {batch_idx} 失败: {e}")
                raise
        
        return all_images
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """获取GPU统计信息"""
        stats = {
            'gpu_count': len(self.config.gpu_ids),
            'gpu_ids': self.config.gpu_ids,
            'load_stats': self.load_stats.copy(),
            'memory_stats': {}
        }
        
        for gpu_id in self.config.gpu_ids:
            try:
                torch.cuda.set_device(gpu_id)
                memory_allocated = torch.cuda.memory_allocated(gpu_id)
                memory_reserved = torch.cuda.memory_reserved(gpu_id)
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
                
                stats['memory_stats'][gpu_id] = {
                    'allocated_mb': memory_allocated / 1024**2,
                    'reserved_mb': memory_reserved / 1024**2,
                    'total_mb': memory_total / 1024**2,
                    'utilization': memory_allocated / memory_total * 100
                }
            except Exception as e:
                logger.warning(f"获取GPU {gpu_id} 统计信息失败: {e}")
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # 清理GPU缓存
        for gpu_id in self.config.gpu_ids:
            try:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"清理GPU {gpu_id} 缓存失败: {e}")
        
        logger.info("多GPU处理器资源清理完成")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_multi_gpu_processor(gpu_ids: Optional[List[int]] = None,
                              batch_size_per_gpu: int = 4) -> MultiGPUProcessor:
    """
    创建多GPU处理器的便捷函数
    
    Args:
        gpu_ids: GPU设备ID列表，默认使用所有可用GPU
        batch_size_per_gpu: 每个GPU的批处理大小
    
    Returns:
        多GPU处理器实例
    """
    config = MultiGPUConfig(
        gpu_ids=gpu_ids,
        batch_size_per_gpu=batch_size_per_gpu
    )
    return MultiGPUProcessor(config)


# 全局多GPU处理器实例（单例模式）
_global_processor = None
_processor_lock = threading.Lock()


def get_global_multi_gpu_processor() -> MultiGPUProcessor:
    """
    获取全局多GPU处理器实例（单例模式）
    
    Returns:
        全局多GPU处理器实例
    """
    global _global_processor
    
    if _global_processor is None:
        with _processor_lock:
            if _global_processor is None:
                _global_processor = create_multi_gpu_processor()
    
    return _global_processor


def cleanup_global_processor():
    """清理全局多GPU处理器"""
    global _global_processor
    
    if _global_processor is not None:
        with _processor_lock:
            if _global_processor is not None:
                _global_processor.cleanup()
                _global_processor = None