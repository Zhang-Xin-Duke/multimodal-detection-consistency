"""多GPU处理器模块

提供多GPU并行处理功能，包括数据分发、模型并行和结果聚合。
支持动态负载均衡和错误恢复。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import logging
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GPUTask:
    """GPU任务"""
    task_id: str
    data: Any
    gpu_id: int
    priority: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class GPUResult:
    """GPU处理结果"""
    task_id: str
    result: Any
    gpu_id: int
    processing_time: float
    success: bool = True
    error: Optional[str] = None


class GPUWorker:
    """GPU工作器
    
    在指定GPU上执行任务的工作器。
    """
    
    def __init__(self, gpu_id: int, model_factory: Callable, device_config: Dict[str, Any]):
        """初始化GPU工作器
        
        Args:
            gpu_id: GPU ID
            model_factory: 模型工厂函数
            device_config: 设备配置
        """
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.model_factory = model_factory
        self.device_config = device_config
        
        self.model = None
        self.is_initialized = False
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
        
        # 性能统计
        self.total_tasks = 0
        self.total_time = 0.0
        self.error_count = 0
        
    def initialize(self):
        """初始化GPU工作器"""
        try:
            torch.cuda.set_device(self.gpu_id)
            
            # 创建模型
            self.model = self.model_factory()
            self.model = self.model.to(self.device)
            
            # 设置模型为评估模式
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # 启用混合精度
            if self.device_config.get('mixed_precision', False):
                self.model = self.model.half()
            
            self.is_initialized = True
            logger.info(f"GPU {self.gpu_id} worker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU {self.gpu_id} worker: {e}")
            raise
    
    def start(self):
        """启动工作器线程"""
        if not self.is_initialized:
            self.initialize()
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info(f"GPU {self.gpu_id} worker started")
    
    def stop(self):
        """停止工作器"""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        logger.info(f"GPU {self.gpu_id} worker stopped")
    
    def submit_task(self, task: GPUTask):
        """提交任务
        
        Args:
            task: GPU任务
        """
        if not self.is_running:
            raise RuntimeError(f"GPU {self.gpu_id} worker is not running")
        
        self.task_queue.put(task)
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[GPUResult]:
        """获取处理结果
        
        Args:
            timeout: 超时时间
            
        Returns:
            处理结果
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _worker_loop(self):
        """工作器主循环"""
        torch.cuda.set_device(self.gpu_id)
        
        while self.is_running:
            try:
                # 获取任务
                task = self.task_queue.get(timeout=1.0)
                
                # 处理任务
                start_time = time.time()
                result = self._process_task(task)
                processing_time = time.time() - start_time
                
                # 创建结果
                gpu_result = GPUResult(
                    task_id=task.task_id,
                    result=result,
                    gpu_id=self.gpu_id,
                    processing_time=processing_time,
                    success=True
                )
                
                # 更新统计
                self.total_tasks += 1
                self.total_time += processing_time
                
                # 返回结果
                self.result_queue.put(gpu_result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"GPU {self.gpu_id} worker error: {e}")
                
                # 创建错误结果
                if 'task' in locals():
                    error_result = GPUResult(
                        task_id=task.task_id,
                        result=None,
                        gpu_id=self.gpu_id,
                        processing_time=0.0,
                        success=False,
                        error=str(e)
                    )
                    self.result_queue.put(error_result)
                
                self.error_count += 1
    
    def _process_task(self, task: GPUTask) -> Any:
        """处理单个任务
        
        Args:
            task: GPU任务
            
        Returns:
            处理结果
        """
        with torch.no_grad():
            # 将数据移动到GPU
            if isinstance(task.data, torch.Tensor):
                data = task.data.to(self.device, non_blocking=True)
            elif isinstance(task.data, (list, tuple)):
                data = [d.to(self.device, non_blocking=True) if isinstance(d, torch.Tensor) else d for d in task.data]
            elif isinstance(task.data, dict):
                data = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in task.data.items()}
            else:
                data = task.data
            
            # 执行模型推理
            if hasattr(self.model, '__call__'):
                if isinstance(data, (list, tuple)):
                    result = self.model(*data)
                elif isinstance(data, dict):
                    result = self.model(**data)
                else:
                    result = self.model(data)
            else:
                raise ValueError(f"Model on GPU {self.gpu_id} is not callable")
            
            # 将结果移动到CPU
            if isinstance(result, torch.Tensor):
                result = result.cpu()
            elif isinstance(result, (list, tuple)):
                result = [r.cpu() if isinstance(r, torch.Tensor) else r for r in result]
            elif isinstance(result, dict):
                result = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取工作器统计信息
        
        Returns:
            统计信息
        """
        avg_time = self.total_time / max(self.total_tasks, 1)
        
        return {
            'gpu_id': self.gpu_id,
            'total_tasks': self.total_tasks,
            'total_time': self.total_time,
            'avg_time': avg_time,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.total_tasks, 1),
            'queue_size': self.task_queue.qsize(),
            'is_running': self.is_running
        }


class MultiGPUProcessor:
    """多GPU处理器
    
    管理多个GPU工作器，提供负载均衡和错误恢复功能。
    """
    
    def __init__(self, gpu_ids: List[int], model_factory: Callable, device_config: Dict[str, Any]):
        """初始化多GPU处理器
        
        Args:
            gpu_ids: GPU ID列表
            model_factory: 模型工厂函数
            device_config: 设备配置
        """
        self.gpu_ids = gpu_ids
        self.model_factory = model_factory
        self.device_config = device_config
        
        # 创建GPU工作器
        self.workers: Dict[int, GPUWorker] = {}
        for gpu_id in gpu_ids:
            self.workers[gpu_id] = GPUWorker(gpu_id, model_factory, device_config)
        
        # 负载均衡
        self.load_balancer = self.device_config.get('load_balancer', 'round_robin')
        self.current_gpu_index = 0
        
        # 任务管理
        self.pending_tasks: Dict[str, GPUTask] = {}
        self.completed_results: Dict[str, GPUResult] = {}
        
        # 统计信息
        self.total_submitted = 0
        self.total_completed = 0
        
        self.is_running = False
    
    def start(self):
        """启动所有GPU工作器"""
        logger.info(f"Starting MultiGPU processor with {len(self.gpu_ids)} GPUs")
        
        for worker in self.workers.values():
            worker.start()
        
        self.is_running = True
        logger.info("MultiGPU processor started successfully")
    
    def stop(self):
        """停止所有GPU工作器"""
        logger.info("Stopping MultiGPU processor")
        
        for worker in self.workers.values():
            worker.stop()
        
        self.is_running = False
        logger.info("MultiGPU processor stopped")
    
    def submit_batch(self, data_batch: List[Any], task_ids: Optional[List[str]] = None) -> List[str]:
        """提交批量任务
        
        Args:
            data_batch: 数据批次
            task_ids: 任务ID列表
            
        Returns:
            任务ID列表
        """
        if not self.is_running:
            raise RuntimeError("MultiGPU processor is not running")
        
        if task_ids is None:
            task_ids = [f"task_{self.total_submitted + i}" for i in range(len(data_batch))]
        
        if len(task_ids) != len(data_batch):
            raise ValueError("task_ids and data_batch must have the same length")
        
        submitted_ids = []
        
        for i, (data, task_id) in enumerate(zip(data_batch, task_ids)):
            # 选择GPU
            gpu_id = self._select_gpu()
            
            # 创建任务
            task = GPUTask(
                task_id=task_id,
                data=data,
                gpu_id=gpu_id
            )
            
            # 提交任务
            self.workers[gpu_id].submit_task(task)
            self.pending_tasks[task_id] = task
            
            submitted_ids.append(task_id)
            self.total_submitted += 1
        
        return submitted_ids
    
    def get_results(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, GPUResult]:
        """获取任务结果
        
        Args:
            task_ids: 任务ID列表
            timeout: 超时时间
            
        Returns:
            任务结果字典
        """
        results = {}
        start_time = time.time()
        
        while len(results) < len(task_ids):
            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # 从所有工作器收集结果
            for worker in self.workers.values():
                result = worker.get_result(timeout=0.1)
                if result and result.task_id in task_ids:
                    results[result.task_id] = result
                    self.completed_results[result.task_id] = result
                    
                    # 从待处理任务中移除
                    if result.task_id in self.pending_tasks:
                        del self.pending_tasks[result.task_id]
                    
                    self.total_completed += 1
        
        return results
    
    def process_batch(self, data_batch: List[Any], timeout: Optional[float] = None) -> List[Any]:
        """处理批量数据（同步）
        
        Args:
            data_batch: 数据批次
            timeout: 超时时间
            
        Returns:
            处理结果列表
        """
        # 提交任务
        task_ids = self.submit_batch(data_batch)
        
        # 获取结果
        results = self.get_results(task_ids, timeout=timeout)
        
        # 按原始顺序排列结果
        ordered_results = []
        for task_id in task_ids:
            if task_id in results:
                result = results[task_id]
                if result.success:
                    ordered_results.append(result.result)
                else:
                    raise RuntimeError(f"Task {task_id} failed: {result.error}")
            else:
                raise TimeoutError(f"Task {task_id} timed out")
        
        return ordered_results
    
    def _select_gpu(self) -> int:
        """选择GPU进行负载均衡
        
        Returns:
            选中的GPU ID
        """
        if self.load_balancer == 'round_robin':
            gpu_id = self.gpu_ids[self.current_gpu_index]
            self.current_gpu_index = (self.current_gpu_index + 1) % len(self.gpu_ids)
            return gpu_id
        
        elif self.load_balancer == 'least_busy':
            # 选择队列最短的GPU
            min_queue_size = float('inf')
            selected_gpu = self.gpu_ids[0]
            
            for gpu_id in self.gpu_ids:
                queue_size = self.workers[gpu_id].task_queue.qsize()
                if queue_size < min_queue_size:
                    min_queue_size = queue_size
                    selected_gpu = gpu_id
            
            return selected_gpu
        
        elif self.load_balancer == 'random':
            return np.random.choice(self.gpu_ids)
        
        else:
            # 默认轮询
            return self.gpu_ids[self.current_gpu_index % len(self.gpu_ids)]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息
        
        Returns:
            统计信息
        """
        worker_stats = {gpu_id: worker.get_stats() for gpu_id, worker in self.workers.items()}
        
        total_tasks = sum(stats['total_tasks'] for stats in worker_stats.values())
        total_time = sum(stats['total_time'] for stats in worker_stats.values())
        total_errors = sum(stats['error_count'] for stats in worker_stats.values())
        
        return {
            'total_submitted': self.total_submitted,
            'total_completed': self.total_completed,
            'pending_tasks': len(self.pending_tasks),
            'total_tasks': total_tasks,
            'total_time': total_time,
            'total_errors': total_errors,
            'avg_time': total_time / max(total_tasks, 1),
            'error_rate': total_errors / max(total_tasks, 1),
            'worker_stats': worker_stats,
            'is_running': self.is_running
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        
        print("=== MultiGPU Processor Stats ===")
        print(f"Total submitted: {stats['total_submitted']}")
        print(f"Total completed: {stats['total_completed']}")
        print(f"Pending tasks: {stats['pending_tasks']}")
        print(f"Average processing time: {stats['avg_time']:.3f}s")
        print(f"Error rate: {stats['error_rate']:.2%}")
        
        print("\nWorker Stats:")
        for gpu_id, worker_stats in stats['worker_stats'].items():
            print(f"  GPU {gpu_id}: {worker_stats['total_tasks']} tasks, "
                  f"{worker_stats['avg_time']:.3f}s avg, "
                  f"{worker_stats['error_rate']:.2%} error rate, "
                  f"queue: {worker_stats['queue_size']}")
        
        print("==============================")


class DistributedProcessor:
    """分布式处理器
    
    支持多节点分布式处理。
    """
    
    def __init__(self, rank: int, world_size: int, backend: str = 'nccl'):
        """初始化分布式处理器
        
        Args:
            rank: 当前进程排名
            world_size: 总进程数
            backend: 分布式后端
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.is_initialized = False
    
    def initialize(self, master_addr: str = 'localhost', master_port: str = '12355'):
        """初始化分布式环境
        
        Args:
            master_addr: 主节点地址
            master_port: 主节点端口
        """
        import os
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        dist.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size
        )
        
        self.is_initialized = True
        logger.info(f"Distributed processor initialized: rank {self.rank}/{self.world_size}")
    
    def wrap_model(self, model: nn.Module, device_ids: Optional[List[int]] = None) -> DDP:
        """包装模型为分布式模型
        
        Args:
            model: 原始模型
            device_ids: 设备ID列表
            
        Returns:
            分布式模型
        """
        if not self.is_initialized:
            raise RuntimeError("Distributed processor not initialized")
        
        if device_ids is None:
            device_ids = [self.rank]
        
        ddp_model = DDP(model, device_ids=device_ids)
        return ddp_model
    
    def create_dataloader(self, dataset, batch_size: int, **kwargs) -> DataLoader:
        """创建分布式数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批处理大小
            **kwargs: 其他参数
            
        Returns:
            数据加载器
        """
        if not self.is_initialized:
            raise RuntimeError("Distributed processor not initialized")
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=kwargs.pop('shuffle', True)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            **kwargs
        )
        
        return dataloader
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """收集所有进程的张量
        
        Args:
            tensor: 输入张量
            
        Returns:
            所有进程的张量列表
        """
        if not self.is_initialized:
            raise RuntimeError("Distributed processor not initialized")
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """归约所有进程的张量
        
        Args:
            tensor: 输入张量
            op: 归约操作
            
        Returns:
            归约后的张量
        """
        if not self.is_initialized:
            raise RuntimeError("Distributed processor not initialized")
        
        dist.all_reduce(tensor, op=op)
        return tensor
    
    def cleanup(self):
        """清理分布式环境"""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("Distributed processor cleaned up")


@dataclass
class MultiGPUConfig:
    """多GPU配置数据类
    
    存储多GPU处理的配置信息。
    """
    # GPU设备配置
    gpu_ids: List[int] = None
    device_map: Dict[str, int] = None
    
    # 并行处理配置
    batch_size_per_gpu: int = 8
    num_workers_per_gpu: int = 2
    max_concurrent_tasks: int = 16
    
    # 负载均衡配置
    enable_load_balancing: bool = True
    memory_threshold: float = 0.8
    task_timeout: float = 300.0
    
    # 分布式配置
    backend: str = 'nccl'
    init_method: str = 'env://'
    world_size: int = 1
    rank: int = 0
    
    # 性能优化配置
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    pin_memory: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        if self.gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        if self.device_map is None:
            self.device_map = {}
    
    @classmethod
    def from_hardware_config(cls, hardware_config) -> 'MultiGPUConfig':
        """从硬件配置创建多GPU配置
        
        Args:
            hardware_config: 硬件配置对象
            
        Returns:
            MultiGPUConfig: 多GPU配置实例
        """
        gpu_count = getattr(hardware_config, 'gpu_count', 0)
        total_memory = getattr(hardware_config, 'total_gpu_memory', 0)
        
        # 根据GPU数量和内存调整配置
        batch_size_per_gpu = 8 if total_memory > 20000 else 4
        max_concurrent_tasks = min(gpu_count * 4, 32)
        
        return cls(
            gpu_ids=list(range(gpu_count)) if gpu_count > 0 else [],
            batch_size_per_gpu=batch_size_per_gpu,
            max_concurrent_tasks=max_concurrent_tasks,
            world_size=gpu_count,
            use_mixed_precision=total_memory > 10000
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            'gpu_ids': self.gpu_ids,
            'device_map': self.device_map,
            'batch_size_per_gpu': self.batch_size_per_gpu,
            'num_workers_per_gpu': self.num_workers_per_gpu,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'enable_load_balancing': self.enable_load_balancing,
            'memory_threshold': self.memory_threshold,
            'task_timeout': self.task_timeout,
            'backend': self.backend,
            'init_method': self.init_method,
            'world_size': self.world_size,
            'rank': self.rank,
            'use_mixed_precision': self.use_mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'pin_memory': self.pin_memory
        }
    
    def validate(self) -> bool:
        """验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        if not self.gpu_ids:
            logger.warning("No GPU IDs specified")
            return False
        
        if self.batch_size_per_gpu <= 0:
            logger.error("Batch size per GPU must be positive")
            return False
        
        if self.memory_threshold <= 0 or self.memory_threshold > 1:
            logger.error("Memory threshold must be between 0 and 1")
            return False
        
        return True