"""CUDA工具模块

提供CUDA相关的工具函数，包括显存监控、异常处理和设备管理。
"""

import torch
import torch.cuda
import psutil
import logging
from typing import Dict, List, Optional, Tuple, Any
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryInfo:
    """GPU内存信息"""
    gpu_id: int
    total_memory: int  # 总内存（字节）
    allocated_memory: int  # 已分配内存（字节）
    reserved_memory: int  # 保留内存（字节）
    free_memory: int  # 可用内存（字节）
    utilization: float  # 利用率（百分比）
    
    @property
    def total_mb(self) -> float:
        """总内存（MB）"""
        return self.total_memory / 1024**2
    
    @property
    def allocated_mb(self) -> float:
        """已分配内存（MB）"""
        return self.allocated_memory / 1024**2
    
    @property
    def reserved_mb(self) -> float:
        """保留内存（MB）"""
        return self.reserved_memory / 1024**2
    
    @property
    def free_mb(self) -> float:
        """可用内存（MB）"""
        return self.free_memory / 1024**2


class CUDAErrorHandler:
    """CUDA错误处理器
    
    提供CUDA操作的安全包装和错误恢复机制。
    """
    
    @staticmethod
    def safe_empty_cache(gpu_id: Optional[int] = None) -> bool:
        """安全清空CUDA缓存
        
        Args:
            gpu_id: GPU ID，None表示当前设备
            
        Returns:
            是否成功清空缓存
        """
        try:
            if gpu_id is not None:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            
            logger.debug(f"Successfully cleared CUDA cache for GPU {gpu_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache for GPU {gpu_id}: {e}")
            return False
    
    @staticmethod
    def safe_synchronize(gpu_id: Optional[int] = None) -> bool:
        """安全同步CUDA设备
        
        Args:
            gpu_id: GPU ID，None表示当前设备
            
        Returns:
            是否成功同步
        """
        try:
            if gpu_id is not None:
                with torch.cuda.device(gpu_id):
                    torch.cuda.synchronize()
            else:
                torch.cuda.synchronize()
            
            logger.debug(f"Successfully synchronized GPU {gpu_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to synchronize GPU {gpu_id}: {e}")
            return False
    
    @staticmethod
    def handle_cuda_oom(func, *args, reduce_factor: float = 0.8, max_retries: int = 3, **kwargs):
        """处理CUDA内存不足错误
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            reduce_factor: 批处理大小缩减因子
            max_retries: 最大重试次数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt < max_retries:
                    logger.warning(f"CUDA OOM detected (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    
                    # 清空缓存
                    CUDAErrorHandler.safe_empty_cache()
                    
                    # 如果有batch_size参数，尝试减少批处理大小
                    if 'batch_size' in kwargs:
                        old_batch_size = kwargs['batch_size']
                        new_batch_size = max(1, int(old_batch_size * reduce_factor))
                        kwargs['batch_size'] = new_batch_size
                        logger.info(f"Reducing batch size from {old_batch_size} to {new_batch_size}")
                    
                    # 等待一段时间再重试
                    time.sleep(1.0)
                    continue
                else:
                    raise
        
        raise RuntimeError(f"Failed to execute function after {max_retries} retries")
    
    @staticmethod
    @contextmanager
    def cuda_error_context(gpu_id: Optional[int] = None):
        """CUDA错误上下文管理器
        
        Args:
            gpu_id: GPU ID
        """
        try:
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
            yield
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA OOM on GPU {gpu_id}: {e}")
                CUDAErrorHandler.safe_empty_cache(gpu_id)
            raise
        except Exception as e:
            logger.error(f"CUDA error on GPU {gpu_id}: {e}")
            raise


class GPUMonitor:
    """GPU监控器
    
    实时监控GPU使用情况和性能指标。
    """
    
    def __init__(self, gpu_ids: Optional[List[int]] = None, monitor_interval: float = 1.0):
        """初始化GPU监控器
        
        Args:
            gpu_ids: 要监控的GPU ID列表
            monitor_interval: 监控间隔（秒）
        """
        self.gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))
        self.monitor_interval = monitor_interval
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.memory_history: Dict[int, List[GPUMemoryInfo]] = {gpu_id: [] for gpu_id in self.gpu_ids}
        self.max_history_length = 1000
        
        # 监控锁
        self._lock = threading.Lock()
    
    def get_gpu_memory_info(self, gpu_id: int) -> GPUMemoryInfo:
        """获取GPU内存信息
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            GPU内存信息
        """
        try:
            torch.cuda.set_device(gpu_id)
            
            # 获取内存信息
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            reserved_memory = torch.cuda.memory_reserved(gpu_id)
            free_memory = total_memory - reserved_memory
            utilization = (allocated_memory / total_memory) * 100
            
            return GPUMemoryInfo(
                gpu_id=gpu_id,
                total_memory=total_memory,
                allocated_memory=allocated_memory,
                reserved_memory=reserved_memory,
                free_memory=free_memory,
                utilization=utilization
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory info for GPU {gpu_id}: {e}")
            raise
    
    def get_all_gpu_memory_info(self) -> Dict[int, GPUMemoryInfo]:
        """获取所有GPU的内存信息
        
        Returns:
            GPU ID到内存信息的映射
        """
        memory_info = {}
        
        for gpu_id in self.gpu_ids:
            try:
                memory_info[gpu_id] = self.get_gpu_memory_info(gpu_id)
            except Exception as e:
                logger.warning(f"Failed to get memory info for GPU {gpu_id}: {e}")
        
        return memory_info
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("GPU monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Started GPU monitoring for GPUs: {self.gpu_ids}")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped GPU monitoring")
    
    def _monitor_loop(self):
        """监控主循环"""
        while self.is_monitoring:
            try:
                # 获取当前内存信息
                current_info = self.get_all_gpu_memory_info()
                
                # 更新历史记录
                with self._lock:
                    for gpu_id, info in current_info.items():
                        if gpu_id in self.memory_history:
                            self.memory_history[gpu_id].append(info)
                            
                            # 限制历史记录长度
                            if len(self.memory_history[gpu_id]) > self.max_history_length:
                                self.memory_history[gpu_id].pop(0)
                
                # 等待下一次监控
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def get_memory_history(self, gpu_id: int, last_n: Optional[int] = None) -> List[GPUMemoryInfo]:
        """获取GPU内存历史记录
        
        Args:
            gpu_id: GPU ID
            last_n: 获取最近N条记录
            
        Returns:
            内存历史记录
        """
        with self._lock:
            history = self.memory_history.get(gpu_id, [])
            if last_n is not None:
                history = history[-last_n:]
            return history.copy()
    
    def get_peak_memory_usage(self, gpu_id: int) -> Optional[GPUMemoryInfo]:
        """获取GPU峰值内存使用
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            峰值内存信息
        """
        history = self.get_memory_history(gpu_id)
        if not history:
            return None
        
        return max(history, key=lambda x: x.allocated_memory)
    
    def print_current_stats(self):
        """打印当前GPU统计信息"""
        current_info = self.get_all_gpu_memory_info()
        
        print("=== GPU Memory Stats ===")
        for gpu_id, info in current_info.items():
            print(f"GPU {gpu_id}:")
            print(f"  Total: {info.total_mb:.1f} MB")
            print(f"  Allocated: {info.allocated_mb:.1f} MB ({info.utilization:.1f}%)")
            print(f"  Reserved: {info.reserved_mb:.1f} MB")
            print(f"  Free: {info.free_mb:.1f} MB")
        print("========================")
    
    def print_peak_stats(self):
        """打印峰值GPU统计信息"""
        print("=== GPU Peak Memory Stats ===")
        for gpu_id in self.gpu_ids:
            peak_info = self.get_peak_memory_usage(gpu_id)
            if peak_info:
                print(f"GPU {gpu_id} Peak:")
                print(f"  Allocated: {peak_info.allocated_mb:.1f} MB ({peak_info.utilization:.1f}%)")
                print(f"  Reserved: {peak_info.reserved_mb:.1f} MB")
            else:
                print(f"GPU {gpu_id}: No data available")
        print("=============================")


class CUDADeviceManager:
    """CUDA设备管理器
    
    管理CUDA设备的分配和使用。
    """
    
    def __init__(self):
        """初始化设备管理器"""
        self.available_devices = list(range(torch.cuda.device_count()))
        self.device_locks = {device_id: threading.Lock() for device_id in self.available_devices}
        self.device_usage = {device_id: 0 for device_id in self.available_devices}
        
        # 设备能力信息
        self.device_capabilities = {}
        for device_id in self.available_devices:
            props = torch.cuda.get_device_properties(device_id)
            self.device_capabilities[device_id] = {
                'name': props.name,
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count
            }
    
    def get_optimal_device(self, memory_required: Optional[int] = None) -> int:
        """获取最优设备
        
        Args:
            memory_required: 所需内存（字节）
            
        Returns:
            最优设备ID
        """
        if not self.available_devices:
            raise RuntimeError("No CUDA devices available")
        
        # 如果指定了内存需求，过滤设备
        candidate_devices = self.available_devices
        if memory_required is not None:
            candidate_devices = [
                device_id for device_id in self.available_devices
                if self.device_capabilities[device_id]['total_memory'] >= memory_required
            ]
            
            if not candidate_devices:
                raise RuntimeError(f"No device has enough memory ({memory_required} bytes required)")
        
        # 选择使用率最低的设备
        optimal_device = min(candidate_devices, key=lambda x: self.device_usage[x])
        return optimal_device
    
    @contextmanager
    def acquire_device(self, device_id: Optional[int] = None, memory_required: Optional[int] = None):
        """获取设备上下文管理器
        
        Args:
            device_id: 指定设备ID，None表示自动选择
            memory_required: 所需内存（字节）
        """
        if device_id is None:
            device_id = self.get_optimal_device(memory_required)
        
        if device_id not in self.available_devices:
            raise ValueError(f"Device {device_id} is not available")
        
        # 获取设备锁
        with self.device_locks[device_id]:
            try:
                # 设置当前设备
                old_device = torch.cuda.current_device() if torch.cuda.is_available() else None
                torch.cuda.set_device(device_id)
                
                # 更新使用计数
                self.device_usage[device_id] += 1
                
                logger.debug(f"Acquired CUDA device {device_id}")
                yield device_id
                
            finally:
                # 恢复原设备
                if old_device is not None:
                    torch.cuda.set_device(old_device)
                
                # 更新使用计数
                self.device_usage[device_id] -= 1
                
                logger.debug(f"Released CUDA device {device_id}")
    
    def get_device_info(self, device_id: int) -> Dict[str, Any]:
        """获取设备信息
        
        Args:
            device_id: 设备ID
            
        Returns:
            设备信息
        """
        if device_id not in self.available_devices:
            raise ValueError(f"Device {device_id} is not available")
        
        capability = self.device_capabilities[device_id]
        
        # 获取当前内存信息
        try:
            monitor = GPUMonitor([device_id])
            memory_info = monitor.get_gpu_memory_info(device_id)
            
            return {
                'device_id': device_id,
                'name': capability['name'],
                'compute_capability': f"{capability['major']}.{capability['minor']}",
                'total_memory_mb': capability['total_memory'] / 1024**2,
                'multi_processor_count': capability['multi_processor_count'],
                'current_usage': self.device_usage[device_id],
                'memory_utilization': memory_info.utilization,
                'allocated_memory_mb': memory_info.allocated_mb,
                'free_memory_mb': memory_info.free_mb
            }
            
        except Exception as e:
            logger.warning(f"Failed to get current memory info for device {device_id}: {e}")
            return {
                'device_id': device_id,
                'name': capability['name'],
                'compute_capability': f"{capability['major']}.{capability['minor']}",
                'total_memory_mb': capability['total_memory'] / 1024**2,
                'multi_processor_count': capability['multi_processor_count'],
                'current_usage': self.device_usage[device_id]
            }
    
    def print_device_info(self):
        """打印所有设备信息"""
        print("=== CUDA Device Information ===")
        for device_id in self.available_devices:
            info = self.get_device_info(device_id)
            print(f"Device {device_id}: {info['name']}")
            print(f"  Compute Capability: {info['compute_capability']}")
            print(f"  Total Memory: {info['total_memory_mb']:.1f} MB")
            print(f"  Multi-processors: {info['multi_processor_count']}")
            print(f"  Current Usage: {info['current_usage']}")
            if 'memory_utilization' in info:
                print(f"  Memory Utilization: {info['memory_utilization']:.1f}%")
                print(f"  Free Memory: {info['free_memory_mb']:.1f} MB")
        print("===============================")


# 全局设备管理器实例
_global_device_manager = None
_device_manager_lock = threading.Lock()


def get_device_manager() -> CUDADeviceManager:
    """获取全局设备管理器实例
    
    Returns:
        设备管理器实例
    """
    global _global_device_manager
    
    if _global_device_manager is None:
        with _device_manager_lock:
            if _global_device_manager is None:
                _global_device_manager = CUDADeviceManager()
    
    return _global_device_manager


def check_cuda_availability() -> bool:
    """检查CUDA可用性
    
    Returns:
        CUDA是否可用
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available")
        return False
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        logger.warning("No CUDA devices found")
        return False
    
    logger.info(f"CUDA is available with {device_count} device(s)")
    return True


def estimate_memory_usage(model: torch.nn.Module, input_shape: Tuple[int, ...], 
                         dtype: torch.dtype = torch.float32) -> int:
    """估算模型内存使用量
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状
        dtype: 数据类型
        
    Returns:
        估算的内存使用量（字节）
    """
    # 计算模型参数内存
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 计算输入内存
    input_memory = torch.tensor(input_shape).prod().item() * torch.tensor([], dtype=dtype).element_size()
    
    # 估算激活内存（简化估算，实际可能更复杂）
    activation_memory = input_memory * 4  # 粗略估算
    
    total_memory = param_memory + input_memory + activation_memory
    
    logger.debug(f"Estimated memory usage: {total_memory / 1024**2:.1f} MB")
    logger.debug(f"  Parameters: {param_memory / 1024**2:.1f} MB")
    logger.debug(f"  Input: {input_memory / 1024**2:.1f} MB")
    logger.debug(f"  Activations: {activation_memory / 1024**2:.1f} MB")
    
    return total_memory


def optimize_batch_size(model: torch.nn.Module, input_shape: Tuple[int, ...], 
                       max_memory_mb: float = 8000, dtype: torch.dtype = torch.float32) -> int:
    """优化批处理大小
    
    Args:
        model: PyTorch模型
        input_shape: 单个样本的输入形状
        max_memory_mb: 最大内存限制（MB）
        dtype: 数据类型
        
    Returns:
        推荐的批处理大小
    """
    max_memory_bytes = max_memory_mb * 1024**2
    
    # 估算单个样本的内存使用
    single_sample_memory = estimate_memory_usage(model, input_shape, dtype)
    
    # 计算最大批处理大小
    max_batch_size = max(1, int(max_memory_bytes / single_sample_memory))
    
    # 选择2的幂次作为批处理大小（通常更高效）
    optimal_batch_size = 1
    while optimal_batch_size * 2 <= max_batch_size:
        optimal_batch_size *= 2
    
    logger.info(f"Recommended batch size: {optimal_batch_size} (max possible: {max_batch_size})")
    
    return optimal_batch_size


def check_cuda_available() -> bool:
    """检查CUDA是否可用
    
    Returns:
        bool: CUDA是否可用
    """
    return torch.cuda.is_available()


def get_cuda_device_count() -> int:
    """获取CUDA设备数量
    
    Returns:
        int: CUDA设备数量
    """
    if not check_cuda_available():
        return 0
    return torch.cuda.device_count()


def get_cuda_device_name(device_id: int = 0) -> str:
    """获取CUDA设备名称
    
    Args:
        device_id: 设备ID
        
    Returns:
        str: 设备名称
    """
    if not check_cuda_available():
        return "CPU"
    
    try:
        return torch.cuda.get_device_name(device_id)
    except Exception:
        return f"CUDA Device {device_id}"


def get_cuda_memory_info(device_id: int = 0) -> Dict[str, int]:
    """获取CUDA内存信息
    
    Args:
        device_id: 设备ID
        
    Returns:
        Dict[str, int]: 内存信息字典
    """
    if not check_cuda_available():
        return {'total': 0, 'allocated': 0, 'reserved': 0, 'free': 0}
    
    try:
        total = torch.cuda.get_device_properties(device_id).total_memory
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        free = total - reserved
        
        return {
            'total': total,
            'allocated': allocated,
            'reserved': reserved,
            'free': free
        }
    except Exception as e:
        logger.warning(f"Failed to get CUDA memory info for device {device_id}: {e}")
        return {'total': 0, 'allocated': 0, 'reserved': 0, 'free': 0}


def clear_cuda_cache():
    """清空CUDA缓存"""
    if check_cuda_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")


def set_cuda_device(device_id: int):
    """设置CUDA设备
    
    Args:
        device_id: 设备ID
    """
    if check_cuda_available() and device_id < get_cuda_device_count():
        torch.cuda.set_device(device_id)
        logger.info(f"CUDA device set to {device_id}")
    else:
        logger.warning(f"Cannot set CUDA device {device_id}")


# 全局设备管理器实例
_global_device_manager = None


def get_device_manager() -> CUDADeviceManager:
    """获取全局设备管理器实例
    
    Returns:
        CUDADeviceManager: 设备管理器实例
    """
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = CUDADeviceManager()
    return _global_device_manager


def estimate_model_memory(model_name: str, precision: str = "fp32") -> int:
    """估算模型内存使用量
    
    Args:
        model_name: 模型名称
        precision: 精度类型 ("fp32", "fp16", "int8")
        
    Returns:
        int: 估算的内存使用量（字节）
    """
    # 常见模型的参数量（百万）
    model_params = {
        "clip-vit-base-patch32": 151,
        "clip-vit-large-patch14": 428,
        "stable-diffusion-v1-5": 860,
        "stable-diffusion-v2-1": 865,
        "qwen2-7b": 7000,
        "qwen2-14b": 14000,
    }
    
    # 根据模型名称匹配参数量
    params_millions = 100  # 默认值
    for key, value in model_params.items():
        if key.lower() in model_name.lower():
            params_millions = value
            break
    
    # 根据精度计算每个参数的字节数
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1
    }
    
    param_bytes = bytes_per_param.get(precision, 4)
    
    # 计算基础内存（参数 + 梯度 + 优化器状态）
    base_memory = params_millions * 1_000_000 * param_bytes
    
    # 添加额外开销（激活值、临时缓冲区等）
    overhead_factor = 2.0
    total_memory = int(base_memory * overhead_factor)
    
    logger.info(f"Estimated memory for {model_name} ({precision}): {total_memory / 1024**3:.2f} GB")
    
    return total_memory