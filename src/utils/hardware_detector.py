"""硬件检测器模块

自动检测系统硬件配置，包括GPU、内存、CPU等信息。
用于自动配置模型和优化性能设置。
"""

import os
import platform
import psutil
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = None

logger = logging.getLogger(__name__)


class HardwareDetector:
    """硬件检测器
    
    检测系统硬件配置并提供优化建议。
    """
    
    def __init__(self):
        """初始化硬件检测器"""
        self._gpu_info: Optional[List[Dict[str, Any]]] = None
        self._cpu_info: Optional[Dict[str, Any]] = None
        self._memory_info: Optional[Dict[str, Any]] = None
        self._system_info: Optional[Dict[str, Any]] = None
    
    def has_cuda(self) -> bool:
        """检查是否支持CUDA
        
        Returns:
            是否支持CUDA
        """
        if not TORCH_AVAILABLE:
            return False
        
        try:
            return torch.cuda.is_available()
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {e}")
            return False
    
    def get_cuda_version(self) -> Optional[str]:
        """获取CUDA版本
        
        Returns:
            CUDA版本字符串
        """
        if not self.has_cuda():
            return None
        
        try:
            return torch.version.cuda
        except Exception as e:
            logger.warning(f"Error getting CUDA version: {e}")
            return None
    
    def get_gpu_count(self) -> int:
        """获取GPU数量
        
        Returns:
            GPU数量
        """
        if not self.has_cuda():
            return 0
        
        try:
            return torch.cuda.device_count()
        except Exception as e:
            logger.warning(f"Error getting GPU count: {e}")
            return 0
    
    def get_available_gpus(self) -> List[int]:
        """获取可用的GPU ID列表
        
        Returns:
            可用GPU ID列表
        """
        gpu_count = self.get_gpu_count()
        if gpu_count == 0:
            return []
        
        available_gpus = []
        
        for gpu_id in range(gpu_count):
            try:
                # 检查GPU是否可用
                if self._is_gpu_available(gpu_id):
                    available_gpus.append(gpu_id)
            except Exception as e:
                logger.warning(f"Error checking GPU {gpu_id}: {e}")
        
        return available_gpus
    
    def _is_gpu_available(self, gpu_id: int) -> bool:
        """检查指定GPU是否可用
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            是否可用
        """
        if not TORCH_AVAILABLE:
            return False
        
        try:
            # 尝试在GPU上创建一个小张量
            device = torch.device(f'cuda:{gpu_id}')
            test_tensor = torch.zeros(1, device=device)
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception:
            return False
    
    def get_gpu_info(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """获取GPU信息
        
        Args:
            refresh: 是否刷新缓存
            
        Returns:
            GPU信息列表
        """
        if self._gpu_info is not None and not refresh:
            return self._gpu_info
        
        gpu_info = []
        
        if self.has_cuda():
            for gpu_id in range(self.get_gpu_count()):
                try:
                    info = self._get_single_gpu_info(gpu_id)
                    gpu_info.append(info)
                except Exception as e:
                    logger.warning(f"Error getting info for GPU {gpu_id}: {e}")
        
        self._gpu_info = gpu_info
        return gpu_info
    
    def _get_single_gpu_info(self, gpu_id: int) -> Dict[str, Any]:
        """获取单个GPU信息
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            GPU信息字典
        """
        if not TORCH_AVAILABLE:
            return {}
        
        info = {
            'id': gpu_id,
            'name': 'Unknown',
            'memory_total': 0,
            'memory_free': 0,
            'memory_used': 0,
            'utilization': 0,
            'temperature': 0,
            'available': False
        }
        
        try:
            # 使用PyTorch获取基本信息
            props = torch.cuda.get_device_properties(gpu_id)
            info['name'] = props.name
            info['memory_total'] = props.total_memory
            
            # 获取内存使用情况
            torch.cuda.set_device(gpu_id)
            memory_allocated = torch.cuda.memory_allocated(gpu_id)
            memory_reserved = torch.cuda.memory_reserved(gpu_id)
            
            info['memory_used'] = memory_allocated
            info['memory_free'] = info['memory_total'] - memory_reserved
            info['available'] = self._is_gpu_available(gpu_id)
            
            # 尝试使用GPUtil获取更详细信息
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpu_id < len(gpus):
                        gpu = gpus[gpu_id]
                        info['utilization'] = gpu.load * 100
                        info['temperature'] = gpu.temperature
                        info['memory_total'] = gpu.memoryTotal * 1024 * 1024  # MB to bytes
                        info['memory_used'] = gpu.memoryUsed * 1024 * 1024
                        info['memory_free'] = gpu.memoryFree * 1024 * 1024
                except Exception as e:
                    logger.debug(f"GPUtil error for GPU {gpu_id}: {e}")
            
        except Exception as e:
            logger.warning(f"Error getting detailed info for GPU {gpu_id}: {e}")
        
        return info
    
    def get_cpu_info(self, refresh: bool = False) -> Dict[str, Any]:
        """获取CPU信息
        
        Args:
            refresh: 是否刷新缓存
            
        Returns:
            CPU信息字典
        """
        if self._cpu_info is not None and not refresh:
            return self._cpu_info
        
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency': 0,
            'current_frequency': 0,
            'usage_percent': 0,
            'architecture': platform.machine(),
            'processor': platform.processor()
        }
        
        try:
            # CPU频率信息
            freq_info = psutil.cpu_freq()
            if freq_info:
                cpu_info['max_frequency'] = freq_info.max
                cpu_info['current_frequency'] = freq_info.current
            
            # CPU使用率
            cpu_info['usage_percent'] = psutil.cpu_percent(interval=1)
            
        except Exception as e:
            logger.warning(f"Error getting CPU info: {e}")
        
        self._cpu_info = cpu_info
        return cpu_info
    
    def get_memory_info(self, refresh: bool = False) -> Dict[str, Any]:
        """获取内存信息
        
        Args:
            refresh: 是否刷新缓存
            
        Returns:
            内存信息字典
        """
        if self._memory_info is not None and not refresh:
            return self._memory_info
        
        memory_info = {
            'total': 0,
            'available': 0,
            'used': 0,
            'free': 0,
            'percent': 0,
            'swap_total': 0,
            'swap_used': 0,
            'swap_free': 0,
            'swap_percent': 0
        }
        
        try:
            # 物理内存
            mem = psutil.virtual_memory()
            memory_info['total'] = mem.total
            memory_info['available'] = mem.available
            memory_info['used'] = mem.used
            memory_info['free'] = mem.free
            memory_info['percent'] = mem.percent
            
            # 交换内存
            swap = psutil.swap_memory()
            memory_info['swap_total'] = swap.total
            memory_info['swap_used'] = swap.used
            memory_info['swap_free'] = swap.free
            memory_info['swap_percent'] = swap.percent
            
        except Exception as e:
            logger.warning(f"Error getting memory info: {e}")
        
        self._memory_info = memory_info
        return memory_info
    
    def get_system_info(self, refresh: bool = False) -> Dict[str, Any]:
        """获取系统信息
        
        Args:
            refresh: 是否刷新缓存
            
        Returns:
            系统信息字典
        """
        if self._system_info is not None and not refresh:
            return self._system_info
        
        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'hostname': platform.node(),
            'python_version': platform.python_version(),
            'torch_version': None,
            'cuda_available': self.has_cuda(),
            'cuda_version': self.get_cuda_version()
        }
        
        if TORCH_AVAILABLE:
            system_info['torch_version'] = torch.__version__
        
        self._system_info = system_info
        return system_info
    
    def get_optimal_batch_size(self, model_memory_mb: float, gpu_id: int = 0) -> int:
        """根据GPU内存估算最优批处理大小
        
        Args:
            model_memory_mb: 模型内存占用(MB)
            gpu_id: GPU ID
            
        Returns:
            建议的批处理大小
        """
        if not self.has_cuda():
            return 1
        
        gpu_info = self.get_gpu_info()
        if gpu_id >= len(gpu_info):
            return 1
        
        gpu = gpu_info[gpu_id]
        available_memory_mb = gpu['memory_free'] / (1024 * 1024)
        
        # 保留20%的内存作为缓冲
        usable_memory_mb = available_memory_mb * 0.8
        
        # 估算批处理大小
        if model_memory_mb <= 0:
            return 32  # 默认值
        
        batch_size = max(1, int(usable_memory_mb / model_memory_mb))
        
        # 限制最大批处理大小
        return min(batch_size, 128)
    
    def get_optimal_num_workers(self) -> int:
        """获取最优的数据加载器工作线程数
        
        Returns:
            建议的工作线程数
        """
        cpu_info = self.get_cpu_info()
        logical_cores = cpu_info.get('logical_cores', 1)
        
        # 通常使用CPU核心数的一半到全部
        # 考虑到其他进程的需要，使用75%
        num_workers = max(1, int(logical_cores * 0.75))
        
        # 限制最大工作线程数
        return min(num_workers, 16)
    
    def suggest_device_config(self) -> Dict[str, Any]:
        """建议设备配置
        
        Returns:
            设备配置建议
        """
        config = {
            'device': 'cpu',
            'gpu_ids': [],
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': False,
            'mixed_precision': False
        }
        
        if self.has_cuda():
            available_gpus = self.get_available_gpus()
            if available_gpus:
                config['device'] = 'cuda'
                config['gpu_ids'] = available_gpus
                config['pin_memory'] = True
                config['mixed_precision'] = True
                
                # 根据GPU数量调整批处理大小
                config['batch_size'] = min(32 * len(available_gpus), 128)
        
        config['num_workers'] = self.get_optimal_num_workers()
        
        return config
    
    def print_hardware_summary(self):
        """打印硬件摘要信息"""
        print("=== Hardware Summary ===")
        
        # 系统信息
        system_info = self.get_system_info()
        print(f"Platform: {system_info['platform']} {system_info['platform_release']}")
        print(f"Architecture: {system_info['architecture']}")
        print(f"Python: {system_info['python_version']}")
        
        if system_info['torch_version']:
            print(f"PyTorch: {system_info['torch_version']}")
        
        # CPU信息
        cpu_info = self.get_cpu_info()
        print(f"\nCPU: {cpu_info['physical_cores']} physical cores, {cpu_info['logical_cores']} logical cores")
        if cpu_info['max_frequency']:
            print(f"CPU Frequency: {cpu_info['max_frequency']:.0f} MHz (max)")
        
        # 内存信息
        memory_info = self.get_memory_info()
        total_gb = memory_info['total'] / (1024**3)
        available_gb = memory_info['available'] / (1024**3)
        print(f"Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
        
        # GPU信息
        if self.has_cuda():
            print(f"\nCUDA: {system_info['cuda_version']}")
            gpu_info = self.get_gpu_info()
            for gpu in gpu_info:
                memory_gb = gpu['memory_total'] / (1024**3)
                free_gb = gpu['memory_free'] / (1024**3)
                print(f"GPU {gpu['id']}: {gpu['name']} ({free_gb:.1f}GB free / {memory_gb:.1f}GB total)")
        else:
            print("\nCUDA: Not available")
        
        # 配置建议
        config = self.suggest_device_config()
        print(f"\nRecommended config:")
        print(f"  Device: {config['device']}")
        if config['gpu_ids']:
            print(f"  GPU IDs: {config['gpu_ids']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Num workers: {config['num_workers']}")
        print(f"  Mixed precision: {config['mixed_precision']}")
        
        print("========================")


@dataclass
class HardwareConfig:
    """硬件配置信息"""
    gpus: List['GPUInfo']
    system: 'SystemInfo'
    recommended_config: Dict[str, Any]
    
    @classmethod
    def from_hardware_detector(cls, detector: 'HardwareDetector') -> 'HardwareConfig':
        """从HardwareDetector实例创建HardwareConfig"""
        # 创建GPU信息列表
        gpu_info = detector.get_gpu_info()
        gpu_infos = []
        if gpu_info.get('gpu_count', 0) > 0:
            for i in range(gpu_info.get('gpu_count', 0)):
                gpu_info_obj = GPUInfo(
                    id=i,
                    name=gpu_info.get('gpu_names', ['Unknown'])[0] if gpu_info.get('gpu_names') else 'Unknown',
                    memory_total=int(gpu_info.get('total_memory', 0)),
                    memory_free=int(gpu_info.get('available_memory', 0)),
                    compute_capability=(7, 5),  # 默认值
                    is_available=True
                )
                gpu_infos.append(gpu_info_obj)
        
        # 创建系统信息
        memory_info = detector.get_memory_info()
        cpu_info = detector.get_cpu_info()
        system_info = detector.get_system_info()
        system_info_obj = SystemInfo(
            cpu_count=cpu_info.get('logical_cores', 0),
            memory_total=int(memory_info.get('total', 0) / (1024**3)),  # 转换为GB然后转为MB
            memory_available=int(memory_info.get('available', 0) / (1024**3)),
            platform=system_info.get('platform', 'Unknown'),
            python_version=system_info.get('python_version', '3.12'),
            torch_version=system_info.get('torch_version', '2.0'),
            cuda_version=system_info.get('cuda_version')
        )
        
        return cls(
            gpus=gpu_infos,
            system=system_info_obj,
            recommended_config=detector.suggest_device_config()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'gpus': [gpu.to_dict() for gpu in self.gpus],
            'system': self.system.to_dict(),
            'recommended_config': self.recommended_config
        }


@dataclass
class GPUInfo:
    """GPU信息"""
    id: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    compute_capability: Tuple[int, int]
    is_available: bool = True
    temperature: Optional[float] = None
    utilization: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'memory_total': self.memory_total,
            'memory_free': self.memory_free,
            'compute_capability': self.compute_capability,
            'is_available': self.is_available,
            'temperature': self.temperature,
            'utilization': self.utilization
        }


@dataclass
class SystemInfo:
    """系统信息"""
    cpu_count: int
    memory_total: int  # MB
    memory_available: int  # MB
    platform: str
    python_version: str
    torch_version: str = '2.0'
    cuda_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'cpu_count': self.cpu_count,
            'memory_total': self.memory_total,
            'memory_available': self.memory_available,
            'platform': self.platform,
            'python_version': self.python_version,
            'torch_version': self.torch_version,
            'cuda_version': self.cuda_version
        }


def detect_and_configure() -> HardwareConfig:
    """检测硬件并生成配置
    
    Returns:
        硬件配置对象
    """
    detector = HardwareDetector()
    return HardwareConfig.from_hardware_detector(detector)