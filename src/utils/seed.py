"""随机种子设置工具模块

提供统一的随机种子设置功能，确保实验的可重现性。
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """设置所有相关库的随机种子
    
    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性模式（可能影响性能）
    """
    # Python内置random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # 启用确定性模式（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 设置PyTorch的确定性算法
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)


def get_random_seed() -> Optional[int]:
    """获取当前PyTorch的随机种子
    
    Returns:
        当前的随机种子值，如果无法获取则返回None
    """
    try:
        return torch.initial_seed()
    except Exception:
        return None


def create_reproducible_generator(seed: int) -> torch.Generator:
    """创建可重现的随机数生成器
    
    Args:
        seed: 随机种子值
        
    Returns:
        配置好的PyTorch随机数生成器
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class SeedContext:
    """随机种子上下文管理器
    
    在with语句块内临时设置随机种子，退出时恢复原状态。
    """
    
    def __init__(self, seed: int, deterministic: bool = False):
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = {}
        
    def __enter__(self):
        # 保存当前状态
        self.original_state = {
            'python_state': random.getstate(),
            'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(),
            'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }
        
        # 设置新的种子
        set_random_seed(self.seed, self.deterministic)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原状态
        random.setstate(self.original_state['python_state'])
        np.random.set_state(self.original_state['numpy_state'])
        torch.set_rng_state(self.original_state['torch_state'])
        
        if self.original_state['cuda_state'] is not None:
            torch.cuda.set_rng_state_all(self.original_state['cuda_state'])
            
        torch.backends.cudnn.deterministic = self.original_state['cudnn_deterministic']
        torch.backends.cudnn.benchmark = self.original_state['cudnn_benchmark']


# 默认种子值
DEFAULT_SEED = 42

# 导出的函数和类
__all__ = [
    'set_random_seed',
    'get_random_seed', 
    'create_reproducible_generator',
    'SeedContext',
    'DEFAULT_SEED'
]