#!/usr/bin/env python3
"""随机种子管理模块

提供统一的随机种子设置功能，确保实验的可重现性。
"""

import os
import random
import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def set_random_seed(seed: int, deterministic: bool = True, benchmark: bool = False):
    """设置全局随机种子
    
    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性模式（可能影响性能）
        benchmark: 是否启用CUDNN基准模式（可能影响可重现性）
    """
    logger.info(f"设置随机种子: {seed}")
    
    # Python内置random模块
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CUDNN设置
    if torch.cuda.is_available():
        if deterministic:
            # 确保确定性，但可能影响性能
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("启用CUDNN确定性模式")
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = benchmark
            if benchmark:
                logger.info("启用CUDNN基准模式")
    
    logger.info(f"随机种子设置完成: {seed}")

def get_random_state():
    """获取当前随机状态
    
    Returns:
        包含各种随机状态的字典
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state()
        if torch.cuda.device_count() > 1:
            state['torch_cuda_random_all'] = torch.cuda.get_rng_state_all()
    
    return state

def set_random_state(state: dict):
    """恢复随机状态
    
    Args:
        state: 随机状态字典
    """
    if 'python_random' in state:
        random.setstate(state['python_random'])
    
    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])
    
    if 'torch_random' in state:
        torch.set_rng_state(state['torch_random'])
    
    if torch.cuda.is_available():
        if 'torch_cuda_random' in state:
            torch.cuda.set_rng_state(state['torch_cuda_random'])
        
        if 'torch_cuda_random_all' in state and torch.cuda.device_count() > 1:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_all'])
    
    logger.info("随机状态已恢复")

def save_random_state(filepath: str):
    """保存当前随机状态到文件
    
    Args:
        filepath: 保存路径
    """
    state = get_random_state()
    torch.save(state, filepath)
    logger.info(f"随机状态已保存到: {filepath}")

def load_random_state(filepath: str):
    """从文件加载随机状态
    
    Args:
        filepath: 文件路径
    """
    state = torch.load(filepath)
    set_random_state(state)
    logger.info(f"随机状态已从文件加载: {filepath}")

class RandomStateManager:
    """随机状态管理器
    
    用于在实验过程中管理和切换随机状态。
    """
    
    def __init__(self, seed: Optional[int] = None):
        """初始化随机状态管理器
        
        Args:
            seed: 初始随机种子
        """
        self.states = {}
        self.current_state = None
        
        if seed is not None:
            self.set_seed(seed)
    
    def set_seed(self, seed: int, state_name: str = 'default'):
        """设置随机种子并保存状态
        
        Args:
            seed: 随机种子
            state_name: 状态名称
        """
        set_random_seed(seed)
        self.save_state(state_name)
        self.current_state = state_name
    
    def save_state(self, state_name: str):
        """保存当前随机状态
        
        Args:
            state_name: 状态名称
        """
        self.states[state_name] = get_random_state()
        logger.info(f"随机状态已保存: {state_name}")
    
    def load_state(self, state_name: str):
        """加载指定的随机状态
        
        Args:
            state_name: 状态名称
        """
        if state_name not in self.states:
            raise ValueError(f"状态 '{state_name}' 不存在")
        
        set_random_state(self.states[state_name])
        self.current_state = state_name
        logger.info(f"随机状态已加载: {state_name}")
    
    def create_checkpoint(self, checkpoint_name: str):
        """创建随机状态检查点
        
        Args:
            checkpoint_name: 检查点名称
        """
        self.save_state(f"checkpoint_{checkpoint_name}")
    
    def restore_checkpoint(self, checkpoint_name: str):
        """恢复到指定检查点
        
        Args:
            checkpoint_name: 检查点名称
        """
        self.load_state(f"checkpoint_{checkpoint_name}")
    
    def list_states(self):
        """列出所有保存的状态
        
        Returns:
            状态名称列表
        """
        return list(self.states.keys())
    
    def remove_state(self, state_name: str):
        """删除指定状态
        
        Args:
            state_name: 状态名称
        """
        if state_name in self.states:
            del self.states[state_name]
            logger.info(f"状态已删除: {state_name}")
        else:
            logger.warning(f"状态不存在: {state_name}")
    
    def export_states(self, filepath: str):
        """导出所有状态到文件
        
        Args:
            filepath: 导出路径
        """
        torch.save({
            'states': self.states,
            'current_state': self.current_state
        }, filepath)
        logger.info(f"状态已导出到: {filepath}")
    
    def import_states(self, filepath: str):
        """从文件导入状态
        
        Args:
            filepath: 文件路径
        """
        data = torch.load(filepath)
        self.states = data['states']
        self.current_state = data.get('current_state')
        logger.info(f"状态已从文件导入: {filepath}")

class ReproducibleExperiment:
    """可重现实验上下文管理器
    
    确保实验在指定的随机状态下运行。
    """
    
    def __init__(self, seed: int, deterministic: bool = True):
        """初始化可重现实验
        
        Args:
            seed: 随机种子
            deterministic: 是否启用确定性模式
        """
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = None
        self.original_cudnn_deterministic = None
        self.original_cudnn_benchmark = None
    
    def __enter__(self):
        """进入上下文"""
        # 保存原始状态
        self.original_state = get_random_state()
        
        if torch.cuda.is_available():
            self.original_cudnn_deterministic = torch.backends.cudnn.deterministic
            self.original_cudnn_benchmark = torch.backends.cudnn.benchmark
        
        # 设置新的随机种子
        set_random_seed(self.seed, self.deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        # 恢复原始状态
        if self.original_state is not None:
            set_random_state(self.original_state)
        
        if torch.cuda.is_available():
            if self.original_cudnn_deterministic is not None:
                torch.backends.cudnn.deterministic = self.original_cudnn_deterministic
            if self.original_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = self.original_cudnn_benchmark

def reproducible_experiment(seed: int, deterministic: bool = True):
    """可重现实验装饰器
    
    Args:
        seed: 随机种子
        deterministic: 是否启用确定性模式
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ReproducibleExperiment(seed, deterministic):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# 全局随机状态管理器
_global_state_manager = None

def get_global_state_manager() -> RandomStateManager:
    """获取全局随机状态管理器
    
    Returns:
        全局随机状态管理器实例
    """
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = RandomStateManager()
    return _global_state_manager

def reset_global_state_manager():
    """重置全局随机状态管理器"""
    global _global_state_manager
    _global_state_manager = None

# 便捷函数
def ensure_reproducibility(seed: int = 42):
    """确保实验可重现性
    
    Args:
        seed: 随机种子
    """
    set_random_seed(seed, deterministic=True, benchmark=False)
    logger.info(f"已启用完全可重现模式，种子: {seed}")

def enable_fast_training(seed: int = 42):
    """启用快速训练模式（可能牺牲一些可重现性）
    
    Args:
        seed: 随机种子
    """
    set_random_seed(seed, deterministic=False, benchmark=True)
    logger.info(f"已启用快速训练模式，种子: {seed}")