"""配置管理器模块

提供全局配置管理功能。
"""

from .config import ConfigManager

# 创建全局配置管理器实例
config_manager = ConfigManager()

# 便捷函数
def get_config(section=None):
    """获取配置"""
    if section:
        return getattr(config_manager.config, section, None)
    return config_manager.config

def load_config(config_path):
    """加载配置文件"""
    return config_manager.load_config(config_path)

def save_config(config_path, config=None):
    """保存配置文件"""
    return config_manager.save_config(config_path, config)

def update_config(updates):
    """更新配置"""
    return config_manager.update_config(updates)

def validate_config(config=None):
    """验证配置"""
    return config_manager.validate_config(config)

__all__ = [
    'config_manager',
    'get_config',
    'load_config', 
    'save_config',
    'update_config',
    'validate_config'
]