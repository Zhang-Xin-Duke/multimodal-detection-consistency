#!/usr/bin/env python3
"""实验日志模块

提供统一的实验日志记录功能，支持结构化日志和实验追踪。
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

@dataclass
class ExperimentInfo:
    """实验信息"""
    experiment_id: str
    name: str
    description: str
    config_path: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"  # running, completed, failed
    results: Optional[Dict] = None
    metrics: Optional[Dict] = None
    artifacts: Optional[List[str]] = None

class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, log_file: str, experiment_name: str = None):
        """初始化实验日志记录器
        
        Args:
            log_file: 日志文件路径
            experiment_name: 实验名称
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成实验ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name or 'exp'}_{timestamp}"
        
        # 初始化实验信息
        self.experiment_info = ExperimentInfo(
            experiment_id=self.experiment_id,
            name=experiment_name or "Unnamed Experiment",
            description="",
            config_path="",
            start_time=datetime.now().isoformat(),
            artifacts=[]
        )
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
        # 记录开始
        self.log_experiment_start()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(f"experiment_{self.experiment_id}")
        logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_experiment_start(self):
        """记录实验开始"""
        self.logger.info(f"实验开始: {self.experiment_info.name}")
        self.logger.info(f"实验ID: {self.experiment_id}")
        self.logger.info(f"开始时间: {self.experiment_info.start_time}")
        
        # 保存实验信息
        self._save_experiment_info()
    
    def log_experiment_end(self, status: str = "completed", results: Dict = None):
        """记录实验结束
        
        Args:
            status: 实验状态
            results: 实验结果
        """
        self.experiment_info.end_time = datetime.now().isoformat()
        self.experiment_info.status = status
        if results:
            self.experiment_info.results = results
        
        self.logger.info(f"实验结束: {self.experiment_info.name}")
        self.logger.info(f"结束时间: {self.experiment_info.end_time}")
        self.logger.info(f"状态: {status}")
        
        # 保存实验信息
        self._save_experiment_info()
    
    def log_config(self, config: Dict, config_path: str = ""):
        """记录配置信息
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        self.experiment_info.config_path = config_path
        
        self.logger.info("实验配置:")
        self.logger.info(json.dumps(config, indent=2, ensure_ascii=False))
        
        # 保存配置到单独文件
        config_file = self.log_file.parent / f"{self.experiment_id}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.add_artifact(str(config_file))
    
    def log_metrics(self, metrics: Dict, step: int = None, prefix: str = ""):
        """记录指标
        
        Args:
            metrics: 指标字典
            step: 步骤编号
            prefix: 指标前缀
        """
        if self.experiment_info.metrics is None:
            self.experiment_info.metrics = {}
        
        # 添加时间戳和步骤信息
        timestamp = datetime.now().isoformat()
        metric_entry = {
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        }
        
        metric_key = f"{prefix}_{step}" if prefix and step is not None else str(len(self.experiment_info.metrics))
        self.experiment_info.metrics[metric_key] = metric_entry
        
        # 记录到日志
        log_msg = f"指标 (步骤 {step}): " if step is not None else "指标: "
        if prefix:
            log_msg = f"{prefix} {log_msg}"
        
        self.logger.info(log_msg)
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
        
        # 保存实验信息
        self._save_experiment_info()
    
    def log_progress(self, current: int, total: int, message: str = ""):
        """记录进度
        
        Args:
            current: 当前进度
            total: 总数
            message: 附加消息
        """
        percentage = (current / total) * 100 if total > 0 else 0
        progress_msg = f"进度: {current}/{total} ({percentage:.1f}%)"
        if message:
            progress_msg += f" - {message}"
        
        self.logger.info(progress_msg)
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误
        
        Args:
            error: 异常对象
            context: 错误上下文
        """
        error_msg = f"错误: {str(error)}"
        if context:
            error_msg = f"{context} - {error_msg}"
        
        self.logger.error(error_msg, exc_info=True)
        
        # 更新实验状态
        self.experiment_info.status = "failed"
        self._save_experiment_info()
    
    def log_artifact(self, artifact_path: str, description: str = ""):
        """记录实验产物
        
        Args:
            artifact_path: 产物文件路径
            description: 产物描述
        """
        self.add_artifact(artifact_path)
        
        log_msg = f"产物: {artifact_path}"
        if description:
            log_msg += f" - {description}"
        
        self.logger.info(log_msg)
    
    def add_artifact(self, artifact_path: str):
        """添加实验产物
        
        Args:
            artifact_path: 产物文件路径
        """
        if self.experiment_info.artifacts is None:
            self.experiment_info.artifacts = []
        
        if artifact_path not in self.experiment_info.artifacts:
            self.experiment_info.artifacts.append(artifact_path)
    
    def set_description(self, description: str):
        """设置实验描述
        
        Args:
            description: 实验描述
        """
        self.experiment_info.description = description
        self.logger.info(f"实验描述: {description}")
        self._save_experiment_info()
    
    def _save_experiment_info(self):
        """保存实验信息到文件"""
        info_file = self.log_file.parent / f"{self.experiment_id}_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.experiment_info), f, indent=2, ensure_ascii=False, default=str)
    
    def get_experiment_summary(self) -> Dict:
        """获取实验摘要
        
        Returns:
            实验摘要字典
        """
        summary = {
            'experiment_id': self.experiment_id,
            'name': self.experiment_info.name,
            'status': self.experiment_info.status,
            'start_time': self.experiment_info.start_time,
            'end_time': self.experiment_info.end_time,
            'duration': None
        }
        
        # 计算持续时间
        if self.experiment_info.end_time:
            start = datetime.fromisoformat(self.experiment_info.start_time)
            end = datetime.fromisoformat(self.experiment_info.end_time)
            duration = end - start
            summary['duration'] = str(duration)
        
        # 添加最新指标
        if self.experiment_info.metrics:
            latest_metrics = list(self.experiment_info.metrics.values())[-1]
            summary['latest_metrics'] = latest_metrics['metrics']
        
        return summary

class ExperimentTracker:
    """实验追踪器
    
    管理多个实验的记录和查询。
    """
    
    def __init__(self, experiments_dir: str):
        """初始化实验追踪器
        
        Args:
            experiments_dir: 实验目录
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.experiments_dir / "experiments_index.json"
        self.experiments_index = self._load_index()
    
    def _load_index(self) -> Dict:
        """加载实验索引
        
        Returns:
            实验索引字典
        """
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """保存实验索引"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiments_index, f, indent=2, ensure_ascii=False)
    
    def register_experiment(self, experiment_logger: ExperimentLogger):
        """注册实验
        
        Args:
            experiment_logger: 实验日志记录器
        """
        experiment_id = experiment_logger.experiment_id
        self.experiments_index[experiment_id] = {
            'name': experiment_logger.experiment_info.name,
            'start_time': experiment_logger.experiment_info.start_time,
            'log_file': str(experiment_logger.log_file),
            'status': experiment_logger.experiment_info.status
        }
        self._save_index()
    
    def update_experiment_status(self, experiment_id: str, status: str):
        """更新实验状态
        
        Args:
            experiment_id: 实验ID
            status: 新状态
        """
        if experiment_id in self.experiments_index:
            self.experiments_index[experiment_id]['status'] = status
            self._save_index()
    
    def list_experiments(self, status: str = None) -> List[Dict]:
        """列出实验
        
        Args:
            status: 过滤状态（可选）
            
        Returns:
            实验列表
        """
        experiments = []
        for exp_id, exp_info in self.experiments_index.items():
            if status is None or exp_info['status'] == status:
                experiments.append({
                    'experiment_id': exp_id,
                    **exp_info
                })
        
        # 按开始时间排序
        experiments.sort(key=lambda x: x['start_time'], reverse=True)
        return experiments
    
    def get_experiment_info(self, experiment_id: str) -> Optional[Dict]:
        """获取实验信息
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验信息字典
        """
        if experiment_id not in self.experiments_index:
            return None
        
        # 尝试加载详细信息
        info_file = self.experiments_dir / f"{experiment_id}_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return self.experiments_index[experiment_id]
    
    def cleanup_failed_experiments(self):
        """清理失败的实验"""
        failed_experiments = [exp_id for exp_id, exp_info in self.experiments_index.items() 
                            if exp_info['status'] == 'failed']
        
        for exp_id in failed_experiments:
            # 删除相关文件
            for pattern in [f"{exp_id}_*.json", f"{exp_id}_*.log"]:
                for file_path in self.experiments_dir.glob(pattern):
                    file_path.unlink()
            
            # 从索引中删除
            del self.experiments_index[exp_id]
        
        self._save_index()
        return len(failed_experiments)

# 全局实验追踪器实例
_global_tracker = None

def get_experiment_tracker(experiments_dir: str = None) -> ExperimentTracker:
    """获取全局实验追踪器
    
    Args:
        experiments_dir: 实验目录（仅在首次调用时使用）
        
    Returns:
        实验追踪器实例
    """
    global _global_tracker
    if _global_tracker is None:
        if experiments_dir is None:
            experiments_dir = "./experiments/logs"
        _global_tracker = ExperimentTracker(experiments_dir)
    return _global_tracker

def create_experiment_logger(name: str, log_dir: str = None) -> ExperimentLogger:
    """创建实验日志记录器
    
    Args:
        name: 实验名称
        log_dir: 日志目录
        
    Returns:
        实验日志记录器
    """
    if log_dir is None:
        log_dir = "./experiments/logs"
    
    log_file = Path(log_dir) / f"{name}.log"
    logger = ExperimentLogger(str(log_file), name)
    
    # 注册到全局追踪器
    tracker = get_experiment_tracker(log_dir)
    tracker.register_experiment(logger)
    
    return logger