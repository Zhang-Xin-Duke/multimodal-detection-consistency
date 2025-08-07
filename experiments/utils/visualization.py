"""可视化模块

提供实验结果的可视化功能，包括：
- ROC曲线和PR曲线
- 分数分布图
- 消融实验结果图
- 对比图像展示
- 混淆矩阵热图
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

class VisualizationManager:
    """可视化管理器
    
    统一管理所有可视化功能，提供一致的接口和样式
    """
    
    def __init__(self, output_dir: str, dpi: int = 300, figsize: Tuple[int, int] = (10, 8)):
        """
        初始化可视化管理器
        
        Args:
            output_dir: 输出目录
            dpi: 图像分辨率
            figsize: 图像尺寸
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        
        # 创建子目录
        (self.output_dir / "roc_curves").mkdir(exist_ok=True)
        (self.output_dir / "pr_curves").mkdir(exist_ok=True)
        (self.output_dir / "distributions").mkdir(exist_ok=True)
        (self.output_dir / "ablations").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "confusion_matrices").mkdir(exist_ok=True)
        
        logger.info(f"可视化管理器初始化完成，输出目录: {output_dir}")
    
    def plot_roc_curves(self, 
                       results: Dict[str, Dict[str, Any]], 
                       title: str = "ROC Curves",
                       filename: str = "roc_curves.png") -> str:
        """绘制ROC曲线
        
        Args:
            results: 结果字典，格式为 {method_name: {'y_true': [...], 'y_scores': [...]}}
            title: 图表标题
            filename: 保存文件名
            
        Returns:
            保存的文件路径
        """
        plt.figure(figsize=self.figsize)
        
        for method_name, data in results.items():
            y_true = np.array(data['y_true'])
            y_scores = np.array(data['y_scores'])
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            
            # 计算AUC
            from sklearn.metrics import auc
            roc_auc = auc(fpr, tpr)
            
            # 绘制曲线
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{method_name} (AUC = {roc_auc:.3f})')
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = self.output_dir / "roc_curves" / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC曲线已保存: {save_path}")
        return str(save_path)
    
    def plot_pr_curves(self, 
                      results: Dict[str, Dict[str, Any]], 
                      title: str = "Precision-Recall Curves",
                      filename: str = "pr_curves.png") -> str:
        """绘制PR曲线
        
        Args:
            results: 结果字典
            title: 图表标题
            filename: 保存文件名
            
        Returns:
            保存的文件路径
        """
        plt.figure(figsize=self.figsize)
        
        for method_name, data in results.items():
            y_true = np.array(data['y_true'])
            y_scores = np.array(data['y_scores'])
            
            # 计算PR曲线
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            
            # 计算AP
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y_true, y_scores)
            
            # 绘制曲线
            plt.plot(recall, precision, linewidth=2, 
                    label=f'{method_name} (AP = {ap:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = self.output_dir / "pr_curves" / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR曲线已保存: {save_path}")
        return str(save_path)
    
    def plot_score_distribution(self, 
                              normal_scores: List[float],
                              adversarial_scores: List[float],
                              title: str = "Score Distribution",
                              filename: str = "score_distribution.png") -> str:
        """绘制分数分布图
        
        Args:
            normal_scores: 正常样本分数
            adversarial_scores: 对抗样本分数
            title: 图表标题
            filename: 保存文件名
            
        Returns:
            保存的文件路径
        """
        plt.figure(figsize=self.figsize)
        
        # 绘制直方图
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', 
                density=True, color='blue')
        plt.hist(adversarial_scores, bins=50, alpha=0.7, label='Adversarial', 
                density=True, color='red')
        
        # 添加统计信息
        normal_mean = np.mean(normal_scores)
        adversarial_mean = np.mean(adversarial_scores)
        
        plt.axvline(normal_mean, color='blue', linestyle='--', alpha=0.8,
                   label=f'Normal Mean: {normal_mean:.3f}')
        plt.axvline(adversarial_mean, color='red', linestyle='--', alpha=0.8,
                   label=f'Adversarial Mean: {adversarial_mean:.3f}')
        
        plt.xlabel('Detection Score')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = self.output_dir / "distributions" / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"分数分布图已保存: {save_path}")
        return str(save_path)
    
    def plot_ablation_results(self, 
                            ablation_data: Dict[str, Dict[str, float]],
                            metrics: List[str] = ['accuracy', 'f1_score', 'roc_auc'],
                            title: str = "Ablation Study Results",
                            filename: str = "ablation_results.png") -> str:
        """绘制消融实验结果
        
        Args:
            ablation_data: 消融数据，格式为 {config_name: {metric: value}}
            metrics: 要显示的指标列表
            title: 图表标题
            filename: 保存文件名
            
        Returns:
            保存的文件路径
        """
        # 准备数据
        configs = list(ablation_data.keys())
        n_configs = len(configs)
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [ablation_data[config].get(metric, 0) for config in configs]
            
            # 创建条形图
            bars = axes[i].bar(range(n_configs), values, alpha=0.8)
            
            # 添加数值标签
            for j, (bar, value) in enumerate(zip(bars, values)):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            axes[i].set_xlabel('Configuration')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xticks(range(n_configs))
            axes[i].set_xticklabels(configs, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, max(values) * 1.1)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.output_dir / "ablations" / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"消融实验结果图已保存: {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(self, 
                            y_true: List[int], 
                            y_pred: List[int],
                            class_names: List[str] = ['Normal', 'Adversarial'],
                            title: str = "Confusion Matrix",
                            filename: str = "confusion_matrix.png") -> str:
        """绘制混淆矩阵热图
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            title: 图表标题
            filename: 保存文件名
            
        Returns:
            保存的文件路径
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(8, 6))
        
        # 创建热图
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage (%)'})
        
        # 添加数量标注
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        
        # 保存图像
        save_path = self.output_dir / "confusion_matrices" / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩阵已保存: {save_path}")
        return str(save_path)
    
    def save_comparison_images(self, 
                             image_groups: Dict[str, List[Union[torch.Tensor, np.ndarray, Image.Image]]],
                             titles: Optional[Dict[str, List[str]]] = None,
                             filename: str = "image_comparison.png",
                             max_images_per_group: int = 5) -> str:
        """保存图像对比
        
        Args:
            image_groups: 图像组字典，格式为 {group_name: [images]}
            titles: 每个图像的标题（可选）
            filename: 保存文件名
            max_images_per_group: 每组最大图像数
            
        Returns:
            保存的文件路径
        """
        group_names = list(image_groups.keys())
        n_groups = len(group_names)
        
        # 确定网格大小
        max_images = min(max_images_per_group, 
                        max(len(images) for images in image_groups.values()))
        
        fig, axes = plt.subplots(n_groups, max_images, 
                               figsize=(max_images * 3, n_groups * 3))
        
        if n_groups == 1:
            axes = axes.reshape(1, -1)
        if max_images == 1:
            axes = axes.reshape(-1, 1)
        
        for i, group_name in enumerate(group_names):
            images = image_groups[group_name][:max_images]
            group_titles = titles.get(group_name, []) if titles else []
            
            for j, image in enumerate(images):
                # 转换图像格式
                if isinstance(image, torch.Tensor):
                    # 假设是归一化的张量
                    if image.dim() == 3 and image.shape[0] in [1, 3]:
                        image = image.permute(1, 2, 0)
                    image = image.cpu().numpy()
                    
                    # 反归一化（如果需要）
                    if image.min() < 0:
                        image = (image - image.min()) / (image.max() - image.min())
                    
                elif isinstance(image, np.ndarray):
                    if image.min() < 0:
                        image = (image - image.min()) / (image.max() - image.min())
                
                elif isinstance(image, Image.Image):
                    image = np.array(image)
                
                # 显示图像
                if image.shape[-1] == 1:  # 灰度图
                    axes[i, j].imshow(image.squeeze(), cmap='gray')
                else:  # 彩色图
                    axes[i, j].imshow(image)
                
                axes[i, j].axis('off')
                
                # 设置标题
                if j < len(group_titles):
                    axes[i, j].set_title(group_titles[j], fontsize=10)
                elif j == 0:
                    axes[i, j].set_title(group_name, fontsize=12, fontweight='bold')
            
            # 隐藏多余的子图
            for j in range(len(images), max_images):
                axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.output_dir / "comparisons" / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"图像对比已保存: {save_path}")
        return str(save_path)
    
    def plot_metrics_comparison(self, 
                              results: Dict[str, Dict[str, float]],
                              metrics: List[str],
                              title: str = "Metrics Comparison",
                              filename: str = "metrics_comparison.png") -> str:
        """绘制指标对比图
        
        Args:
            results: 结果字典，格式为 {method_name: {metric: value}}
            metrics: 要对比的指标列表
            title: 图表标题
            filename: 保存文件名
            
        Returns:
            保存的文件路径
        """
        # 准备数据
        methods = list(results.keys())
        n_methods = len(methods)
        n_metrics = len(metrics)
        
        # 创建数据矩阵
        data_matrix = np.zeros((n_methods, n_metrics))
        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics):
                data_matrix[i, j] = results[method].get(metric, 0)
        
        # 创建热图
        plt.figure(figsize=(n_metrics * 2, n_methods * 0.8))
        
        sns.heatmap(data_matrix, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=methods,
                   annot=True, fmt='.3f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Score'})
        
        plt.title(title)
        plt.xlabel('Metrics')
        plt.ylabel('Methods')
        plt.tight_layout()
        
        # 保存图像
        save_path = self.output_dir / "comparisons" / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"指标对比图已保存: {save_path}")
        return str(save_path)
    
    def create_summary_report(self, 
                            results_summary: Dict[str, Any],
                            filename: str = "summary_report.png") -> str:
        """创建汇总报告图
        
        Args:
            results_summary: 结果汇总
            filename: 保存文件名
            
        Returns:
            保存的文件路径
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 攻击成功率对比
        if 'attack_success_rate' in results_summary:
            ax1 = fig.add_subplot(gs[0, 0])
            asr_data = results_summary['attack_success_rate']
            methods = list(asr_data.keys())
            values = [asr_data[method]['mean'] for method in methods]
            errors = [asr_data[method]['std'] for method in methods]
            
            ax1.bar(methods, values, yerr=errors, capsize=5, alpha=0.8)
            ax1.set_title('Attack Success Rate')
            ax1.set_ylabel('ASR')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. 检测性能对比
        if 'detection_metrics' in results_summary:
            ax2 = fig.add_subplot(gs[0, 1])
            det_data = results_summary['detection_metrics']
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for method in det_data:
                values = [det_data[method][metric]['mean'] for metric in metrics if metric in det_data[method]]
                ax2.plot(metrics[:len(values)], values, marker='o', label=method)
            
            ax2.set_title('Detection Performance')
            ax2.set_ylabel('Score')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. ROC AUC对比
        if 'roc_auc_comparison' in results_summary:
            ax3 = fig.add_subplot(gs[0, 2])
            roc_data = results_summary['roc_auc_comparison']
            methods = list(roc_data.keys())
            values = list(roc_data.values())
            
            ax3.bar(methods, values, alpha=0.8)
            ax3.set_title('ROC AUC Comparison')
            ax3.set_ylabel('AUC')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 消融实验结果
        if 'ablation_results' in results_summary:
            ax4 = fig.add_subplot(gs[1, :])
            ablation_data = results_summary['ablation_results']
            
            configs = list(ablation_data.keys())
            metrics = ['accuracy', 'f1_score', 'roc_auc']
            
            x = np.arange(len(configs))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [ablation_data[config].get(metric, 0) for config in configs]
                ax4.bar(x + i * width, values, width, label=metric, alpha=0.8)
            
            ax4.set_title('Ablation Study Results')
            ax4.set_xlabel('Configuration')
            ax4.set_ylabel('Score')
            ax4.set_xticks(x + width)
            ax4.set_xticklabels(configs, rotation=45)
            ax4.legend()
        
        # 5. 效率分析
        if 'efficiency_analysis' in results_summary:
            ax5 = fig.add_subplot(gs[2, 0])
            eff_data = results_summary['efficiency_analysis']
            
            components = list(eff_data.keys())
            times = [eff_data[comp]['time'] for comp in components]
            
            ax5.bar(components, times, alpha=0.8)
            ax5.set_title('Processing Time')
            ax5.set_ylabel('Time (ms)')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. 内存使用
        if 'memory_usage' in results_summary:
            ax6 = fig.add_subplot(gs[2, 1])
            mem_data = results_summary['memory_usage']
            
            components = list(mem_data.keys())
            memory = [mem_data[comp] for comp in components]
            
            ax6.bar(components, memory, alpha=0.8, color='orange')
            ax6.set_title('Memory Usage')
            ax6.set_ylabel('Memory (GB)')
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. 总体性能雷达图
        if 'overall_performance' in results_summary:
            ax7 = fig.add_subplot(gs[2, 2], projection='polar')
            perf_data = results_summary['overall_performance']
            
            categories = list(perf_data.keys())
            values = list(perf_data.values())
            
            # 闭合雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax7.plot(angles, values, 'o-', linewidth=2)
            ax7.fill(angles, values, alpha=0.25)
            ax7.set_xticks(angles[:-1])
            ax7.set_xticklabels(categories)
            ax7.set_title('Overall Performance')
        
        plt.suptitle('Experimental Results Summary', fontsize=16, fontweight='bold')
        
        # 保存图像
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"汇总报告已保存: {save_path}")
        return str(save_path)

# 便捷函数
def plot_roc_curve(y_true: List[int], 
                  y_scores: List[float], 
                  title: str = "ROC Curve",
                  save_path: Optional[str] = None) -> Optional[str]:
    """绘制单个ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def plot_pr_curve(y_true: List[int], 
                 y_scores: List[float], 
                 title: str = "Precision-Recall Curve",
                 save_path: Optional[str] = None) -> Optional[str]:
    """绘制单个PR曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def plot_score_distribution(normal_scores: List[float],
                          adversarial_scores: List[float],
                          title: str = "Score Distribution",
                          save_path: Optional[str] = None) -> Optional[str]:
    """绘制分数分布图"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', 
            density=True, color='blue')
    plt.hist(adversarial_scores, bins=50, alpha=0.7, label='Adversarial', 
            density=True, color='red')
    
    plt.xlabel('Detection Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def plot_ablation_results(ablation_data: Dict[str, Dict[str, float]],
                        metrics: List[str] = ['accuracy', 'f1_score', 'roc_auc'],
                        title: str = "Ablation Study Results",
                        save_path: Optional[str] = None) -> Optional[str]:
    """绘制消融实验结果"""
    configs = list(ablation_data.keys())
    n_configs = len(configs)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [ablation_data[config].get(metric, 0) for config in configs]
        
        bars = axes[i].bar(range(n_configs), values, alpha=0.8)
        
        for j, (bar, value) in enumerate(zip(bars, values)):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        axes[i].set_xlabel('Configuration')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_xticks(range(n_configs))
        axes[i].set_xticklabels(configs, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def save_comparison_images(image_groups: Dict[str, List[Union[torch.Tensor, np.ndarray, Image.Image]]],
                         titles: Optional[Dict[str, List[str]]] = None,
                         save_path: str = "image_comparison.png",
                         max_images_per_group: int = 5) -> str:
    """保存图像对比"""
    group_names = list(image_groups.keys())
    n_groups = len(group_names)
    
    max_images = min(max_images_per_group, 
                    max(len(images) for images in image_groups.values()))
    
    fig, axes = plt.subplots(n_groups, max_images, 
                           figsize=(max_images * 3, n_groups * 3))
    
    if n_groups == 1:
        axes = axes.reshape(1, -1)
    if max_images == 1:
        axes = axes.reshape(-1, 1)
    
    for i, group_name in enumerate(group_names):
        images = image_groups[group_name][:max_images]
        group_titles = titles.get(group_name, []) if titles else []
        
        for j, image in enumerate(images):
            # 转换图像格式
            if isinstance(image, torch.Tensor):
                if image.dim() == 3 and image.shape[0] in [1, 3]:
                    image = image.permute(1, 2, 0)
                image = image.cpu().numpy()
                if image.min() < 0:
                    image = (image - image.min()) / (image.max() - image.min())
            
            elif isinstance(image, np.ndarray):
                if image.min() < 0:
                    image = (image - image.min()) / (image.max() - image.min())
            
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            # 显示图像
            if image.shape[-1] == 1:
                axes[i, j].imshow(image.squeeze(), cmap='gray')
            else:
                axes[i, j].imshow(image)
            
            axes[i, j].axis('off')
            
            if j < len(group_titles):
                axes[i, j].set_title(group_titles[j], fontsize=10)
            elif j == 0:
                axes[i, j].set_title(group_name, fontsize=12, fontweight='bold')
        
        # 隐藏多余的子图
        for j in range(len(images), max_images):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"图像对比已保存: {save_path}")
    return save_path