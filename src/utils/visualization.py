"""可视化工具模块

提供实验结果的可视化功能，包括指标图表、分布图、热力图等。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

logger = logging.getLogger(__name__)

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


@dataclass
class PlotConfig:
    """绘图配置"""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    format: str = 'png'
    save_dir: str = './outputs/plots'
    style: str = 'seaborn'
    color_palette: str = 'husl'
    font_size: int = 12
    title_size: int = 14
    label_size: int = 10


class MetricsVisualizer:
    """指标可视化器"""
    
    def __init__(self, config: PlotConfig = None):
        """
        初始化可视化器
        
        Args:
            config: 绘图配置
        """
        self.config = config or PlotConfig()
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
    
    def plot_retrieval_metrics(self, 
                              metrics_dict: Dict[str, Dict[int, float]],
                              title: str = "检索性能指标",
                              save_name: str = "retrieval_metrics") -> Figure:
        """
        绘制检索指标图
        
        Args:
            metrics_dict: 指标字典，格式为 {metric_name: {k: score}}
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)
        fig.suptitle(title, fontsize=self.config.title_size)
        
        # 准备数据
        k_values = []
        for metric_scores in metrics_dict.values():
            k_values.extend(metric_scores.keys())
        k_values = sorted(list(set(k_values)))
        
        # 绘制每个指标
        metric_names = list(metrics_dict.keys())
        colors = sns.color_palette(self.config.color_palette, len(metric_names))
        
        for i, (metric_name, metric_scores) in enumerate(metrics_dict.items()):
            ax = axes[i // 2, i % 2]
            
            scores = [metric_scores.get(k, 0) for k in k_values]
            ax.plot(k_values, scores, marker='o', linewidth=2, 
                   markersize=6, color=colors[i], label=metric_name)
            
            ax.set_xlabel('K值', fontsize=self.config.label_size)
            ax.set_ylabel('分数', fontsize=self.config.label_size)
            ax.set_title(f'{metric_name}@K', fontsize=self.config.font_size)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 如果指标少于4个，隐藏多余的子图
        for i in range(len(metric_names), 4):
            axes[i // 2, i % 2].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / f"{save_name}.{self.config.format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"检索指标图已保存: {save_path}")
        
        return fig
    
    def plot_detection_metrics(self,
                              metrics: Dict[str, float],
                              title: str = "检测性能指标",
                              save_name: str = "detection_metrics") -> Figure:
        """
        绘制检测指标图
        
        Args:
            metrics: 检测指标字典
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figsize)
        fig.suptitle(title, fontsize=self.config.title_size)
        
        # 主要指标柱状图
        main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        main_values = [metrics.get(metric, 0) for metric in main_metrics]
        main_labels = ['准确率', '精确率', '召回率', 'F1分数']
        
        bars1 = ax1.bar(main_labels, main_values, 
                        color=sns.color_palette(self.config.color_palette, len(main_labels)))
        ax1.set_ylabel('分数', fontsize=self.config.label_size)
        ax1.set_title('主要检测指标', fontsize=self.config.font_size)
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars1, main_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # AUC和AP指标
        auc_ap_metrics = ['auc_score', 'ap_score']
        auc_ap_values = [metrics.get(metric, 0) for metric in auc_ap_metrics]
        auc_ap_labels = ['AUC', 'AP']
        
        bars2 = ax2.bar(auc_ap_labels, auc_ap_values,
                        color=sns.color_palette(self.config.color_palette, len(auc_ap_labels)))
        ax2.set_ylabel('分数', fontsize=self.config.label_size)
        ax2.set_title('AUC和AP指标', fontsize=self.config.font_size)
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars2, auc_ap_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / f"{save_name}.{self.config.format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"检测指标图已保存: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self,
                             cm: np.ndarray,
                             class_names: List[str] = None,
                             title: str = "混淆矩阵",
                             save_name: str = "confusion_matrix") -> Figure:
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f'类别{i}' for i in range(cm.shape[0])]
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': '样本数量'})
        
        ax.set_xlabel('预测标签', fontsize=self.config.label_size)
        ax.set_ylabel('真实标签', fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / f"{save_name}.{self.config.format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存: {save_path}")
        
        return fig
    
    def plot_similarity_distribution(self,
                                   similarities: np.ndarray,
                                   labels: np.ndarray = None,
                                   title: str = "相似度分布",
                                   save_name: str = "similarity_distribution") -> Figure:
        """
        绘制相似度分布图
        
        Args:
            similarities: 相似度数组
            labels: 标签数组（可选）
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(1, 2, figsize=self.config.figsize)
        fig.suptitle(title, fontsize=self.config.title_size)
        
        # 整体分布直方图
        axes[0].hist(similarities, bins=50, alpha=0.7, density=True,
                    color=sns.color_palette(self.config.color_palette)[0])
        axes[0].set_xlabel('相似度', fontsize=self.config.label_size)
        axes[0].set_ylabel('密度', fontsize=self.config.label_size)
        axes[0].set_title('整体相似度分布', fontsize=self.config.font_size)
        axes[0].grid(True, alpha=0.3)
        
        # 如果有标签，绘制分类分布
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = sns.color_palette(self.config.color_palette, len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                label_similarities = similarities[mask]
                axes[1].hist(label_similarities, bins=30, alpha=0.6,
                           density=True, color=colors[i], 
                           label=f'标签 {label}')
            
            axes[1].set_xlabel('相似度', fontsize=self.config.label_size)
            axes[1].set_ylabel('密度', fontsize=self.config.label_size)
            axes[1].set_title('按标签分类的相似度分布', fontsize=self.config.font_size)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            # 如果没有标签，绘制箱线图
            axes[1].boxplot(similarities, vert=True)
            axes[1].set_ylabel('相似度', fontsize=self.config.label_size)
            axes[1].set_title('相似度箱线图', fontsize=self.config.font_size)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / f"{save_name}.{self.config.format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"相似度分布图已保存: {save_path}")
        
        return fig
    
    def plot_threshold_analysis(self,
                               thresholds: np.ndarray,
                               metrics: Dict[str, np.ndarray],
                               title: str = "阈值分析",
                               save_name: str = "threshold_analysis") -> Figure:
        """
        绘制阈值分析图
        
        Args:
            thresholds: 阈值数组
            metrics: 指标字典，格式为 {metric_name: scores_array}
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        colors = sns.color_palette(self.config.color_palette, len(metrics))
        
        for i, (metric_name, scores) in enumerate(metrics.items()):
            ax.plot(thresholds, scores, marker='o', linewidth=2,
                   markersize=4, color=colors[i], label=metric_name)
        
        ax.set_xlabel('阈值', fontsize=self.config.label_size)
        ax.set_ylabel('指标分数', fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / f"{save_name}.{self.config.format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"阈值分析图已保存: {save_path}")
        
        return fig


class InteractiveVisualizer:
    """交互式可视化器（使用Plotly）"""
    
    def __init__(self, save_dir: str = './outputs/interactive_plots'):
        """
        初始化交互式可视化器
        
        Args:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def create_metrics_dashboard(self,
                                retrieval_metrics: Dict[str, Any],
                                detection_metrics: Dict[str, float],
                                title: str = "实验结果仪表板") -> str:
        """
        创建指标仪表板
        
        Args:
            retrieval_metrics: 检索指标
            detection_metrics: 检测指标
            title: 仪表板标题
            
        Returns:
            保存的HTML文件路径
        """
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('检索指标@K', '检测指标', '相似度分布', '性能对比'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 检索指标@K
        if 'recall_at_k' in retrieval_metrics:
            k_values = list(retrieval_metrics['recall_at_k'].keys())
            recall_values = list(retrieval_metrics['recall_at_k'].values())
            
            fig.add_trace(
                go.Scatter(x=k_values, y=recall_values, mode='lines+markers',
                          name='Recall@K', line=dict(color='blue')),
                row=1, col=1
            )
        
        if 'precision_at_k' in retrieval_metrics:
            precision_values = list(retrieval_metrics['precision_at_k'].values())
            fig.add_trace(
                go.Scatter(x=k_values, y=precision_values, mode='lines+markers',
                          name='Precision@K', line=dict(color='red')),
                row=1, col=1
            )
        
        # 检测指标柱状图
        detection_names = list(detection_metrics.keys())
        detection_values = list(detection_metrics.values())
        
        fig.add_trace(
            go.Bar(x=detection_names, y=detection_values, name='检测指标',
                  marker_color='lightblue'),
            row=1, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=800
        )
        
        # 保存HTML文件
        html_path = self.save_dir / "metrics_dashboard.html"
        pyo.plot(fig, filename=str(html_path), auto_open=False)
        
        logger.info(f"交互式仪表板已保存: {html_path}")
        return str(html_path)
    
    def create_similarity_heatmap(self,
                                 similarity_matrix: np.ndarray,
                                 labels: List[str] = None,
                                 title: str = "相似度热力图") -> str:
        """
        创建相似度热力图
        
        Args:
            similarity_matrix: 相似度矩阵
            labels: 标签列表
            title: 图表标题
            
        Returns:
            保存的HTML文件路径
        """
        if labels is None:
            labels = [f'样本{i}' for i in range(similarity_matrix.shape[0])]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="样本",
            yaxis_title="样本"
        )
        
        # 保存HTML文件
        html_path = self.save_dir / "similarity_heatmap.html"
        pyo.plot(fig, filename=str(html_path), auto_open=False)
        
        logger.info(f"相似度热力图已保存: {html_path}")
        return str(html_path)


class ExperimentVisualizer:
    """实验可视化器"""
    
    def __init__(self, config: PlotConfig = None):
        """
        初始化实验可视化器
        
        Args:
            config: 绘图配置
        """
        self.metrics_viz = MetricsVisualizer(config)
        self.interactive_viz = InteractiveVisualizer()
        self.config = config or PlotConfig()
    
    def visualize_experiment_results(self,
                                   results: Dict[str, Any],
                                   experiment_name: str = "实验结果") -> Dict[str, str]:
        """
        可视化实验结果
        
        Args:
            results: 实验结果字典
            experiment_name: 实验名称
            
        Returns:
            保存的文件路径字典
        """
        saved_files = {}
        
        try:
            # 检索指标可视化
            if 'retrieval' in results:
                retrieval_metrics = results['retrieval']
                metrics_dict = {
                    'Recall': retrieval_metrics.recall_at_k,
                    'Precision': retrieval_metrics.precision_at_k,
                    'NDCG': retrieval_metrics.ndcg_at_k
                }
                
                fig = self.metrics_viz.plot_retrieval_metrics(
                    metrics_dict, 
                    title=f"{experiment_name} - 检索性能",
                    save_name=f"{experiment_name}_retrieval"
                )
                plt.close(fig)
                saved_files['retrieval_plot'] = str(self.metrics_viz.save_dir / f"{experiment_name}_retrieval.{self.config.format}")
            
            # 检测指标可视化
            if 'detection' in results:
                detection_metrics = results['detection']
                metrics_dict = {
                    'accuracy': detection_metrics.accuracy,
                    'precision': detection_metrics.precision,
                    'recall': detection_metrics.recall,
                    'f1_score': detection_metrics.f1_score,
                    'auc_score': detection_metrics.auc_score,
                    'ap_score': detection_metrics.ap_score
                }
                
                fig = self.metrics_viz.plot_detection_metrics(
                    metrics_dict,
                    title=f"{experiment_name} - 检测性能",
                    save_name=f"{experiment_name}_detection"
                )
                plt.close(fig)
                saved_files['detection_plot'] = str(self.metrics_viz.save_dir / f"{experiment_name}_detection.{self.config.format}")
                
                # 混淆矩阵
                if hasattr(detection_metrics, 'confusion_matrix'):
                    fig = self.metrics_viz.plot_confusion_matrix(
                        detection_metrics.confusion_matrix,
                        class_names=['正常', '对抗'],
                        title=f"{experiment_name} - 混淆矩阵",
                        save_name=f"{experiment_name}_confusion_matrix"
                    )
                    plt.close(fig)
                    saved_files['confusion_matrix'] = str(self.metrics_viz.save_dir / f"{experiment_name}_confusion_matrix.{self.config.format}")
            
            # 相似度分布
            if 'similarities' in results:
                similarities = results['similarities']
                labels = results.get('labels', None)
                
                fig = self.metrics_viz.plot_similarity_distribution(
                    similarities, labels,
                    title=f"{experiment_name} - 相似度分布",
                    save_name=f"{experiment_name}_similarity_dist"
                )
                plt.close(fig)
                saved_files['similarity_distribution'] = str(self.metrics_viz.save_dir / f"{experiment_name}_similarity_dist.{self.config.format}")
            
            # 交互式仪表板
            if 'retrieval' in results and 'detection' in results:
                dashboard_path = self.interactive_viz.create_metrics_dashboard(
                    results['retrieval'].__dict__,
                    {
                        'accuracy': results['detection'].accuracy,
                        'precision': results['detection'].precision,
                        'recall': results['detection'].recall,
                        'f1_score': results['detection'].f1_score
                    },
                    title=f"{experiment_name} - 交互式仪表板"
                )
                saved_files['interactive_dashboard'] = dashboard_path
            
            logger.info(f"实验结果可视化完成，共保存 {len(saved_files)} 个文件")
            
        except Exception as e:
            logger.error(f"可视化过程中出现错误: {e}")
            raise
        
        return saved_files
    
    def create_comparison_plot(self,
                              results_dict: Dict[str, Dict[str, Any]],
                              metric_name: str = 'f1_score',
                              title: str = "方法对比",
                              save_name: str = "method_comparison") -> str:
        """
        创建方法对比图
        
        Args:
            results_dict: 结果字典，格式为 {method_name: results}
            metric_name: 对比的指标名称
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        methods = list(results_dict.keys())
        scores = []
        
        for method, results in results_dict.items():
            if 'detection' in results:
                score = getattr(results['detection'], metric_name, 0)
                scores.append(score)
            else:
                scores.append(0)
        
        bars = ax.bar(methods, scores, 
                     color=sns.color_palette(self.config.color_palette, len(methods)))
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel(metric_name, fontsize=self.config.label_size)
        ax.set_title(title, fontsize=self.config.title_size)
        ax.set_ylim(0, max(scores) * 1.1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        save_path = self.metrics_viz.save_dir / f"{save_name}.{self.config.format}"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"方法对比图已保存: {save_path}")
        return str(save_path)


def create_experiment_visualizer(config: PlotConfig = None) -> ExperimentVisualizer:
    """
    创建实验可视化器
    
    Args:
        config: 绘图配置
        
    Returns:
        实验可视化器实例
    """
    return ExperimentVisualizer(config)