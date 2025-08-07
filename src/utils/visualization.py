"""可视化模块

提供ROC/PR/TSNE/相似度分布等图表功能。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class ROCVisualizer:
    """ROC曲线可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """初始化ROC可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_single_roc(self, y_true: np.ndarray, y_scores: np.ndarray, 
                       label: str = 'ROC Curve', save_path: Optional[str] = None) -> plt.Figure:
        """绘制单个ROC曲线
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            label: 曲线标签
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = np.trapz(tpr, fpr)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制ROC曲线
        ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {auc_score:.3f})')
        
        # 绘制对角线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        # 设置图形属性
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC曲线已保存到: {save_path}")
        
        return fig
    
    def plot_multiple_roc(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                         save_path: Optional[str] = None) -> plt.Figure:
        """绘制多个ROC曲线对比
        
        Args:
            results: 结果字典，格式为 {method_name: (y_true, y_scores)}
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (method_name, (y_true, y_scores)) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_score = np.trapz(tpr, fpr)
            
            color = self.colors[i % len(self.colors)]
            ax.plot(fpr, tpr, linewidth=2, color=color, 
                   label=f'{method_name} (AUC = {auc_score:.3f})')
        
        # 绘制对角线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        # 设置图形属性
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC对比图已保存到: {save_path}")
        
        return fig
    
    def plot_interactive_roc(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                            save_path: Optional[str] = None) -> go.Figure:
        """绘制交互式ROC曲线
        
        Args:
            results: 结果字典
            save_path: 保存路径
            
        Returns:
            plotly图形对象
        """
        fig = go.Figure()
        
        for method_name, (y_true, y_scores) in results.items():
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auc_score = np.trapz(tpr, fpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{method_name} (AUC = {auc_score:.3f})',
                line=dict(width=3),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'FPR: %{x:.3f}<br>' +
                             'TPR: %{y:.3f}<br>' +
                             'Threshold: %{customdata:.3f}<extra></extra>',
                customdata=thresholds
            ))
        
        # 添加对角线
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray'),
            showlegend=True
        ))
        
        fig.update_layout(
            title='Interactive ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"交互式ROC图已保存到: {save_path}")
        
        return fig


class PRVisualizer:
    """Precision-Recall曲线可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """初始化PR可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_single_pr(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      label: str = 'PR Curve', save_path: Optional[str] = None) -> plt.Figure:
        """绘制单个PR曲线
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            label: 曲线标签
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap_score = np.trapz(precision, recall)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制PR曲线
        ax.plot(recall, precision, linewidth=2, label=f'{label} (AP = {ap_score:.3f})')
        
        # 绘制基线
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                  label=f'Baseline (AP = {baseline:.3f})')
        
        # 设置图形属性
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR曲线已保存到: {save_path}")
        
        return fig
    
    def plot_multiple_pr(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                        save_path: Optional[str] = None) -> plt.Figure:
        """绘制多个PR曲线对比
        
        Args:
            results: 结果字典，格式为 {method_name: (y_true, y_scores)}
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        baseline = None
        for i, (method_name, (y_true, y_scores)) in enumerate(results.items()):
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap_score = np.trapz(precision, recall)
            
            if baseline is None:
                baseline = np.sum(y_true) / len(y_true)
            
            color = self.colors[i % len(self.colors)]
            ax.plot(recall, precision, linewidth=2, color=color, 
                   label=f'{method_name} (AP = {ap_score:.3f})')
        
        # 绘制基线
        if baseline is not None:
            ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                      label=f'Baseline (AP = {baseline:.3f})')
        
        # 设置图形属性
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR对比图已保存到: {save_path}")
        
        return fig


class DistributionVisualizer:
    """分布可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """初始化分布可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
    
    def plot_similarity_distribution(self, similarities: Dict[str, np.ndarray], 
                                   labels: Optional[Dict[str, np.ndarray]] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """绘制相似度分布图
        
        Args:
            similarities: 相似度字典，格式为 {method_name: similarity_scores}
            labels: 标签字典，格式为 {method_name: labels}
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        n_methods = len(similarities)
        fig, axes = plt.subplots(1, n_methods, figsize=(self.figsize[0], self.figsize[1]//2))
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method_name, scores) in enumerate(similarities.items()):
            ax = axes[i]
            
            if labels and method_name in labels:
                # 分别绘制正常和对抗样本的分布
                method_labels = labels[method_name]
                normal_scores = scores[method_labels == 0]
                adv_scores = scores[method_labels == 1]
                
                ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', 
                       color='blue', density=True)
                ax.hist(adv_scores, bins=50, alpha=0.7, label='Adversarial', 
                       color='red', density=True)
                ax.legend()
            else:
                # 绘制整体分布
                ax.hist(scores, bins=50, alpha=0.7, color='skyblue', density=True)
            
            ax.set_xlabel('Similarity Score', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"相似度分布图已保存到: {save_path}")
        
        return fig
    
    def plot_score_distribution_comparison(self, scores_dict: Dict[str, Dict[str, np.ndarray]], 
                                         save_path: Optional[str] = None) -> plt.Figure:
        """绘制分数分布对比图
        
        Args:
            scores_dict: 分数字典，格式为 {method: {class: scores}}
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(scores_dict.keys())
        classes = list(next(iter(scores_dict.values())).keys())
        
        x_pos = np.arange(len(methods))
        width = 0.35
        
        for i, class_name in enumerate(classes):
            means = [np.mean(scores_dict[method][class_name]) for method in methods]
            stds = [np.std(scores_dict[method][class_name]) for method in methods]
            
            ax.bar(x_pos + i * width, means, width, yerr=stds, 
                  label=class_name, alpha=0.8, capsize=5)
        
        ax.set_xlabel('Methods', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + width / 2)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分数分布对比图已保存到: {save_path}")
        
        return fig
    
    def plot_violin_distribution(self, data_dict: Dict[str, np.ndarray], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """绘制小提琴图分布
        
        Args:
            data_dict: 数据字典
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 准备数据
        data_list = []
        labels = []
        
        for method_name, scores in data_dict.items():
            data_list.append(scores)
            labels.append(method_name)
        
        # 绘制小提琴图
        parts = ax.violinplot(data_list, positions=range(len(data_list)), 
                             showmeans=True, showmedians=True)
        
        # 设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_list)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Score Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"小提琴图已保存到: {save_path}")
        
        return fig


class DimensionalityVisualizer:
    """降维可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """初始化降维可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
    
    def plot_tsne(self, embeddings: np.ndarray, labels: np.ndarray, 
                 perplexity: int = 30, n_iter: int = 1000,
                 save_path: Optional[str] = None) -> plt.Figure:
        """绘制t-SNE降维图
        
        Args:
            embeddings: 嵌入向量 (N, D)
            labels: 标签 (N,)
            perplexity: t-SNE困惑度参数
            n_iter: 迭代次数
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        logger.info(f"开始t-SNE降维，数据形状: {embeddings.shape}")
        
        # 执行t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                   random_state=42, verbose=1)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 绘制结果
        fig, ax = plt.subplots(figsize=self.figsize)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = 'Adversarial' if label == 1 else 'Normal'
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=label_name, alpha=0.6, s=20)
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"t-SNE图已保存到: {save_path}")
        
        return fig
    
    def plot_pca(self, embeddings: np.ndarray, labels: np.ndarray, 
                save_path: Optional[str] = None) -> plt.Figure:
        """绘制PCA降维图
        
        Args:
            embeddings: 嵌入向量 (N, D)
            labels: 标签 (N,)
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        logger.info(f"开始PCA降维，数据形状: {embeddings.shape}")
        
        # 执行PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # 绘制结果
        fig, ax = plt.subplots(figsize=self.figsize)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = 'Adversarial' if label == 1 else 'Normal'
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=label_name, alpha=0.6, s=20)
        
        # 添加方差解释比例
        explained_var = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({explained_var[0]:.2%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.2%} variance)', fontsize=12)
        ax.set_title('PCA Visualization', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PCA图已保存到: {save_path}")
        
        return fig
    
    def plot_interactive_tsne(self, embeddings: np.ndarray, labels: np.ndarray, 
                             metadata: Optional[Dict[str, List]] = None,
                             perplexity: int = 30, n_iter: int = 1000,
                             save_path: Optional[str] = None) -> go.Figure:
        """绘制交互式t-SNE图
        
        Args:
            embeddings: 嵌入向量
            labels: 标签
            metadata: 额外的元数据信息
            perplexity: t-SNE困惑度参数
            n_iter: 迭代次数
            save_path: 保存路径
            
        Returns:
            plotly图形对象
        """
        logger.info(f"开始交互式t-SNE降维，数据形状: {embeddings.shape}")
        
        # 执行t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                   random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 准备数据
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': ['Adversarial' if l == 1 else 'Normal' for l in labels]
        })
        
        # 添加元数据
        if metadata:
            for key, values in metadata.items():
                df[key] = values
        
        # 创建交互式图
        fig = px.scatter(df, x='x', y='y', color='label', 
                        title='Interactive t-SNE Visualization',
                        hover_data=list(metadata.keys()) if metadata else None)
        
        fig.update_layout(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            width=800, height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"交互式t-SNE图已保存到: {save_path}")
        
        return fig


class ConfusionMatrixVisualizer:
    """混淆矩阵可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (8, 6)):
        """初始化混淆矩阵可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: Optional[List[str]] = None,
                             normalize: bool = False,
                             save_path: Optional[str] = None) -> plt.Figure:
        """绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            normalize: 是否归一化
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        if class_names is None:
            class_names = ['Normal', 'Adversarial']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {save_path}")
        
        return fig
    
    def plot_multiple_confusion_matrices(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                       class_names: Optional[List[str]] = None,
                                       normalize: bool = False,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """绘制多个混淆矩阵对比
        
        Args:
            results: 结果字典，格式为 {method_name: (y_true, y_pred)}
            class_names: 类别名称
            normalize: 是否归一化
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        n_methods = len(results)
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        
        if n_methods == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        if class_names is None:
            class_names = ['Normal', 'Adversarial']
        
        for i, (method_name, (y_true, y_pred)) in enumerate(results.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cm = confusion_matrix(y_true, y_pred)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
            else:
                fmt = 'd'
            
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar=False)
            
            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        # 隐藏多余的子图
        for i in range(n_methods, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵对比图已保存到: {save_path}")
        
        return fig


class MetricsVisualizer:
    """指标可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """初始化指标可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """绘制指标对比图
        
        Args:
            metrics_dict: 指标字典，格式为 {method: {metric: value}}
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        # 准备数据
        methods = list(metrics_dict.keys())
        metrics = list(next(iter(metrics_dict.values())).keys())
        
        # 创建数据矩阵
        data = np.array([[metrics_dict[method][metric] for metric in metrics] 
                        for method in methods])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制热力图
        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')
        
        # 设置标签
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(methods)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(len(methods)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}', 
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Metrics Comparison Heatmap', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Metric Value', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"指标对比图已保存到: {save_path}")
        
        return fig
    
    def plot_radar_chart(self, metrics_dict: Dict[str, Dict[str, float]], 
                        save_path: Optional[str] = None) -> plt.Figure:
        """绘制雷达图
        
        Args:
            metrics_dict: 指标字典
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        methods = list(metrics_dict.keys())
        metrics = list(next(iter(metrics_dict.values())).keys())
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            values = [metrics_dict[method][metric] for metric in metrics]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"雷达图已保存到: {save_path}")
        
        return fig


class VisualizationManager:
    """可视化管理器
    
    统一管理所有可视化功能。
    """
    
    def __init__(self, output_dir: str = "./figures"):
        """初始化可视化管理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各个可视化器
        self.roc_visualizer = ROCVisualizer()
        self.pr_visualizer = PRVisualizer()
        self.dist_visualizer = DistributionVisualizer()
        self.dim_visualizer = DimensionalityVisualizer()
        self.cm_visualizer = ConfusionMatrixVisualizer()
        self.metrics_visualizer = MetricsVisualizer()
    
    def create_comprehensive_report(self, results: Dict[str, Any], 
                                  experiment_name: str = "experiment") -> Dict[str, str]:
        """创建综合可视化报告
        
        Args:
            results: 实验结果
            experiment_name: 实验名称
            
        Returns:
            生成的图片路径字典
        """
        logger.info(f"开始创建综合可视化报告: {experiment_name}")
        
        saved_paths = {}
        
        # 1. ROC曲线
        if 'detection_results' in results:
            roc_path = self.output_dir / f"{experiment_name}_roc_curves.png"
            self.roc_visualizer.plot_multiple_roc(
                results['detection_results'], str(roc_path)
            )
            saved_paths['roc'] = str(roc_path)
        
        # 2. PR曲线
        if 'detection_results' in results:
            pr_path = self.output_dir / f"{experiment_name}_pr_curves.png"
            self.pr_visualizer.plot_multiple_pr(
                results['detection_results'], str(pr_path)
            )
            saved_paths['pr'] = str(pr_path)
        
        # 3. 相似度分布
        if 'similarity_scores' in results:
            dist_path = self.output_dir / f"{experiment_name}_similarity_distribution.png"
            self.dist_visualizer.plot_similarity_distribution(
                results['similarity_scores'], 
                results.get('labels'), 
                str(dist_path)
            )
            saved_paths['distribution'] = str(dist_path)
        
        # 4. t-SNE可视化
        if 'embeddings' in results and 'labels' in results:
            tsne_path = self.output_dir / f"{experiment_name}_tsne.png"
            self.dim_visualizer.plot_tsne(
                results['embeddings'], 
                results['labels'], 
                save_path=str(tsne_path)
            )
            saved_paths['tsne'] = str(tsne_path)
        
        # 5. 混淆矩阵
        if 'confusion_matrices' in results:
            cm_path = self.output_dir / f"{experiment_name}_confusion_matrices.png"
            self.cm_visualizer.plot_multiple_confusion_matrices(
                results['confusion_matrices'], 
                save_path=str(cm_path)
            )
            saved_paths['confusion_matrix'] = str(cm_path)
        
        # 6. 指标对比
        if 'metrics_comparison' in results:
            metrics_path = self.output_dir / f"{experiment_name}_metrics_comparison.png"
            self.metrics_visualizer.plot_metrics_comparison(
                results['metrics_comparison'], 
                str(metrics_path)
            )
            saved_paths['metrics'] = str(metrics_path)
        
        # 7. 雷达图
        if 'metrics_comparison' in results:
            radar_path = self.output_dir / f"{experiment_name}_radar_chart.png"
            self.metrics_visualizer.plot_radar_chart(
                results['metrics_comparison'], 
                str(radar_path)
            )
            saved_paths['radar'] = str(radar_path)
        
        logger.info(f"综合可视化报告创建完成，共生成 {len(saved_paths)} 个图表")
        return saved_paths
    
    def create_interactive_dashboard(self, results: Dict[str, Any], 
                                   experiment_name: str = "experiment") -> str:
        """创建交互式仪表板
        
        Args:
            results: 实验结果
            experiment_name: 实验名称
            
        Returns:
            HTML文件路径
        """
        logger.info(f"开始创建交互式仪表板: {experiment_name}")
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curves', 'Similarity Distribution', 
                           't-SNE Visualization', 'Metrics Comparison'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 添加ROC曲线
        if 'detection_results' in results:
            for method_name, (y_true, y_scores) in results['detection_results'].items():
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc_score = np.trapz(tpr, fpr)
                
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode='lines', 
                             name=f'{method_name} (AUC={auc_score:.3f})'),
                    row=1, col=1
                )
        
        # 添加相似度分布
        if 'similarity_scores' in results:
            for method_name, scores in results['similarity_scores'].items():
                fig.add_trace(
                    go.Histogram(x=scores, name=method_name, opacity=0.7),
                    row=1, col=2
                )
        
        # 添加t-SNE（如果有嵌入数据）
        if 'embeddings' in results and 'labels' in results:
            # 这里简化处理，实际应该先计算t-SNE
            pass
        
        # 更新布局
        fig.update_layout(
            title=f'Interactive Dashboard - {experiment_name}',
            height=800,
            showlegend=True
        )
        
        # 保存HTML文件
        html_path = self.output_dir / f"{experiment_name}_dashboard.html"
        fig.write_html(str(html_path))
        
        logger.info(f"交互式仪表板已保存到: {html_path}")
        return str(html_path)


def create_visualization_manager(output_dir: str = "./figures") -> VisualizationManager:
    """创建可视化管理器
    
    Args:
        output_dir: 输出目录
        
    Returns:
        可视化管理器实例
    """
    return VisualizationManager(output_dir)


class ExperimentVisualizer:
    """实验可视化器
    
    提供实验结果的综合可视化功能。
    """
    
    def __init__(self, output_dir: str = "./experiment_figures"):
        """初始化实验可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各种可视化器
        self.roc_visualizer = ROCVisualizer()
        self.pr_visualizer = PRVisualizer()
        self.tsne_visualizer = TSNEVisualizer()
        self.similarity_visualizer = SimilarityDistributionVisualizer()
        self.manager = VisualizationManager(str(self.output_dir))
        
        logger.info(f"实验可视化器初始化完成，输出目录: {self.output_dir}")
    
    def visualize_experiment_results(self, 
                                   experiment_results: Dict[str, Any],
                                   experiment_name: str = "experiment") -> Dict[str, str]:
        """可视化实验结果
        
        Args:
            experiment_results: 实验结果字典
            experiment_name: 实验名称
            
        Returns:
            Dict[str, str]: 生成的图表文件路径
        """
        saved_files = {}
        
        try:
            # 检测性能可视化
            if 'detection_metrics' in experiment_results:
                detection_files = self._visualize_detection_results(
                    experiment_results['detection_metrics'], experiment_name
                )
                saved_files.update(detection_files)
            
            # 攻击效果可视化
            if 'attack_results' in experiment_results:
                attack_files = self._visualize_attack_results(
                    experiment_results['attack_results'], experiment_name
                )
                saved_files.update(attack_files)
            
            # 嵌入向量可视化
            if 'embeddings' in experiment_results:
                embedding_files = self._visualize_embeddings(
                    experiment_results['embeddings'], experiment_name
                )
                saved_files.update(embedding_files)
            
            # 相似度分布可视化
            if 'similarity_scores' in experiment_results:
                similarity_files = self._visualize_similarity_distributions(
                    experiment_results['similarity_scores'], experiment_name
                )
                saved_files.update(similarity_files)
            
            # 生成综合报告
            report_file = self._generate_experiment_report(
                experiment_results, experiment_name
            )
            saved_files['report'] = report_file
            
            logger.info(f"实验 {experiment_name} 可视化完成，共生成 {len(saved_files)} 个文件")
            
        except Exception as e:
            logger.error(f"实验可视化失败: {e}")
            raise
        
        return saved_files
    
    def _visualize_detection_results(self, 
                                   detection_metrics: Dict[str, Any],
                                   experiment_name: str) -> Dict[str, str]:
        """可视化检测结果
        
        Args:
            detection_metrics: 检测指标
            experiment_name: 实验名称
            
        Returns:
            Dict[str, str]: 生成的文件路径
        """
        files = {}
        
        # ROC曲线
        if 'y_true' in detection_metrics and 'y_scores' in detection_metrics:
            roc_path = self.output_dir / f"{experiment_name}_roc_curve.png"
            self.roc_visualizer.plot_single_roc(
                detection_metrics['y_true'],
                detection_metrics['y_scores'],
                label=f"{experiment_name} ROC",
                save_path=str(roc_path)
            )
            files['roc_curve'] = str(roc_path)
        
        # PR曲线
        if 'y_true' in detection_metrics and 'y_scores' in detection_metrics:
            pr_path = self.output_dir / f"{experiment_name}_pr_curve.png"
            self.pr_visualizer.plot_single_pr(
                detection_metrics['y_true'],
                detection_metrics['y_scores'],
                label=f"{experiment_name} PR",
                save_path=str(pr_path)
            )
            files['pr_curve'] = str(pr_path)
        
        return files
    
    def _visualize_attack_results(self, 
                                attack_results: Dict[str, Any],
                                experiment_name: str) -> Dict[str, str]:
        """可视化攻击结果
        
        Args:
            attack_results: 攻击结果
            experiment_name: 实验名称
            
        Returns:
            Dict[str, str]: 生成的文件路径
        """
        files = {}
        
        # 攻击成功率可视化
        if 'success_rates' in attack_results:
            success_path = self.output_dir / f"{experiment_name}_attack_success.png"
            self._plot_attack_success_rates(
                attack_results['success_rates'], str(success_path)
            )
            files['attack_success'] = str(success_path)
        
        return files
    
    def _visualize_embeddings(self, 
                            embeddings: Dict[str, np.ndarray],
                            experiment_name: str) -> Dict[str, str]:
        """可视化嵌入向量
        
        Args:
            embeddings: 嵌入向量字典
            experiment_name: 实验名称
            
        Returns:
            Dict[str, str]: 生成的文件路径
        """
        files = {}
        
        # t-SNE可视化
        if 'features' in embeddings and 'labels' in embeddings:
            tsne_path = self.output_dir / f"{experiment_name}_tsne.png"
            self.tsne_visualizer.plot_tsne(
                embeddings['features'],
                embeddings['labels'],
                save_path=str(tsne_path)
            )
            files['tsne'] = str(tsne_path)
        
        return files
    
    def _visualize_similarity_distributions(self, 
                                          similarity_scores: Dict[str, np.ndarray],
                                          experiment_name: str) -> Dict[str, str]:
        """可视化相似度分布
        
        Args:
            similarity_scores: 相似度分数
            experiment_name: 实验名称
            
        Returns:
            Dict[str, str]: 生成的文件路径
        """
        files = {}
        
        # 相似度分布图
        if 'clean_scores' in similarity_scores and 'adversarial_scores' in similarity_scores:
            dist_path = self.output_dir / f"{experiment_name}_similarity_dist.png"
            self.similarity_visualizer.plot_similarity_distribution(
                similarity_scores['clean_scores'],
                similarity_scores['adversarial_scores'],
                save_path=str(dist_path)
            )
            files['similarity_distribution'] = str(dist_path)
        
        return files
    
    def _plot_attack_success_rates(self, success_rates: Dict[str, float], save_path: str):
        """绘制攻击成功率
        
        Args:
            success_rates: 攻击成功率字典
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(success_rates.keys())
        rates = list(success_rates.values())
        
        bars = ax.bar(methods, rates, color='skyblue', alpha=0.7)
        ax.set_ylabel('攻击成功率')
        ax.set_title('不同攻击方法的成功率对比')
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_experiment_report(self, 
                                  experiment_results: Dict[str, Any],
                                  experiment_name: str) -> str:
        """生成实验报告
        
        Args:
            experiment_results: 实验结果
            experiment_name: 实验名称
            
        Returns:
            str: 报告文件路径
        """
        report_path = self.output_dir / f"{experiment_name}_report.html"
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>实验报告 - {experiment_name}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .metric {{ margin: 10px 0; }}
                .section {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
            </style>
        </head>
        <body>
            <h1>实验报告: {experiment_name}</h1>
            <div class="section">
                <h2>实验概述</h2>
                <p>实验时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # 添加检测指标
        if 'detection_metrics' in experiment_results:
            metrics = experiment_results['detection_metrics']
            html_content += """
            <div class="section">
                <h2>检测性能指标</h2>
            """
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    html_content += f'<div class="metric">{key}: {value:.4f}</div>'
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def create_comparison_dashboard(self, 
                                  experiments: Dict[str, Dict[str, Any]],
                                  dashboard_name: str = "comparison") -> str:
        """创建实验对比仪表板
        
        Args:
            experiments: 实验结果字典
            dashboard_name: 仪表板名称
            
        Returns:
            str: 仪表板文件路径
        """
        dashboard_path = self.output_dir / f"{dashboard_name}_dashboard.html"
        
        # 使用VisualizationManager创建交互式仪表板
        return self.manager.create_interactive_dashboard(
            experiments, str(dashboard_path)
        )
    
    def save_experiment_config(self, config: Dict[str, Any], experiment_name: str):
        """保存实验配置
        
        Args:
            config: 实验配置
            experiment_name: 实验名称
        """
        config_path = self.output_dir / f"{experiment_name}_config.json"
        
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验配置已保存到: {config_path}")