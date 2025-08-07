#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化图表生成器
多模态检测一致性防御方法 - 论文级别图表生成

作者: 张昕 (ZHANG XIN)
学校: Duke University
邮箱: zhang.xin@duke.edu
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    论文级别图表生成器
    生成高质量的实验结果可视化图表
    """
    
    def __init__(self, results_dir: str, output_dir: str, formats: List[str] = None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats or ['png', 'pdf', 'svg']
        
        # 设置图表样式
        self.figure_size = (12, 8)
        self.dpi = 300
        self.font_size = 12
        
        # 颜色方案
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F4B942',
            'info': '#7209B7'
        }
        
        # 数据集和攻击方法映射
        self.dataset_names = {
            "coco": "MS COCO",
            "flickr": "Flickr30K",
            "flickr30k": "Flickr30K",
            "cc3m": "CC3M",
            "vg": "Visual Genome"
        }
        
        self.attack_names = {
            "pgd": "PGD",
            "hubness": "Hubness",
            "fsta": "FSTA",
            "sma": "SMA"
        }
        
        # 加载数据
        self.data = self._load_all_data()
    
    def _load_all_data(self) -> Dict:
        """
        加载所有实验数据
        """
        logger.info("加载实验数据...")
        
        data = {
            'defense_effectiveness': defaultdict(dict),
            'baseline_comparison': defaultdict(dict),
            'ablation_studies': defaultdict(dict),
            'efficiency_analysis': defaultdict(dict)
        }
        
        # 加载防御效果数据
        defense_dir = self.results_dir / "defense_effectiveness"
        if defense_dir.exists():
            for result_file in defense_dir.glob("*/results.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    parts = result_file.parent.name.split('_')
                    if len(parts) >= 3:
                        dataset = parts[0]
                        attack = parts[1]
                        defense_type = '_'.join(parts[2:])
                        
                        key = f"{dataset}_{attack}"
                        data['defense_effectiveness'][key][defense_type] = result_data
                except Exception as e:
                    logger.error(f"加载文件失败 {result_file}: {e}")
        
        # 加载其他数据...
        self._load_baseline_data(data)
        self._load_ablation_data(data)
        self._load_efficiency_data(data)
        
        logger.info("数据加载完成")
        return data
    
    def _load_baseline_data(self, data: Dict) -> None:
        """加载基线对比数据"""
        baseline_dir = self.results_dir / "baseline_comparison"
        if baseline_dir.exists():
            for result_file in baseline_dir.glob("*/results.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    parts = result_file.parent.name.split('_')
                    if len(parts) >= 3:
                        dataset = parts[0]
                        attack = parts[1]
                        baseline = '_'.join(parts[2:])
                        
                        key = f"{dataset}_{attack}"
                        data['baseline_comparison'][key][baseline] = result_data
                except Exception as e:
                    logger.error(f"加载基线数据失败 {result_file}: {e}")
    
    def _load_ablation_data(self, data: Dict) -> None:
        """加载消融实验数据"""
        ablation_dir = self.results_dir / "ablation_studies"
        if ablation_dir.exists():
            for result_file in ablation_dir.glob("*/results.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    parts = result_file.parent.name.split('_')
                    if len(parts) >= 3:
                        dataset = parts[0]
                        attack = parts[1]
                        ablation = '_'.join(parts[2:])
                        
                        key = f"{dataset}_{attack}"
                        data['ablation_studies'][key][ablation] = result_data
                except Exception as e:
                    logger.error(f"加载消融数据失败 {result_file}: {e}")
    
    def _load_efficiency_data(self, data: Dict) -> None:
        """加载效率分析数据"""
        efficiency_dir = self.results_dir / "efficiency_analysis"
        if efficiency_dir.exists():
            for result_file in efficiency_dir.glob("*/results.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    module_name = result_file.parent.name.replace('_efficiency', '')
                    data['efficiency_analysis'][module_name] = result_data
                except Exception as e:
                    logger.error(f"加载效率数据失败 {result_file}: {e}")
    
    def _save_figure(self, fig, filename: str) -> None:
        """
        保存图表到多种格式
        """
        for fmt in self.formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, format=fmt, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"图表已保存: {filepath}")
    
    def generate_defense_effectiveness_chart(self) -> None:
        """
        生成防御效果对比图表
        """
        logger.info("生成防御效果对比图表...")
        
        # 准备数据
        datasets = []
        attacks = []
        asr_no_defense = []
        asr_with_defense = []
        defense_success_rates = []
        
        for key, defenses in self.data['defense_effectiveness'].items():
            parts = key.split('_')
            dataset = self.dataset_names.get(parts[0], parts[0])
            attack = self.attack_names.get(parts[1], parts[1])
            
            no_defense = defenses.get('no_defense', {})
            full_defense = defenses.get('full_defense', {})
            
            asr_baseline = no_defense.get('attack_success_rate_baseline', 0.0) * 100
            asr_defended = full_defense.get('attack_success_rate_defended', 0.0) * 100
            defense_rate = (asr_baseline - asr_defended) / asr_baseline * 100 if asr_baseline > 0 else 0.0
            
            datasets.append(dataset)
            attacks.append(attack)
            asr_no_defense.append(asr_baseline)
            asr_with_defense.append(asr_defended)
            defense_success_rates.append(defense_rate)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 攻击成功率对比 (柱状图)
        x = np.arange(len(datasets))
        width = 0.35
        
        ax1.bar(x - width/2, asr_no_defense, width, label='无防御', color=self.colors['primary'], alpha=0.8)
        ax1.bar(x + width/2, asr_with_defense, width, label='有防御', color=self.colors['secondary'], alpha=0.8)
        
        ax1.set_xlabel('数据集-攻击方法组合')
        ax1.set_ylabel('攻击成功率 (%)')
        ax1.set_title('防御前后攻击成功率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{d}\n{a}" for d, a in zip(datasets, attacks)], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 防御成功率 (柱状图)
        bars = ax2.bar(x, defense_success_rates, color=self.colors['accent'], alpha=0.8)
        ax2.set_xlabel('数据集-攻击方法组合')
        ax2.set_ylabel('防御成功率 (%)')
        ax2.set_title('防御成功率')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{d}\n{a}" for d, a in zip(datasets, attacks)], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, rate in zip(bars, defense_success_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. 数据集维度分析 (箱线图)
        dataset_groups = defaultdict(list)
        for d, rate in zip(datasets, defense_success_rates):
            dataset_groups[d].append(rate)
        
        dataset_names = list(dataset_groups.keys())
        dataset_values = [dataset_groups[name] for name in dataset_names]
        
        bp = ax3.boxplot(dataset_values, labels=dataset_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(dataset_names))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('数据集')
        ax3.set_ylabel('防御成功率 (%)')
        ax3.set_title('不同数据集的防御效果分布')
        ax3.grid(True, alpha=0.3)
        
        # 4. 攻击方法维度分析 (箱线图)
        attack_groups = defaultdict(list)
        for a, rate in zip(attacks, defense_success_rates):
            attack_groups[a].append(rate)
        
        attack_names = list(attack_groups.keys())
        attack_values = [attack_groups[name] for name in attack_names]
        
        bp = ax4.boxplot(attack_values, labels=attack_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(attack_names))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_xlabel('攻击方法')
        ax4.set_ylabel('防御成功率 (%)')
        ax4.set_title('不同攻击方法的防御效果分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'defense_effectiveness_comprehensive')
        plt.close()
    
    def generate_baseline_comparison_chart(self) -> None:
        """
        生成基线模型对比图表
        """
        logger.info("生成基线模型对比图表...")
        
        # 收集基线方法的性能数据
        baseline_stats = defaultdict(list)
        
        for key, baselines in self.data['baseline_comparison'].items():
            for baseline, metrics in baselines.items():
                defense_rate = metrics.get('defense_success_rate', 0.0) * 100
                baseline_stats[baseline].append(defense_rate)
        
        # 计算平均值和标准差
        baseline_names = []
        mean_rates = []
        std_rates = []
        
        baseline_display_names = {
            'no_defense': '无防御',
            'unimodal_anomaly': '单模态异常检测',
            'random_variants': '随机文本变体',
            'retrieval_only': '仅检索参考',
            'generative_only': '仅生成参考',
            'full_defense': '完整防御方法'
        }
        
        for baseline, rates in baseline_stats.items():
            if rates:
                baseline_names.append(baseline_display_names.get(baseline, baseline))
                mean_rates.append(np.mean(rates))
                std_rates.append(np.std(rates))
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 基线方法性能对比 (柱状图)
        x = np.arange(len(baseline_names))
        bars = ax1.bar(x, mean_rates, yerr=std_rates, capsize=5, 
                      color=sns.color_palette("viridis", len(baseline_names)), alpha=0.8)
        
        ax1.set_xlabel('基线方法')
        ax1.set_ylabel('平均防御成功率 (%)')
        ax1.set_title('基线方法性能对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(baseline_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, mean_val, std_val in zip(bars, mean_rates, std_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 1,
                    f'{mean_val:.1f}±{std_val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 2. 性能提升分析 (雷达图)
        if len(baseline_names) >= 3:
            angles = np.linspace(0, 2 * np.pi, len(baseline_names), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            values = mean_rates + [mean_rates[0]]  # 闭合数据
            
            ax2 = plt.subplot(122, projection='polar')
            ax2.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
            ax2.fill(angles, values, alpha=0.25, color=self.colors['primary'])
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(baseline_names)
            ax2.set_ylim(0, max(mean_rates) * 1.1)
            ax2.set_title('基线方法性能雷达图', pad=20)
            ax2.grid(True)
        
        plt.tight_layout()
        self._save_figure(fig, 'baseline_comparison')
        plt.close()
    
    def generate_ablation_study_chart(self) -> None:
        """
        生成消融实验图表
        """
        logger.info("生成消融实验图表...")
        
        # 收集消融实验数据
        ablation_stats = defaultdict(list)
        
        for key, ablations in self.data['ablation_studies'].items():
            for ablation, metrics in ablations.items():
                defense_rate = metrics.get('defense_success_rate', 0.0) * 100
                processing_time = metrics.get('avg_processing_time', 0.0)
                gpu_memory = metrics.get('gpu_memory_usage', 0.0)
                
                ablation_stats[ablation].append({
                    'defense_rate': defense_rate,
                    'processing_time': processing_time,
                    'gpu_memory': gpu_memory
                })
        
        # 准备数据
        ablation_names = []
        mean_defense_rates = []
        mean_processing_times = []
        mean_gpu_memory = []
        
        ablation_display_names = {
            'full_defense': '完整防御方法',
            'no_text_variants': '无文本变体增强',
            'no_genref': '无SD生成参考',
            'no_retrieval': '无检索参考',
            'consistency_only': '仅一致性度量',
            'fixed_threshold': '固定阈值'
        }
        
        for ablation, metrics_list in ablation_stats.items():
            if metrics_list:
                ablation_names.append(ablation_display_names.get(ablation, ablation))
                mean_defense_rates.append(np.mean([m['defense_rate'] for m in metrics_list]))
                mean_processing_times.append(np.mean([m['processing_time'] for m in metrics_list]))
                mean_gpu_memory.append(np.mean([m['gpu_memory'] for m in metrics_list]))
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 防御成功率对比
        x = np.arange(len(ablation_names))
        bars = ax1.bar(x, mean_defense_rates, color=sns.color_palette("plasma", len(ablation_names)), alpha=0.8)
        
        ax1.set_xlabel('模型变体')
        ax1.set_ylabel('防御成功率 (%)')
        ax1.set_title('消融实验 - 防御成功率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(ablation_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, mean_defense_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 2. 处理时间对比
        bars = ax2.bar(x, mean_processing_times, color=sns.color_palette("viridis", len(ablation_names)), alpha=0.8)
        
        ax2.set_xlabel('模型变体')
        ax2.set_ylabel('平均处理时间 (ms)')
        ax2.set_title('消融实验 - 处理时间对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(ablation_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. GPU内存使用对比
        bars = ax3.bar(x, mean_gpu_memory, color=sns.color_palette("coolwarm", len(ablation_names)), alpha=0.8)
        
        ax3.set_xlabel('模型变体')
        ax3.set_ylabel('GPU内存使用 (GB)')
        ax3.set_title('消融实验 - GPU内存使用对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(ablation_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. 效率-效果权衡图
        ax4.scatter(mean_processing_times, mean_defense_rates, 
                   s=np.array(mean_gpu_memory) * 50, alpha=0.6,
                   c=range(len(ablation_names)), cmap='tab10')
        
        for i, name in enumerate(ablation_names):
            ax4.annotate(name, (mean_processing_times[i], mean_defense_rates[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('平均处理时间 (ms)')
        ax4.set_ylabel('防御成功率 (%)')
        ax4.set_title('效率-效果权衡分析\n(气泡大小表示GPU内存使用)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'ablation_study_comprehensive')
        plt.close()
    
    def generate_efficiency_analysis_chart(self) -> None:
        """
        生成效率分析图表
        """
        logger.info("生成效率分析图表...")
        
        # 准备数据
        modules = []
        processing_times = []
        gpu_memory_usage = []
        
        module_display_names = {
            'text_augment': '文本增强模块',
            'retrieval': '检索参考模块',
            'sd_generation': 'SD生成模块',
            'consistency_detection': '一致性检测模块',
            'full_pipeline': '完整流水线'
        }
        
        for module, metrics in self.data['efficiency_analysis'].items():
            modules.append(module_display_names.get(module, module))
            processing_times.append(metrics.get('avg_processing_time_ms', 0.0))
            gpu_memory_usage.append(metrics.get('gpu_memory_gb', 0.0))
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 处理时间分布 (柱状图)
        x = np.arange(len(modules))
        bars = ax1.bar(x, processing_times, color=sns.color_palette("rocket", len(modules)), alpha=0.8)
        
        ax1.set_xlabel('模块')
        ax1.set_ylabel('处理时间 (ms/query)')
        ax1.set_title('各模块处理时间对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modules, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, time in zip(bars, processing_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(processing_times) * 0.01,
                    f'{time:.1f}ms', ha='center', va='bottom')
        
        # 2. GPU内存使用 (柱状图)
        bars = ax2.bar(x, gpu_memory_usage, color=sns.color_palette("mako", len(modules)), alpha=0.8)
        
        ax2.set_xlabel('模块')
        ax2.set_ylabel('GPU内存使用 (GB/query)')
        ax2.set_title('各模块GPU内存使用对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(modules, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. 处理时间占比 (饼图)
        if sum(processing_times) > 0:
            ax3.pie(processing_times, labels=modules, autopct='%1.1f%%', startangle=90)
            ax3.set_title('各模块处理时间占比')
        
        # 4. 内存使用占比 (饼图)
        if sum(gpu_memory_usage) > 0:
            ax4.pie(gpu_memory_usage, labels=modules, autopct='%1.1f%%', startangle=90)
            ax4.set_title('各模块GPU内存使用占比')
        
        plt.tight_layout()
        self._save_figure(fig, 'efficiency_analysis_comprehensive')
        plt.close()
    
    def generate_summary_dashboard(self) -> None:
        """
        生成综合仪表板
        """
        logger.info("生成综合仪表板...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 主要指标概览
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.text(0.5, 0.8, '多模态检测一致性防御方法 - 实验结果仪表板', 
                    ha='center', va='center', fontsize=24, fontweight='bold')
        ax_main.text(0.5, 0.5, f'作者: 张昕 (ZHANG XIN) | Duke University', 
                    ha='center', va='center', fontsize=16)
        ax_main.text(0.5, 0.2, f'生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                    ha='center', va='center', fontsize=14)
        ax_main.axis('off')
        
        # 添加更多子图...
        # 这里可以添加关键指标的小图表
        
        self._save_figure(fig, 'comprehensive_dashboard')
        plt.close()
    
    def run(self) -> None:
        """
        运行完整的图表生成流程
        """
        logger.info("开始生成可视化图表...")
        
        # 生成各种图表
        self.generate_defense_effectiveness_chart()
        self.generate_baseline_comparison_chart()
        self.generate_ablation_study_chart()
        self.generate_efficiency_analysis_chart()
        self.generate_summary_dashboard()
        
        logger.info(f"可视化图表生成完成，输出目录: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="生成可视化图表")
    parser.add_argument("--results_dir", type=str, required=True, help="实验结果目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--format", type=str, default="png,pdf,svg", help="输出格式，逗号分隔")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    formats = args.format.split(',')
    
    # 创建图表生成器并运行
    generator = ChartGenerator(args.results_dir, args.output_dir, formats)
    generator.run()

if __name__ == "__main__":
    main()