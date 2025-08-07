#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX表格生成器
多模态检测一致性防御方法 - 论文级别LaTeX表格生成

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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LaTeXTableGenerator:
    """
    LaTeX表格生成器
    生成论文级别的LaTeX表格
    """
    
    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        self.defense_names = {
            "no_defense": "无防御",
            "full_defense": "完整防御方法",
            "unimodal_anomaly": "单模态异常检测",
            "random_variants": "随机文本变体",
            "retrieval_only": "仅检索参考",
            "generative_only": "仅生成参考",
            "no_text_variants": "无文本变体增强",
            "no_genref": "无Stable Diffusion生成参考",
            "no_retrieval": "无检索参考",
            "consistency_only": "仅一致性度量模块",
            "fixed_threshold": "固定阈值"
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
        
        # 加载其他数据
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
    
    def generate_defense_effectiveness_latex(self) -> str:
        """
        生成防御效果对比LaTeX表格
        """
        logger.info("生成防御效果对比LaTeX表格...")
        
        latex_content = []
        
        # 表格开始
        latex_content.append("% 防御效果对比表格 (主要实验)")
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{防御效果对比实验结果}")
        latex_content.append("\\label{tab:defense_effectiveness}")
        latex_content.append("\\begin{tabular}{|l|l|c|c|c|c|}")
        latex_content.append("\\hline")
        latex_content.append("数据集 & 攻击方法 & 无防御ASR (\\%) & 有防御ASR (\\%) & 防御成功率(\\%) $\\uparrow$ & Top-k 检索精度(\\%) \\\\")
        latex_content.append("\\hline")
        
        # 数据行
        for key, defenses in self.data['defense_effectiveness'].items():
            parts = key.split('_')
            dataset = self.dataset_names.get(parts[0], parts[0])
            attack = self.attack_names.get(parts[1], parts[1])
            
            no_defense = defenses.get('no_defense', {})
            full_defense = defenses.get('full_defense', {})
            
            asr_no_defense = no_defense.get('attack_success_rate_baseline', 0.0) * 100
            asr_with_defense = full_defense.get('attack_success_rate_defended', 0.0) * 100
            defense_success_rate = (asr_no_defense - asr_with_defense) / asr_no_defense * 100 if asr_no_defense > 0 else 0.0
            top_k_accuracy = full_defense.get('retrieval_accuracy', 0.0) * 100
            
            latex_content.append(
                f"{dataset} & {attack} & {asr_no_defense:.1f} & {asr_with_defense:.1f} & "
                f"\\textbf{{{defense_success_rate:.1f}}} & {top_k_accuracy:.1f} \\\\"
            )
            latex_content.append("\\hline")
        
        # 表格结束
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        return "\n".join(latex_content)
    
    def generate_baseline_comparison_latex(self) -> str:
        """
        生成基线模型对比LaTeX表格
        """
        logger.info("生成基线模型对比LaTeX表格...")
        
        # 计算基线方法的平均性能
        baseline_stats = defaultdict(list)
        
        for key, baselines in self.data['baseline_comparison'].items():
            for baseline, metrics in baselines.items():
                defense_rate = metrics.get('defense_success_rate', 0.0) * 100
                baseline_stats[baseline].append(defense_rate)
        
        latex_content = []
        
        # 表格开始
        latex_content.append("% 基线模型对比表格")
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{基线模型对比实验结果}")
        latex_content.append("\\label{tab:baseline_comparison}")
        latex_content.append("\\begin{tabular}{|l|c|c|}")
        latex_content.append("\\hline")
        latex_content.append("基线方法 & 平均防御成功率(\\%) & 标准差 \\\\")
        latex_content.append("\\hline")
        
        # 数据行
        for baseline, success_rates in baseline_stats.items():
            if success_rates:
                avg_success_rate = np.mean(success_rates)
                std_success_rate = np.std(success_rates)
                baseline_name = self.defense_names.get(baseline, baseline)
                
                # 突出显示最佳结果
                if avg_success_rate == max([np.mean(rates) for rates in baseline_stats.values() if rates]):
                    latex_content.append(
                        f"\\textbf{{{baseline_name}}} & \\textbf{{{avg_success_rate:.1f}}} & {std_success_rate:.1f} \\\\"
                    )
                else:
                    latex_content.append(
                        f"{baseline_name} & {avg_success_rate:.1f} & {std_success_rate:.1f} \\\\"
                    )
                latex_content.append("\\hline")
        
        # 表格结束
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        return "\n".join(latex_content)
    
    def generate_ablation_study_latex(self) -> str:
        """
        生成消融实验LaTeX表格
        """
        logger.info("生成消融实验LaTeX表格...")
        
        # 计算消融实验的平均性能
        ablation_stats = defaultdict(list)
        
        for key, ablations in self.data['ablation_studies'].items():
            for ablation, metrics in ablations.items():
                ablation_stats[ablation].append({
                    'defense_success_rate': metrics.get('defense_success_rate', 0.0) * 100,
                    'top_k_accuracy': metrics.get('retrieval_accuracy', 0.0) * 100,
                    'processing_time': metrics.get('avg_processing_time', 0.0),
                    'gpu_memory': metrics.get('gpu_memory_usage', 0.0)
                })
        
        latex_content = []
        
        # 表格开始
        latex_content.append("% 消融实验表格")
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{消融实验结果}")
        latex_content.append("\\label{tab:ablation_study}")
        latex_content.append("\\begin{tabular}{|l|c|c|c|c|}")
        latex_content.append("\\hline")
        latex_content.append("模型变体 & 防御成功率 (\\%) & Top-k 检索精度(\\%) & GPU显存占用 (GB) & 处理速度 (query/s) \\\\")
        latex_content.append("\\hline")
        
        # 数据行
        for ablation, metrics_list in ablation_stats.items():
            if metrics_list:
                avg_defense = np.mean([m['defense_success_rate'] for m in metrics_list])
                avg_accuracy = np.mean([m['top_k_accuracy'] for m in metrics_list])
                avg_time = np.mean([m['processing_time'] for m in metrics_list])
                avg_memory = np.mean([m['gpu_memory'] for m in metrics_list])
                
                ablation_name = self.defense_names.get(ablation, ablation)
                throughput = 1000 / avg_time if avg_time > 0 else 0
                
                # 突出显示完整防御方法
                if ablation == 'full_defense':
                    latex_content.append(
                        f"\\textbf{{{ablation_name}}} & \\textbf{{{avg_defense:.1f}}} & "
                        f"\\textbf{{{avg_accuracy:.1f}}} & \\textbf{{{avg_memory:.1f}}} & "
                        f"\\textbf{{{throughput:.1f}}} \\\\"
                    )
                else:
                    latex_content.append(
                        f"{ablation_name} & {avg_defense:.1f} & {avg_accuracy:.1f} & "
                        f"{avg_memory:.1f} & {throughput:.1f} \\\\"
                    )
                latex_content.append("\\hline")
        
        # 表格结束
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        return "\n".join(latex_content)
    
    def generate_efficiency_analysis_latex(self) -> str:
        """
        生成效率性能分析LaTeX表格
        """
        logger.info("生成效率性能分析LaTeX表格...")
        
        module_descriptions = {
            'text_augment': '文本增强模块',
            'retrieval': '检索参考模块',
            'sd_generation': 'Stable Diffusion生成模块',
            'consistency_detection': '一致性检测模块',
            'full_pipeline': '总处理时间'
        }
        
        latex_content = []
        
        # 表格开始
        latex_content.append("% 效率性能分析表格")
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{效率性能分析结果}")
        latex_content.append("\\label{tab:efficiency_analysis}")
        latex_content.append("\\begin{tabular}{|l|c|c|l|}")
        latex_content.append("\\hline")
        latex_content.append("模块 & 耗时(ms/query) $\\downarrow$ & GPU内存(GB/query) $\\downarrow$ & 说明 \\\\")
        latex_content.append("\\hline")
        
        # 数据行
        for module, metrics in self.data['efficiency_analysis'].items():
            module_name = module_descriptions.get(module, module)
            avg_time = metrics.get('avg_processing_time_ms', 0.0)
            gpu_memory = metrics.get('gpu_memory_gb', 0.0)
            description = self._get_module_description(module)
            
            # 突出显示瓶颈模块
            if module == 'sd_generation' and avg_time > 1000:  # 如果SD模块耗时超过1秒
                latex_content.append(
                    f"\\textbf{{{module_name}}} & \\textbf{{{avg_time:.1f}}} & "
                    f"\\textbf{{{gpu_memory:.2f}}} & {description} \\\\"
                )
            else:
                latex_content.append(
                    f"{module_name} & {avg_time:.1f} & {gpu_memory:.2f} & {description} \\\\"
                )
            latex_content.append("\\hline")
        
        # 表格结束
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        return "\n".join(latex_content)
    
    def _get_module_description(self, module: str) -> str:
        """
        获取模块描述
        """
        descriptions = {
            'text_augment': '文本变体生成',
            'retrieval': 'CLIP检索及FAISS索引',
            'sd_generation': '图像合成及CLIP编码',
            'consistency_detection': '计算相似度与投票决策',
            'full_pipeline': '端到端处理'
        }
        return descriptions.get(module, '未知模块')
    
    def generate_comprehensive_latex_document(self) -> str:
        """
        生成完整的LaTeX文档
        """
        logger.info("生成完整的LaTeX文档...")
        
        latex_content = []
        
        # 文档头部
        latex_content.append("% 多模态检测一致性防御方法 - 实验结果表格")
        latex_content.append("% 作者: 张昕 (ZHANG XIN)")
        latex_content.append("% 学校: Duke University")
        latex_content.append("% 邮箱: zhang.xin@duke.edu")
        latex_content.append("")
        latex_content.append("\\documentclass[12pt]{article}")
        latex_content.append("\\usepackage[utf8]{inputenc}")
        latex_content.append("\\usepackage[T1]{fontenc}")
        latex_content.append("\\usepackage{amsmath}")
        latex_content.append("\\usepackage{amsfonts}")
        latex_content.append("\\usepackage{amssymb}")
        latex_content.append("\\usepackage{graphicx}")
        latex_content.append("\\usepackage{booktabs}")
        latex_content.append("\\usepackage{array}")
        latex_content.append("\\usepackage{multirow}")
        latex_content.append("\\usepackage{xcolor}")
        latex_content.append("\\usepackage{geometry}")
        latex_content.append("\\geometry{a4paper, margin=1in}")
        latex_content.append("")
        latex_content.append("\\title{多模态检测一致性防御方法 - 实验结果}")
        latex_content.append("\\author{张昕 (ZHANG XIN) \\\\ Duke University \\\\ zhang.xin@duke.edu}")
        latex_content.append("\\date{\\today}")
        latex_content.append("")
        latex_content.append("\\begin{document}")
        latex_content.append("")
        latex_content.append("\\maketitle")
        latex_content.append("")
        
        # 添加各个表格
        latex_content.append("\\section{防御效果对比实验 (主要实验)}")
        latex_content.append("")
        latex_content.append(self.generate_defense_effectiveness_latex())
        latex_content.append("")
        
        latex_content.append("\\section{基线模型对比实验}")
        latex_content.append("")
        latex_content.append(self.generate_baseline_comparison_latex())
        latex_content.append("")
        
        latex_content.append("\\section{消融实验}")
        latex_content.append("")
        latex_content.append(self.generate_ablation_study_latex())
        latex_content.append("")
        
        latex_content.append("\\section{效率性能分析}")
        latex_content.append("")
        latex_content.append(self.generate_efficiency_analysis_latex())
        latex_content.append("")
        
        # 文档结尾
        latex_content.append("\\end{document}")
        
        return "\n".join(latex_content)
    
    def save_latex_tables(self) -> None:
        """
        保存所有LaTeX表格
        """
        logger.info("保存LaTeX表格...")
        
        # 保存单独的表格
        tables = {
            'defense_effectiveness': self.generate_defense_effectiveness_latex(),
            'baseline_comparison': self.generate_baseline_comparison_latex(),
            'ablation_study': self.generate_ablation_study_latex(),
            'efficiency_analysis': self.generate_efficiency_analysis_latex()
        }
        
        for table_name, latex_content in tables.items():
            output_file = self.output_dir / f"{table_name}_table.tex"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            logger.info(f"LaTeX表格已保存: {output_file}")
        
        # 保存完整文档
        complete_doc = self.generate_comprehensive_latex_document()
        doc_file = self.output_dir / "complete_tables_document.tex"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(complete_doc)
        logger.info(f"完整LaTeX文档已保存: {doc_file}")
        
        # 生成编译脚本
        compile_script = self.output_dir / "compile_latex.sh"
        with open(compile_script, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n")
            f.write("# LaTeX编译脚本\n")
            f.write("# 使用方法: ./compile_latex.sh\n\n")
            f.write("echo \"编译LaTeX文档...\"\n")
            f.write("pdflatex complete_tables_document.tex\n")
            f.write("pdflatex complete_tables_document.tex  # 二次编译确保引用正确\n")
            f.write("echo \"编译完成: complete_tables_document.pdf\"\n")
        
        compile_script.chmod(0o755)  # 设置可执行权限
        logger.info(f"LaTeX编译脚本已保存: {compile_script}")
    
    def run(self) -> None:
        """
        运行完整的LaTeX表格生成流程
        """
        logger.info("开始生成LaTeX表格...")
        
        self.save_latex_tables()
        
        logger.info(f"LaTeX表格生成完成，输出目录: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="生成LaTeX表格")
    parser.add_argument("--results_dir", type=str, required=True, help="实验结果目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建LaTeX表格生成器并运行
    generator = LaTeXTableGenerator(args.results_dir, args.output_dir)
    generator.run()

if __name__ == "__main__":
    main()