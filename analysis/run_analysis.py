#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一分析脚本
多模态检测一致性防御方法 - 一键生成论文级别分析

作者: 张昕 (ZHANG XIN)
学校: Duke University
邮箱: zhang.xin@duke.edu
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入分析模块
from generate_comprehensive_report import ComprehensiveReportGenerator
from generate_charts import ChartGenerator
from generate_latex_tables import LaTeXTableGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedAnalysisRunner:
    """
    统一分析运行器
    整合所有分析功能，一键生成论文级别的图表和表格
    """
    
    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.reports_dir = self.output_dir / "reports"
        self.charts_dir = self.output_dir / "charts"
        self.latex_dir = self.output_dir / "latex"
        
        for dir_path in [self.reports_dir, self.charts_dir, self.latex_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"分析输出目录: {self.output_dir}")
        logger.info(f"  - 报告目录: {self.reports_dir}")
        logger.info(f"  - 图表目录: {self.charts_dir}")
        logger.info(f"  - LaTeX目录: {self.latex_dir}")
    
    def check_results_directory(self) -> bool:
        """
        检查实验结果目录是否存在
        """
        if not self.results_dir.exists():
            logger.error(f"实验结果目录不存在: {self.results_dir}")
            return False
        
        # 检查是否有实验数据
        experiment_types = ['defense_effectiveness', 'baseline_comparison', 'ablation_studies', 'efficiency_analysis']
        found_data = False
        
        for exp_type in experiment_types:
            exp_dir = self.results_dir / exp_type
            if exp_dir.exists() and any(exp_dir.iterdir()):
                logger.info(f"发现实验数据: {exp_type}")
                found_data = True
        
        if not found_data:
            logger.warning("未发现任何实验数据，请先运行实验")
            return False
        
        return True
    
    def generate_comprehensive_reports(self) -> bool:
        """
        生成综合报告
        """
        try:
            logger.info("生成综合报告...")
            
            report_generator = ComprehensiveReportGenerator(
                str(self.results_dir),
                str(self.reports_dir)
            )
            
            report_generator.run()
            logger.info("综合报告生成完成")
            return True
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            return False
    
    def generate_charts(self) -> bool:
        """
        生成可视化图表
        """
        try:
            logger.info("生成可视化图表...")
            
            chart_generator = ChartGenerator(
                str(self.results_dir),
                str(self.charts_dir)
            )
            
            chart_generator.run()
            logger.info("可视化图表生成完成")
            return True
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {e}")
            return False
    
    def generate_latex_tables(self) -> bool:
        """
        生成LaTeX表格
        """
        try:
            logger.info("生成LaTeX表格...")
            
            latex_generator = LaTeXTableGenerator(
                str(self.results_dir),
                str(self.latex_dir)
            )
            
            latex_generator.run()
            logger.info("LaTeX表格生成完成")
            return True
            
        except Exception as e:
            logger.error(f"生成LaTeX表格失败: {e}")
            return False
    
    def generate_summary_index(self) -> None:
        """
        生成总结索引文件
        """
        logger.info("生成总结索引...")
        
        index_content = []
        
        # 标题
        index_content.append("# 多模态检测一致性防御方法 - 实验分析结果")
        index_content.append("")
        index_content.append("**作者**: 张昕 (ZHANG XIN)")
        index_content.append("**学校**: Duke University")
        index_content.append("**邮箱**: zhang.xin@duke.edu")
        index_content.append("")
        index_content.append("---")
        index_content.append("")
        
        # 目录结构
        index_content.append("## 📁 分析结果目录结构")
        index_content.append("")
        index_content.append("```")
        index_content.append(f"{self.output_dir.name}/")
        index_content.append("├── reports/          # 综合报告")
        index_content.append("│   ├── comprehensive_report.md")
        index_content.append("│   ├── defense_effectiveness.csv")
        index_content.append("│   ├── baseline_comparison.csv")
        index_content.append("│   ├── ablation_studies.csv")
        index_content.append("│   └── efficiency_analysis.csv")
        index_content.append("├── charts/           # 可视化图表")
        index_content.append("│   ├── defense_effectiveness_comparison.png")
        index_content.append("│   ├── baseline_models_comparison.png")
        index_content.append("│   ├── ablation_study_results.png")
        index_content.append("│   ├── efficiency_analysis.png")
        index_content.append("│   └── comprehensive_dashboard.png")
        index_content.append("├── latex/            # LaTeX表格")
        index_content.append("│   ├── defense_effectiveness_table.tex")
        index_content.append("│   ├── baseline_comparison_table.tex")
        index_content.append("│   ├── ablation_study_table.tex")
        index_content.append("│   ├── efficiency_analysis_table.tex")
        index_content.append("│   ├── complete_tables_document.tex")
        index_content.append("│   └── compile_latex.sh")
        index_content.append("└── README.md         # 本文件")
        index_content.append("```")
        index_content.append("")
        
        # 快速使用指南
        index_content.append("## 🚀 快速使用指南")
        index_content.append("")
        index_content.append("### 1. 查看综合报告")
        index_content.append("```bash")
        index_content.append("# 查看Markdown格式的综合报告")
        index_content.append("cat reports/comprehensive_report.md")
        index_content.append("")
        index_content.append("# 查看CSV格式的数据表格")
        index_content.append("ls reports/*.csv")
        index_content.append("```")
        index_content.append("")
        
        index_content.append("### 2. 查看可视化图表")
        index_content.append("```bash")
        index_content.append("# 查看所有生成的图表")
        index_content.append("ls charts/*.png")
        index_content.append("")
        index_content.append("# 推荐查看综合仪表板")
        index_content.append("open charts/comprehensive_dashboard.png")
        index_content.append("```")
        index_content.append("")
        
        index_content.append("### 3. 编译LaTeX表格")
        index_content.append("```bash")
        index_content.append("cd latex/")
        index_content.append("./compile_latex.sh")
        index_content.append("# 生成的PDF文件: complete_tables_document.pdf")
        index_content.append("```")
        index_content.append("")
        
        # 主要实验结果
        index_content.append("## 📊 主要实验结果")
        index_content.append("")
        index_content.append("### 防御效果对比实验 (主要实验)")
        index_content.append("- **文件**: `reports/defense_effectiveness.csv`")
        index_content.append("- **图表**: `charts/defense_effectiveness_comparison.png`")
        index_content.append("- **LaTeX**: `latex/defense_effectiveness_table.tex`")
        index_content.append("- **说明**: 对比无防御和有防御情况下的攻击成功率")
        index_content.append("")
        
        index_content.append("### 基线模型对比实验")
        index_content.append("- **文件**: `reports/baseline_comparison.csv`")
        index_content.append("- **图表**: `charts/baseline_models_comparison.png`")
        index_content.append("- **LaTeX**: `latex/baseline_comparison_table.tex`")
        index_content.append("- **说明**: 与其他防御方法的性能对比")
        index_content.append("")
        
        index_content.append("### 消融实验")
        index_content.append("- **文件**: `reports/ablation_studies.csv`")
        index_content.append("- **图表**: `charts/ablation_study_results.png`")
        index_content.append("- **LaTeX**: `latex/ablation_study_table.tex`")
        index_content.append("- **说明**: 各个模块对整体性能的贡献分析")
        index_content.append("")
        
        index_content.append("### 效率性能分析")
        index_content.append("- **文件**: `reports/efficiency_analysis.csv`")
        index_content.append("- **图表**: `charts/efficiency_analysis.png`")
        index_content.append("- **LaTeX**: `latex/efficiency_analysis_table.tex`")
        index_content.append("- **说明**: 各个模块的计算开销和内存使用分析")
        index_content.append("")
        
        # 论文写作建议
        index_content.append("## 📝 论文写作建议")
        index_content.append("")
        index_content.append("### 表格使用")
        index_content.append("1. **主要实验表格**: 使用 `defense_effectiveness_table.tex`")
        index_content.append("2. **基线对比表格**: 使用 `baseline_comparison_table.tex`")
        index_content.append("3. **消融实验表格**: 使用 `ablation_study_table.tex`")
        index_content.append("4. **效率分析表格**: 使用 `efficiency_analysis_table.tex`")
        index_content.append("")
        
        index_content.append("### 图表使用")
        index_content.append("1. **防御效果对比**: 展示攻击成功率的下降")
        index_content.append("2. **基线模型对比**: 展示与其他方法的优势")
        index_content.append("3. **消融实验结果**: 展示各模块的重要性")
        index_content.append("4. **效率分析**: 展示计算开销分布")
        index_content.append("")
        
        # 技术说明
        index_content.append("## 🔧 技术说明")
        index_content.append("")
        index_content.append("### 数据格式")
        index_content.append("- **CSV文件**: 可直接导入Excel或其他数据分析工具")
        index_content.append("- **PNG图表**: 高分辨率，适合论文插图")
        index_content.append("- **LaTeX表格**: 符合学术论文格式要求")
        index_content.append("")
        
        index_content.append("### 图表特性")
        index_content.append("- **高分辨率**: 300 DPI，适合印刷")
        index_content.append("- **专业配色**: 使用学术论文标准配色方案")
        index_content.append("- **清晰标注**: 包含详细的轴标签和图例")
        index_content.append("")
        
        index_content.append("### LaTeX编译要求")
        index_content.append("- **必需包**: amsmath, amsfonts, amssymb, graphicx, booktabs")
        index_content.append("- **编译器**: pdflatex (推荐) 或 xelatex")
        index_content.append("- **编码**: UTF-8")
        index_content.append("")
        
        # 联系信息
        index_content.append("## 📧 联系信息")
        index_content.append("")
        index_content.append("如有问题或建议，请联系:")
        index_content.append("- **姓名**: 张昕 (ZHANG XIN)")
        index_content.append("- **邮箱**: zhang.xin@duke.edu")
        index_content.append("- **学校**: Duke University")
        index_content.append("")
        index_content.append("---")
        index_content.append("")
        index_content.append("*本分析报告由自动化脚本生成，确保结果的一致性和可重复性。*")
        
        # 保存索引文件
        index_file = self.output_dir / "README.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(index_content))
        
        logger.info(f"总结索引已保存: {index_file}")
    
    def run(self, skip_reports: bool = False, skip_charts: bool = False, skip_latex: bool = False) -> bool:
        """
        运行完整的分析流程
        
        Args:
            skip_reports: 跳过报告生成
            skip_charts: 跳过图表生成
            skip_latex: 跳过LaTeX表格生成
        
        Returns:
            bool: 是否成功完成所有分析
        """
        logger.info("开始统一分析流程...")
        
        # 检查实验结果目录
        if not self.check_results_directory():
            return False
        
        success_count = 0
        total_tasks = 3
        
        # 生成综合报告
        if not skip_reports:
            if self.generate_comprehensive_reports():
                success_count += 1
            else:
                logger.warning("综合报告生成失败，但继续其他任务")
        else:
            logger.info("跳过综合报告生成")
            total_tasks -= 1
        
        # 生成可视化图表
        if not skip_charts:
            if self.generate_charts():
                success_count += 1
            else:
                logger.warning("可视化图表生成失败，但继续其他任务")
        else:
            logger.info("跳过可视化图表生成")
            total_tasks -= 1
        
        # 生成LaTeX表格
        if not skip_latex:
            if self.generate_latex_tables():
                success_count += 1
            else:
                logger.warning("LaTeX表格生成失败，但继续其他任务")
        else:
            logger.info("跳过LaTeX表格生成")
            total_tasks -= 1
        
        # 生成总结索引
        self.generate_summary_index()
        
        # 输出结果
        logger.info(f"分析完成: {success_count}/{total_tasks} 个任务成功")
        logger.info(f"分析结果保存在: {self.output_dir}")
        
        if success_count == total_tasks:
            logger.info("🎉 所有分析任务成功完成!")
            return True
        else:
            logger.warning(f"⚠️  部分任务失败，请检查日志")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="统一分析脚本 - 生成论文级别的图表和表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整分析
  python run_analysis.py --results_dir ./results --output_dir ./analysis_output
  
  # 只生成图表
  python run_analysis.py --results_dir ./results --output_dir ./analysis_output --skip-reports --skip-latex
  
  # 详细输出
  python run_analysis.py --results_dir ./results --output_dir ./analysis_output --verbose
        """
    )
    
    parser.add_argument("--results_dir", type=str, required=True, help="实验结果目录")
    parser.add_argument("--output_dir", type=str, required=True, help="分析输出目录")
    parser.add_argument("--skip-reports", action="store_true", help="跳过综合报告生成")
    parser.add_argument("--skip-charts", action="store_true", help="跳过可视化图表生成")
    parser.add_argument("--skip-latex", action="store_true", help="跳过LaTeX表格生成")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建统一分析运行器并运行
    runner = UnifiedAnalysisRunner(args.results_dir, args.output_dir)
    success = runner.run(
        skip_reports=args.skip_reports,
        skip_charts=args.skip_charts,
        skip_latex=args.skip_latex
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()