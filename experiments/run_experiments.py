#!/usr/bin/env python3
"""运行多模态检测一致性实验

这是主要的实验运行脚本，支持不同的实验配置和数据集。
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import json

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigManager
from utils.data_loader import DataLoaderManager
from pipeline import MultiModalDetectionPipeline, create_detection_pipeline
from evaluation import ExperimentEvaluator, ExperimentConfig, create_experiment_evaluator
from attacks import HubnessAttacker, create_hubness_attacker

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行多模态检测一致性实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本配置
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--experiment-name', '-n',
        type=str,
        default='multimodal_detection_experiment',
        help='实验名称'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./results',
        help='输出目录'
    )
    
    # 数据集配置
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['coco', 'flickr30k', 'custom'],
        default='coco',
        help='使用的数据集'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='数据目录'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='使用的样本数量（-1表示使用全部）'
    )
    
    # 攻击配置
    parser.add_argument(
        '--attack-method',
        type=str,
        choices=['hubness', 'none'],
        default='hubness',
        help='攻击方法'
    )
    
    parser.add_argument(
        '--attack-ratio',
        type=float,
        default=0.3,
        help='攻击样本比例'
    )
    
    # 模型配置
    parser.add_argument(
        '--clip-model',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='CLIP模型名称'
    )
    
    parser.add_argument(
        '--qwen-model',
        type=str,
        default='Qwen/Qwen2-1.5B-Instruct',
        help='Qwen模型名称'
    )
    
    parser.add_argument(
        '--sd-model',
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help='Stable Diffusion模型名称'
    )
    
    # 实验配置
    parser.add_argument(
        '--use-cross-validation',
        action='store_true',
        help='使用交叉验证'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='交叉验证折数'
    )
    
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        default=True,
        help='生成可视化图表'
    )
    
    # 设备配置
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='计算设备'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批处理大小'
    )
    
    # 调试选项
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """加载和合并配置"""
    try:
        # 加载配置文件
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        # 使用命令行参数覆盖配置
        if hasattr(args, 'clip_model'):
            config['models']['clip']['model_name'] = args.clip_model
        if hasattr(args, 'qwen_model'):
            config['models']['qwen']['model_name'] = args.qwen_model
        if hasattr(args, 'sd_model'):
            config['models']['stable_diffusion']['model_name'] = args.sd_model
        
        config['experiment']['device'] = args.device
        config['experiment']['batch_size'] = args.batch_size
        
        return config
        
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return {}


def load_dataset(args: argparse.Namespace) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """加载数据集"""
    try:
        logger.info(f"加载数据集: {args.dataset}")
        
        # 创建数据加载器管理器
        data_manager = DataLoaderManager()
        
        if args.dataset == 'coco':
            # 加载COCO数据集
            train_loader = data_manager.create_coco_loader(
                data_dir=args.data_dir,
                split='train',
                batch_size=args.batch_size
            )
            val_loader = data_manager.create_coco_loader(
                data_dir=args.data_dir,
                split='val',
                batch_size=args.batch_size
            )
            test_loader = val_loader  # 使用验证集作为测试集
            
        elif args.dataset == 'flickr30k':
            # 加载Flickr30k数据集
            train_loader = data_manager.create_flickr30k_loader(
                data_dir=args.data_dir,
                split='train',
                batch_size=args.batch_size
            )
            val_loader = data_manager.create_flickr30k_loader(
                data_dir=args.data_dir,
                split='val',
                batch_size=args.batch_size
            )
            test_loader = data_manager.create_flickr30k_loader(
                data_dir=args.data_dir,
                split='test',
                batch_size=args.batch_size
            )
            
        else:
            raise ValueError(f"不支持的数据集: {args.dataset}")
        
        # 转换为列表格式
        def loader_to_list(loader, max_samples=None):
            data_list = []
            count = 0
            for batch in loader:
                for i in range(len(batch['images'])):
                    data_list.append((
                        batch['images'][i],
                        batch['texts'][i],
                        False  # 初始标记为非对抗样本
                    ))
                    count += 1
                    if max_samples and count >= max_samples:
                        break
                if max_samples and count >= max_samples:
                    break
            return data_list
        
        # 限制样本数量
        max_samples = args.num_samples if args.num_samples > 0 else None
        
        train_data = loader_to_list(train_loader, max_samples)
        val_data = loader_to_list(val_loader, max_samples // 2 if max_samples else None)
        test_data = loader_to_list(test_loader, max_samples // 2 if max_samples else None)
        
        logger.info(f"数据集加载完成: 训练={len(train_data)}, 验证={len(val_data)}, 测试={len(test_data)}")
        
        return train_data, val_data, test_data
        
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        return [], [], []


def generate_adversarial_samples(data: List[Tuple], args: argparse.Namespace, config: Dict[str, Any]) -> List[Tuple]:
    """生成对抗样本"""
    try:
        if args.attack_method == 'none':
            return data
        
        logger.info(f"生成对抗样本: {args.attack_method}")
        
        # 计算攻击样本数量
        num_attack_samples = int(len(data) * args.attack_ratio)
        
        if args.attack_method == 'hubness':
            # 创建Hubness攻击器
            attacker = create_hubness_attacker(config.get('attack', {}))
            
            # 选择要攻击的样本
            import random
            random.seed(42)
            attack_indices = random.sample(range(len(data)), num_attack_samples)
            
            # 生成对抗样本
            adversarial_data = []
            for i, (image, text, _) in enumerate(data):
                if i in attack_indices:
                    # 生成对抗样本
                    try:
                        adv_result = attacker.attack_single(image, text)
                        if adv_result['success']:
                            adversarial_data.append((
                                adv_result['adversarial_image'],
                                adv_result['adversarial_text'],
                                True  # 标记为对抗样本
                            ))
                        else:
                            adversarial_data.append((image, text, False))
                    except Exception as e:
                        logger.warning(f"对抗样本生成失败 (索引 {i}): {e}")
                        adversarial_data.append((image, text, False))
                else:
                    adversarial_data.append((image, text, False))
            
            logger.info(f"对抗样本生成完成: {num_attack_samples}/{len(data)} 个样本")
            return adversarial_data
        
        else:
            logger.warning(f"未知的攻击方法: {args.attack_method}")
            return data
            
    except Exception as e:
        logger.error(f"对抗样本生成失败: {e}")
        return data


def run_experiment(args: argparse.Namespace):
    """运行主实验"""
    try:
        logger.info(f"开始实验: {args.experiment_name}")
        start_time = time.time()
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        config = load_config(args.config, args)
        if not config:
            logger.error("配置加载失败，退出实验")
            return
        
        # 保存实验配置
        experiment_config_file = output_dir / "experiment_config.json"
        with open(experiment_config_file, 'w') as f:
            json.dump({
                'args': vars(args),
                'config': config
            }, f, indent=2)
        
        # 加载数据集
        train_data, val_data, test_data = load_dataset(args)
        if not test_data:
            logger.error("数据集加载失败，退出实验")
            return
        
        # 生成对抗样本
        test_data_with_attacks = generate_adversarial_samples(test_data, args, config)
        
        # 创建检测管道
        logger.info("创建检测管道")
        pipeline = create_detection_pipeline(config)
        
        # 创建实验评估器
        experiment_config = ExperimentConfig(
            experiment_name=args.experiment_name,
            use_cross_validation=args.use_cross_validation,
            cv_folds=args.cv_folds,
            generate_plots=args.generate_plots,
            results_dir=str(output_dir)
        )
        
        evaluator = create_experiment_evaluator(experiment_config)
        
        # 运行实验评估
        logger.info("开始实验评估")
        result = evaluator.evaluate_pipeline(pipeline, test_data_with_attacks)
        
        # 生成实验报告
        logger.info("生成实验报告")
        report = evaluator.generate_experiment_report(result)
        
        # 保存报告
        report_file = output_dir / "experiment_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 输出结果摘要
        logger.info("=" * 50)
        logger.info("实验结果摘要:")
        logger.info(f"实验名称: {result.experiment_name}")
        logger.info(f"总样本数: {result.performance_stats.get('total_samples', 'N/A')}")
        
        for metric, value in result.metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        if result.cv_mean:
            logger.info("交叉验证结果:")
            for metric, mean_val in result.cv_mean.items():
                std_val = result.cv_std.get(metric, 0)
                logger.info(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"实验总耗时: {total_time:.2f} 秒")
        logger.info(f"结果保存在: {output_dir}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        raise


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 设置调试模式
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 运行实验
        run_experiment(args)
        
    except KeyboardInterrupt:
        logger.info("实验被用户中断")
    except Exception as e:
        logger.error(f"实验失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()