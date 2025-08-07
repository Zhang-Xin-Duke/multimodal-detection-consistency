#!/usr/bin/env python3
"""
多模态检测系统统一部署脚本
集成硬件检测、动态配置生成和服务部署功能
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

from src.utils.hardware_detector import HardwareDetector
from src.utils.dynamic_config import DynamicConfigManager
from quick_start import QuickStartManager
from auto_deploy import AutoDeployManager


class UnifiedDeployManager:
    """统一部署管理器"""
    
    def __init__(self):
        self.hardware_detector = HardwareDetector()
        self.config_manager = DynamicConfigManager()
        self.quick_start = QuickStartManager()
        self.auto_deploy = AutoDeployManager()
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def detect_hardware(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """检测硬件配置"""
        print("\n🔍 正在检测硬件配置...")
        
        hardware_info = self.hardware_detector.detect_hardware()
        self.hardware_detector.print_hardware_summary(hardware_info)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(hardware_info, f, indent=2, ensure_ascii=False)
            print(f"✅ 硬件信息已保存到: {save_path}")
        
        return hardware_info
    
    def generate_config(self, hardware_info: Dict[str, Any], 
                       output_path: Optional[str] = None,
                       profile: Optional[str] = None) -> Dict[str, Any]:
        """生成动态配置"""
        print("\n⚙️  正在生成动态配置...")
        
        config = self.config_manager.generate_config(hardware_info, force_profile=profile)
        self.config_manager.print_config_summary(config)
        
        if not output_path:
            output_path = "dynamic/unified_config.yaml"
        
        # 确保目录存在
        full_path = output_path if os.path.isabs(output_path) else os.path.join("configs", output_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        self.config_manager.save_config(config, output_path)
        print(f"✅ 配置已保存到: {output_path}")
        
        return config
    
    def deploy_system(self, config_path: str, mode: str = "start") -> bool:
        """部署系统"""
        print("\n🚀 正在部署系统...")
        
        try:
            if mode == "quick":
                success = self.quick_start.start_service(config_path=config_path)
                if not success:
                    raise Exception("Quick start service failed")
            elif mode == "auto":
                # AutoDeployManager的启动逻辑
                if not self.auto_deploy.initialize_services():
                    raise Exception("Auto deploy initialization failed")
                if not self.auto_deploy.start_services():
                    raise Exception("Auto deploy services failed")
            else:
                success = self.quick_start.start_service(config_path=config_path)
                if not success:
                    raise Exception("Quick start service failed")
            
            print("✅ 系统部署成功！")
            return True
            
        except Exception as e:
            print(f"❌ 系统部署失败: {e}")
            self.logger.error(f"部署失败: {e}")
            return False
    
    def run_full_deployment(self, 
                           profile: Optional[str] = None,
                           config_output: Optional[str] = None,
                           hardware_output: Optional[str] = None,
                           deploy_mode: str = "quick") -> bool:
        """完整部署流程"""
        print("\n🎯 开始完整部署流程...")
        
        try:
            # 1. 硬件检测
            hardware_info = self.detect_hardware(hardware_output)
            
            # 2. 生成配置
            config = self.generate_config(hardware_info, config_output, profile)
            
            # 3. 部署系统
            config_path = config_output or "configs/dynamic/unified_config.yaml"
            success = self.deploy_system(config_path, deploy_mode)
            
            if success:
                print("\n🎉 完整部署流程成功完成！")
                print(f"   - 硬件检测: ✅")
                print(f"   - 配置生成: ✅")
                print(f"   - 系统部署: ✅")
                print(f"   - 配置文件: {config_path}")
                
                # 显示系统信息
                print("\n📊 系统配置摘要:")
                print(f"   - 配置档案: {config['deployment']['profile']}")
                print(f"   - GPU数量: {hardware_info['gpu_count']}")
                print(f"   - 总GPU内存: {hardware_info['total_gpu_memory']:.1f}GB")
                print(f"   - 批处理大小: {config['stable_diffusion']['batch_size']}")
                print(f"   - 并发生成数: {config['stable_diffusion']['concurrent_generations']}")
                
                return True
            else:
                print("\n❌ 部署流程失败！")
                return False
                
        except Exception as e:
            print(f"\n❌ 部署流程异常: {e}")
            self.logger.error(f"部署流程异常: {e}")
            return False
    
    def interactive_mode(self):
        """交互模式"""
        print("\n🎮 进入交互模式...")
        print("可用命令:")
        print("  1. detect - 检测硬件")
        print("  2. config - 生成配置")
        print("  3. deploy - 部署系统")
        print("  4. full - 完整部署")
        print("  5. status - 查看状态")
        print("  6. quit - 退出")
        
        hardware_info = None
        config_path = None
        
        while True:
            try:
                command = input("\n请输入命令: ").strip().lower()
                
                if command == "detect" or command == "1":
                    hardware_info = self.detect_hardware()
                    
                elif command == "config" or command == "2":
                    if hardware_info is None:
                        print("请先执行硬件检测 (detect)")
                        continue
                    
                    profile = input("请输入配置档案 (high_performance/medium/standard/basic/cpu_fallback, 回车使用自动): ").strip()
                    if not profile:
                        profile = None
                    
                    config = self.generate_config(hardware_info, profile=profile)
                    config_path = "configs/dynamic/interactive_config.yaml"
                    
                elif command == "deploy" or command == "3":
                    if config_path is None:
                        print("请先生成配置 (config)")
                        continue
                    
                    mode = input("请选择部署模式 (quick/auto, 回车使用quick): ").strip()
                    if not mode:
                        mode = "quick"
                    
                    self.deploy_system(config_path, mode)
                    
                elif command == "full" or command == "4":
                    profile = input("请输入配置档案 (回车使用自动): ").strip() or None
                    mode = input("请选择部署模式 (quick/auto, 回车使用quick): ").strip() or "quick"
                    
                    self.run_full_deployment(profile=profile, deploy_mode=mode)
                    
                elif command == "status" or command == "5":
                    print("\n📊 当前状态:")
                    print(f"   - 硬件检测: {'✅' if hardware_info else '❌'}")
                    print(f"   - 配置生成: {'✅' if config_path else '❌'}")
                    
                elif command == "quit" or command == "6" or command == "q":
                    print("👋 退出交互模式")
                    break
                    
                else:
                    print("❌ 未知命令，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n👋 退出交互模式")
                break
            except Exception as e:
                print(f"❌ 命令执行失败: {e}")


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="多模态检测系统统一部署工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python deploy.py                           # 完整部署流程
  python deploy.py --detect-only             # 仅检测硬件
  python deploy.py --config-only             # 仅生成配置
  python deploy.py --deploy-only             # 仅部署系统
  python deploy.py --interactive             # 交互模式
  python deploy.py --profile high_performance # 使用指定配置档案
  python deploy.py --deploy-mode auto        # 使用自动部署模式
        """
    )
    
    # 运行模式
    parser.add_argument("--detect-only", action="store_true", help="仅检测硬件")
    parser.add_argument("--config-only", action="store_true", help="仅生成配置")
    parser.add_argument("--deploy-only", action="store_true", help="仅部署系统")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    
    # 配置选项
    parser.add_argument("--profile", choices=["high_performance", "medium", "standard", "basic", "cpu_fallback"],
                       help="强制使用指定配置档案")
    parser.add_argument("--config-input", type=str, help="输入配置文件路径")
    parser.add_argument("--config-output", type=str, help="输出配置文件路径")
    parser.add_argument("--hardware-output", type=str, help="硬件信息输出文件路径")
    
    # 部署选项
    parser.add_argument("--deploy-mode", choices=["quick", "auto"], default="quick", help="部署模式")
    
    # 其他选项
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 创建部署管理器
    manager = UnifiedDeployManager()
    
    try:
        if args.interactive:
            # 交互模式
            manager.interactive_mode()
            
        elif args.detect_only:
            # 仅检测硬件
            manager.detect_hardware(args.hardware_output)
            
        elif args.config_only:
            # 仅生成配置
            if args.config_input:
                # 从文件加载硬件信息
                with open(args.config_input, 'r', encoding='utf-8') as f:
                    hardware_info = json.load(f)
            else:
                # 自动检测硬件
                hardware_info = manager.detect_hardware(args.hardware_output)
            
            manager.generate_config(hardware_info, args.config_output, args.profile)
            
        elif args.deploy_only:
            # 仅部署系统
            config_path = args.config_input or "configs/dynamic/unified_config.yaml"
            if not os.path.exists(config_path):
                print(f"❌ 配置文件不存在: {config_path}")
                print("请先生成配置或指定正确的配置文件路径")
                sys.exit(1)
            
            manager.deploy_system(config_path, args.deploy_mode)
            
        else:
            # 完整部署流程
            success = manager.run_full_deployment(
                profile=args.profile,
                config_output=args.config_output,
                hardware_output=args.hardware_output,
                deploy_mode=args.deploy_mode
            )
            
            if not success:
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n👋 用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()