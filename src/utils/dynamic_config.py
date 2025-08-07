"""动态配置管理模块

根据硬件检测结果自动生成和管理配置文件。
"""

import json
import yaml
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .hardware_detector import HardwareDetector, HardwareConfig
from ..models.sd_model import StableDiffusionConfig
from ..models.multi_gpu_sd_manager import MultiGPUSDConfig
from .multi_gpu_processor import MultiGPUConfig

logger = logging.getLogger(__name__)


@dataclass
class DynamicConfigProfile:
    """动态配置档案"""
    name: str
    description: str
    hardware_requirements: Dict[str, Any]
    config_template: Dict[str, Any]
    priority: int = 0  # 优先级，数字越大优先级越高


class DynamicConfigManager:
    """动态配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.hardware_detector = HardwareDetector()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 预定义配置档案
        self.profiles = self._load_predefined_profiles()
    
    def _load_predefined_profiles(self) -> Dict[str, DynamicConfigProfile]:
        """加载预定义配置档案"""
        profiles = {}
        
        # 高性能配置 (6+ GPUs, 24GB+ each)
        profiles["high_performance"] = DynamicConfigProfile(
            name="high_performance",
            description="高性能配置 - 适用于6+个24GB+显存的GPU",
            hardware_requirements={
                "min_gpu_count": 6,
                "min_gpu_memory_mb": 24000,
                "min_total_memory_mb": 144000
            },
            config_template={
                "stable_diffusion": {
                    "model_name": "runwayml/stable-diffusion-v1-5",
                    "device": "auto",
                    "torch_dtype": "float16",
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "height": 512,
                    "width": 512,
                    "num_images_per_prompt": 4,
                    "enable_attention_slicing": True,
                    "enable_xformers": True,
                    "enable_cpu_offload": False,
                    "enable_safety_checker": False,
                    "enable_multi_gpu": True,
                    "models_per_gpu": 2,
                    "max_models_per_gpu": 2
                },
                "multi_gpu": {
                    "batch_size_per_gpu": 8,
                    "max_workers": 12,
                    "memory_fraction": 0.9,
                    "enable_mixed_precision": True,
                    "enable_compile": True,
                    "load_balancing": True
                },
                "detection": {
                    "batch_size": 32,
                    "num_workers": 8,
                    "prefetch_factor": 4
                }
            },
            priority=100
        )
        
        # 中等性能配置 (4-5 GPUs, 12-24GB each)
        profiles["medium_performance"] = DynamicConfigProfile(
            name="medium_performance",
            description="中等性能配置 - 适用于4-5个12-24GB显存的GPU",
            hardware_requirements={
                "min_gpu_count": 4,
                "max_gpu_count": 5,
                "min_gpu_memory_mb": 12000,
                "min_total_memory_mb": 48000
            },
            config_template={
                "stable_diffusion": {
                    "model_name": "runwayml/stable-diffusion-v1-5",
                    "device": "auto",
                    "torch_dtype": "float16",
                    "num_inference_steps": 40,
                    "guidance_scale": 7.5,
                    "height": 512,
                    "width": 512,
                    "num_images_per_prompt": 2,
                    "enable_attention_slicing": True,
                    "enable_xformers": True,
                    "enable_cpu_offload": False,
                    "enable_safety_checker": False,
                    "enable_multi_gpu": True,
                    "models_per_gpu": 1,
                    "max_models_per_gpu": 2
                },
                "multi_gpu": {
                    "batch_size_per_gpu": 4,
                    "max_workers": 8,
                    "memory_fraction": 0.85,
                    "enable_mixed_precision": True,
                    "enable_compile": True,
                    "load_balancing": True
                },
                "detection": {
                    "batch_size": 16,
                    "num_workers": 6,
                    "prefetch_factor": 3
                }
            },
            priority=80
        )
        
        # 标准配置 (2-3 GPUs, 8-12GB each)
        profiles["standard"] = DynamicConfigProfile(
            name="standard",
            description="标准配置 - 适用于2-3个8-12GB显存的GPU",
            hardware_requirements={
                "min_gpu_count": 2,
                "max_gpu_count": 3,
                "min_gpu_memory_mb": 8000,
                "min_total_memory_mb": 16000
            },
            config_template={
                "stable_diffusion": {
                    "model_name": "runwayml/stable-diffusion-v1-5",
                    "device": "auto",
                    "torch_dtype": "float16",
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "height": 512,
                    "width": 512,
                    "num_images_per_prompt": 1,
                    "enable_attention_slicing": True,
                    "enable_xformers": True,
                    "enable_cpu_offload": True,
                    "enable_safety_checker": False,
                    "enable_multi_gpu": True,
                    "models_per_gpu": 1,
                    "max_models_per_gpu": 1
                },
                "multi_gpu": {
                    "batch_size_per_gpu": 2,
                    "max_workers": 4,
                    "memory_fraction": 0.8,
                    "enable_mixed_precision": True,
                    "enable_compile": False,
                    "load_balancing": True
                },
                "detection": {
                    "batch_size": 8,
                    "num_workers": 4,
                    "prefetch_factor": 2
                }
            },
            priority=60
        )
        
        # 基础配置 (1 GPU, 6-8GB)
        profiles["basic"] = DynamicConfigProfile(
            name="basic",
            description="基础配置 - 适用于单个6-8GB显存的GPU",
            hardware_requirements={
                "min_gpu_count": 1,
                "max_gpu_count": 1,
                "min_gpu_memory_mb": 6000,
                "min_total_memory_mb": 6000
            },
            config_template={
                "stable_diffusion": {
                    "model_name": "runwayml/stable-diffusion-v1-5",
                    "device": "auto",
                    "torch_dtype": "float16",
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "height": 512,
                    "width": 512,
                    "num_images_per_prompt": 1,
                    "enable_attention_slicing": True,
                    "enable_xformers": False,
                    "enable_cpu_offload": True,
                    "enable_safety_checker": False,
                    "enable_multi_gpu": False,
                    "models_per_gpu": 1,
                    "max_models_per_gpu": 1
                },
                "multi_gpu": {
                    "batch_size_per_gpu": 1,
                    "max_workers": 1,
                    "memory_fraction": 0.7,
                    "enable_mixed_precision": False,
                    "enable_compile": False,
                    "load_balancing": False
                },
                "detection": {
                    "batch_size": 4,
                    "num_workers": 2,
                    "prefetch_factor": 1
                }
            },
            priority=40
        )
        
        # CPU回退配置
        profiles["cpu_fallback"] = DynamicConfigProfile(
            name="cpu_fallback",
            description="CPU回退配置 - 无可用GPU时使用",
            hardware_requirements={
                "min_gpu_count": 0,
                "max_gpu_count": 0
            },
            config_template={
                "stable_diffusion": {
                    "model_name": "runwayml/stable-diffusion-v1-5",
                    "device": "cpu",
                    "torch_dtype": "float32",
                    "num_inference_steps": 10,
                    "guidance_scale": 7.5,
                    "height": 256,
                    "width": 256,
                    "num_images_per_prompt": 1,
                    "enable_attention_slicing": False,
                    "enable_xformers": False,
                    "enable_cpu_offload": False,
                    "enable_safety_checker": False,
                    "enable_multi_gpu": False,
                    "models_per_gpu": 1,
                    "max_models_per_gpu": 1
                },
                "multi_gpu": {
                    "batch_size_per_gpu": 1,
                    "max_workers": 1,
                    "memory_fraction": 0.5,
                    "enable_mixed_precision": False,
                    "enable_compile": False,
                    "load_balancing": False
                },
                "detection": {
                    "batch_size": 1,
                    "num_workers": 1,
                    "prefetch_factor": 1
                }
            },
            priority=10
        )
        
        return profiles
    
    def select_best_profile(self, hardware_config: HardwareConfig) -> DynamicConfigProfile:
        """根据硬件配置选择最佳配置档案"""
        available_gpus = [gpu for gpu in hardware_config.gpus if gpu.is_available]
        gpu_count = len(available_gpus)
        
        if gpu_count == 0:
            return self.profiles["cpu_fallback"]
        
        total_gpu_memory = sum(gpu.memory_total for gpu in available_gpus)
        min_gpu_memory = min(gpu.memory_total for gpu in available_gpus)
        
        # 筛选符合条件的配置档案
        suitable_profiles = []
        
        for profile in self.profiles.values():
            req = profile.hardware_requirements
            
            # 检查GPU数量
            if "min_gpu_count" in req and gpu_count < req["min_gpu_count"]:
                continue
            if "max_gpu_count" in req and gpu_count > req["max_gpu_count"]:
                continue
            
            # 检查GPU内存
            if "min_gpu_memory_mb" in req and min_gpu_memory < req["min_gpu_memory_mb"]:
                continue
            if "min_total_memory_mb" in req and total_gpu_memory < req["min_total_memory_mb"]:
                continue
            
            suitable_profiles.append(profile)
        
        # 选择优先级最高的配置档案
        if suitable_profiles:
            best_profile = max(suitable_profiles, key=lambda p: p.priority)
            self.logger.info(f"选择配置档案: {best_profile.name} ({best_profile.description})")
            return best_profile
        else:
            # 如果没有合适的配置档案，使用基础配置
            self.logger.warning("未找到合适的配置档案，使用基础配置")
            return self.profiles["basic"]
    
    def generate_config(self, hardware_config: Optional[HardwareConfig] = None, 
                        force_profile: Optional[str] = None) -> Dict[str, Any]:
        """生成动态配置
        
        Args:
            hardware_config: HardwareConfig对象或包含硬件信息的字典
            force_profile: 强制使用的配置档案名称
        """
        if hardware_config is None:
            hardware_config = self.hardware_detector.detect_hardware()
        
        # 如果传入的是字典，转换为HardwareConfig对象
        if isinstance(hardware_config, dict):
            from .hardware_detector import GPUInfo, SystemInfo
            
            # 创建GPU信息列表
            gpu_infos = []
            if 'gpu_info' in hardware_config:
                for gpu_data in hardware_config['gpu_info']:
                    if isinstance(gpu_data, dict):
                        gpu_info = GPUInfo(
                            id=gpu_data.get('id', 0),
                            name=gpu_data.get('name', 'Unknown'),
                            memory_total=gpu_data.get('memory_total', 0),
                            memory_free=gpu_data.get('memory_free', 0),
                            compute_capability=(7, 5),  # 默认值
                            is_available=gpu_data.get('is_available', True),
                            temperature=gpu_data.get('temperature'),
                            utilization=gpu_data.get('utilization')
                        )
                        gpu_infos.append(gpu_info)
            
            # 如果没有详细GPU信息，根据gpu_count创建基本信息
            if not gpu_infos and hardware_config.get('gpu_count', 0) > 0:
                avg_memory = hardware_config.get('total_gpu_memory', 0) / hardware_config.get('gpu_count', 1)
                for i in range(hardware_config.get('gpu_count', 0)):
                    gpu_info = GPUInfo(
                        id=i,
                        name='GPU',
                        memory_total=int(avg_memory * 1024),  # 转换为MB
                        memory_free=int(avg_memory * 1024),
                        compute_capability=(7, 5),  # 默认值
                        is_available=True,
                        temperature=None,
                        utilization=None
                    )
                    gpu_infos.append(gpu_info)
            
            # 创建系统信息
            system_info = SystemInfo(
                cpu_count=hardware_config.get('cpu_cores', 0),
                memory_total=hardware_config.get('system_memory', 0) * 1024,  # 转换为MB
                memory_available=int(hardware_config.get('system_memory', 0) * 1024 * 0.8),
                platform=hardware_config.get('platform', 'Unknown'),
                python_version='3.12',  # 默认值
                torch_version='2.0',    # 默认值
                cuda_version=hardware_config.get('cuda_version')
            )
            
            # 创建HardwareConfig对象
            hardware_config = HardwareConfig(
                gpus=gpu_infos,
                system=system_info,
                recommended_config={}
            )
        
        # 选择最佳配置档案
        if force_profile and force_profile in self.profiles:
            profile = self.profiles[force_profile]
        else:
            profile = self.select_best_profile(hardware_config)
        
        # 复制配置模板
        config = profile.config_template.copy()
        
        # 根据实际硬件调整配置
        available_gpus = [gpu for gpu in hardware_config.gpus if gpu.is_available]
        
        if available_gpus:
            # 更新GPU相关配置
            config["stable_diffusion"]["gpu_ids"] = [gpu.id for gpu in available_gpus]
            config["multi_gpu"]["gpu_ids"] = [gpu.id for gpu in available_gpus]
            
            # 根据实际GPU数量调整并发设置
            gpu_count = len(available_gpus)
            models_per_gpu = config["stable_diffusion"]["models_per_gpu"]
            max_concurrent = min(
                gpu_count * models_per_gpu * 2,
                config["multi_gpu"]["max_workers"]
            )
            config["stable_diffusion"]["max_concurrent_generations"] = max_concurrent
            
            # 特殊GPU优化
            gpu_names = [gpu.name for gpu in available_gpus]
            if any("A100" in name or "A200" in name for name in gpu_names):
                config["stable_diffusion"]["enable_tensor_cores"] = True
                config["stable_diffusion"]["enable_flash_attention"] = True
            elif any("RTX 4090" in name or "RTX 4080" in name for name in gpu_names):
                config["stable_diffusion"]["enable_tensor_cores"] = True
                config["stable_diffusion"]["enable_flash_attention"] = False
        
        # 添加硬件信息到配置
        config["hardware_info"] = {
            "gpu_count": len(available_gpus),
            "gpu_names": [gpu.name for gpu in available_gpus],
            "total_gpu_memory_gb": sum(gpu.memory_total for gpu in available_gpus) / 1024,
            "system_memory_gb": hardware_config.system.memory_total / 1024,
            "cpu_count": hardware_config.system.cpu_count,
            "cuda_version": hardware_config.system.cuda_version,
            "profile_used": profile.name
        }
        
        return config
    
    def create_sd_config(self, dynamic_config: Dict[str, Any]) -> StableDiffusionConfig:
        """从动态配置创建SD配置对象"""
        sd_config = dynamic_config["stable_diffusion"]
        
        return StableDiffusionConfig(
            model_name=sd_config["model_name"],
            device=sd_config["device"],
            torch_dtype=sd_config["torch_dtype"],
            num_inference_steps=sd_config["num_inference_steps"],
            guidance_scale=sd_config["guidance_scale"],
            height=sd_config["height"],
            width=sd_config["width"],
            num_images_per_prompt=sd_config["num_images_per_prompt"],
            enable_attention_slicing=sd_config["enable_attention_slicing"],
            enable_xformers=sd_config["enable_xformers"],
            enable_cpu_offload=sd_config["enable_cpu_offload"],
            enable_safety_checker=sd_config["enable_safety_checker"],
            enable_multi_gpu=sd_config["enable_multi_gpu"],
            gpu_ids=sd_config.get("gpu_ids"),
            models_per_gpu=sd_config["models_per_gpu"],
            max_models_per_gpu=sd_config["max_models_per_gpu"]
        )
    
    def create_multi_gpu_config(self, dynamic_config: Dict[str, Any]) -> MultiGPUConfig:
        """从动态配置创建多GPU配置对象"""
        mg_config = dynamic_config["multi_gpu"]
        
        return MultiGPUConfig(
            gpu_ids=mg_config.get("gpu_ids"),
            batch_size_per_gpu=mg_config["batch_size_per_gpu"],
            max_workers=mg_config["max_workers"],
            memory_fraction=mg_config["memory_fraction"],
            enable_mixed_precision=mg_config["enable_mixed_precision"],
            enable_compile=mg_config["enable_compile"],
            load_balancing=mg_config["load_balancing"]
        )
    
    def save_config(self, config: Dict[str, Any], filename: str = "dynamic_config.yaml") -> str:
        """保存配置到文件"""
        # 如果filename是绝对路径，直接使用；否则相对于config_dir
        if os.path.isabs(filename):
            config_path = Path(filename)
        else:
            config_path = self.config_dir / filename
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        self.logger.info(f"动态配置已保存到: {config_path}")
        return str(config_path)
    
    def load_config(self, filename: str = "dynamic_config.yaml") -> Optional[Dict[str, Any]]:
        """从文件加载配置"""
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"动态配置已从 {config_path} 加载")
            return config
            
        except Exception as e:
            self.logger.error(f"加载动态配置失败: {e}")
            return None
    
    def auto_configure(self, save_config: bool = True) -> Dict[str, Any]:
        """自动检测硬件并生成配置"""
        self.logger.info("开始自动配置...")
        
        # 检测硬件
        hardware_config = self.hardware_detector.detect_hardware()
        
        # 生成配置
        dynamic_config = self.generate_config(hardware_config)
        
        # 保存配置
        if save_config:
            self.save_config(dynamic_config)
            
            # 同时保存硬件信息
            hardware_path = self.config_dir / "hardware_config.json"
            self.hardware_detector.save_config(hardware_config, str(hardware_path))
        
        self.logger.info("自动配置完成")
        return dynamic_config
    
    def print_config_summary(self, config: Dict[str, Any]) -> None:
        """打印配置摘要"""
        print("\n" + "="*60)
        print("⚙️  动态配置摘要")
        print("="*60)
        
        # 硬件信息
        if "hardware_info" in config:
            hw = config["hardware_info"]
            print(f"\n🖥️  硬件信息:")
            print(f"   GPU数量: {hw['gpu_count']}")
            if hw['gpu_count'] > 0:
                print(f"   GPU型号: {', '.join(set(hw['gpu_names']))}")
                print(f"   总GPU内存: {hw['total_gpu_memory_gb']:.1f}GB")
            print(f"   系统内存: {hw['system_memory_gb']:.1f}GB")
            print(f"   CPU核心: {hw['cpu_count']}")
            print(f"   CUDA版本: {hw['cuda_version'] or 'N/A'}")
            print(f"   使用配置档案: {hw['profile_used']}")
        
        # SD配置
        if "stable_diffusion" in config:
            sd = config["stable_diffusion"]
            print(f"\n🎨 Stable Diffusion配置:")
            print(f"   模型: {sd['model_name']}")
            print(f"   设备: {sd['device']}")
            print(f"   数据类型: {sd['torch_dtype']}")
            print(f"   推理步数: {sd['num_inference_steps']}")
            print(f"   每提示图像数: {sd['num_images_per_prompt']}")
            print(f"   多GPU: {'启用' if sd['enable_multi_gpu'] else '禁用'}")
            if sd['enable_multi_gpu'] and 'gpu_ids' in sd:
                print(f"   GPU ID: {sd['gpu_ids']}")
                print(f"   每GPU模型数: {sd['models_per_gpu']}")
        
        # 多GPU配置
        if "multi_gpu" in config:
            mg = config["multi_gpu"]
            print(f"\n🔧 多GPU配置:")
            print(f"   每GPU批处理大小: {mg['batch_size_per_gpu']}")
            print(f"   最大工作线程: {mg['max_workers']}")
            print(f"   内存使用比例: {mg['memory_fraction']*100:.0f}%")
            print(f"   混合精度: {'启用' if mg['enable_mixed_precision'] else '禁用'}")
            print(f"   编译优化: {'启用' if mg['enable_compile'] else '禁用'}")
            print(f"   负载均衡: {'启用' if mg['load_balancing'] else '禁用'}")
        
        print("="*60 + "\n")


def auto_configure_system() -> Dict[str, Any]:
    """自动配置系统的便捷函数"""
    manager = DynamicConfigManager()
    config = manager.auto_configure()
    manager.print_config_summary(config)
    return config


def main():
    """Main entry point for console script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Configuration Generator")
    parser.add_argument("--hardware-info", type=str, help="硬件信息JSON文件路径")
    parser.add_argument("--output", "-o", type=str, help="输出配置文件路径")
    parser.add_argument("--profile", choices=["high_performance", "medium", "standard", "basic", "cpu_fallback"], 
                        help="强制使用指定配置档案")
    parser.add_argument("--test", action="store_true", help="运行测试模式")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    manager = DynamicConfigManager()
    
    if args.test:
        # 测试模式：使用预定义的硬件配置
        print("=== 动态配置管理测试 ===")
        
        test_configs = [
            {
                'gpu_count': 6,
                'total_gpu_memory': 144.0,  # 6 * 24GB RTX 4090
                'system_memory': 128.0,
                'cpu_cores': 32
            },
            {
                'gpu_count': 2,
                'total_gpu_memory': 16.0,   # 2 * 8GB RTX 3070
                'system_memory': 32.0,
                'cpu_cores': 16
            },
            {
                'gpu_count': 0,
                'total_gpu_memory': 0.0,    # CPU only
                'system_memory': 16.0,
                'cpu_cores': 8
            }
        ]
        
        for i, hardware_info in enumerate(test_configs, 1):
            print(f"\n--- 测试配置 {i} ---")
            print(f"GPU数量: {hardware_info['gpu_count']}")
            print(f"GPU内存: {hardware_info['total_gpu_memory']}GB")
            print(f"系统内存: {hardware_info['system_memory']}GB")
            print(f"CPU核心: {hardware_info['cpu_cores']}")
            
            config = manager.generate_config(hardware_info, force_profile=args.profile)
            manager.print_config_summary(config)
            
            if args.output:
                config_path = f"{args.output}_test_{i}.yaml"
                manager.save_config(config, config_path)
                print(f"配置已保存到: {config_path}")
    
    else:
        # 正常模式：使用实际硬件检测或指定的硬件信息
        if args.hardware_info:
            # 从文件加载硬件信息
            import json
            with open(args.hardware_info, 'r', encoding='utf-8') as f:
                hardware_info = json.load(f)
        else:
            # 自动检测硬件
            from .hardware_detector import HardwareDetector
            detector = HardwareDetector()
            hardware_info = detector.detect_hardware()
            print("=== 硬件检测结果 ===")
            detector.print_summary(hardware_info)
        
        print("\n=== 生成动态配置 ===")
        config = manager.generate_config(hardware_info, force_profile=args.profile)
        
        if args.verbose:
            manager.print_config_summary(config)
        
        # 保存配置
        if args.output:
            manager.save_config(config, args.output)
            print(f"✅ 配置已保存到: {args.output}")
        else:
            # 默认保存路径
            output_path = "configs/dynamic/auto_generated_config.yaml"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            manager.save_config(config, output_path)
            print(f"✅ 配置已保存到: {output_path}")


if __name__ == "__main__":
    main()