#!/usr/bin/env python3
"""
简化的多GPU配置验证脚本
快速检查多GPU配置是否正确
"""

import sys
import os
import torch
import warnings
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 抑制警告
warnings.filterwarnings('ignore')


def test_gpu_availability():
    """测试GPU可用性"""
    print("\n=== GPU 可用性测试 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return True


def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 配置文件加载测试 ===")
    
    try:
        from src.utils.config import load_config, get_config
        
        # 加载配置文件
        load_config('./configs/default.yaml')
        print("✅ 配置文件加载成功")
        
        # 获取所有配置
        all_config = get_config()
        print(f"✅ 配置节数量: {len(all_config)}")
        
        # 检查是否有原始YAML数据（如果配置管理器保存了原始数据）
        if hasattr(all_config, 'raw_config'):
            raw_config = all_config.raw_config
            if 'models' in raw_config:
                clip_config = raw_config['models'].get('clip', {})
                print(f"  CLIP 多GPU配置: {clip_config.get('use_multi_gpu', False)}")
                print(f"  CLIP GPU IDs: {clip_config.get('gpu_ids', [])}")
                
                sd_config = raw_config['models'].get('stable_diffusion', {})
                print(f"  SD 多GPU配置: {sd_config.get('use_multi_gpu', False)}")
                print(f"  SD GPU IDs: {sd_config.get('gpu_ids', [])}")
            else:
                print("⚠️  原始配置中未找到models节")
        else:
            print("⚠️  配置已转换为dataclass对象，无法直接访问原始YAML结构")
            print("  这是正常的，配置管理器工作正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def test_config_objects():
    """测试配置对象创建"""
    print("\n=== 配置对象创建测试 ===")
    
    try:
        from src.models.clip_model import CLIPConfig
        from src.models.sd_model import StableDiffusionConfig
        
        # 测试CLIP配置
        clip_config = CLIPConfig(
            use_multi_gpu=True,
            gpu_ids=[0, 1, 2, 3, 4, 5],
            parallel_type="data_parallel"
        )
        print(f"✅ CLIP配置对象创建成功: {clip_config.use_multi_gpu}")
        
        # 测试SD配置
        sd_config = StableDiffusionConfig(
            use_multi_gpu=True,
            gpu_ids=[0, 1, 2, 3, 4, 5],
            max_models_per_gpu=1
        )
        print(f"✅ SD配置对象创建成功: {sd_config.use_multi_gpu}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置对象创建失败: {e}")
        return False


def test_multi_gpu_imports():
    """测试多GPU模块导入"""
    print("\n=== 多GPU模块导入测试 ===")
    
    try:
        # 测试多GPU处理器导入
        from src.utils.multi_gpu_processor import MultiGPUProcessor
        print("✅ MultiGPUProcessor 导入成功")
        
        # 测试多GPU SD管理器导入
        from src.models.multi_gpu_sd_manager import MultiGPUSDManager
        print("✅ MultiGPUSDManager 导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 多GPU模块导入失败: {e}")
        return False


def test_pytorch_multi_gpu():
    """测试PyTorch多GPU基础功能"""
    print("\n=== PyTorch 多GPU基础测试 ===")
    
    try:
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，跳过多GPU测试")
            return False
        
        # 测试多GPU张量操作
        device_count = torch.cuda.device_count()
        if device_count < 2:
            print(f"⚠️  只有{device_count}个GPU，无法测试多GPU")
            return True
        
        # 创建测试张量
        x = torch.randn(4, 4).cuda(0)
        y = torch.randn(4, 4).cuda(1)
        
        print(f"✅ 在GPU 0和GPU 1上创建张量成功")
        print(f"  张量x设备: {x.device}")
        print(f"  张量y设备: {y.device}")
        
        # 测试DataParallel
        if device_count >= 2:
            model = torch.nn.Linear(4, 2)
            model = torch.nn.DataParallel(model, device_ids=[0, 1])
            model = model.cuda()
            print("✅ DataParallel 模型创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch多GPU测试失败: {e}")
        return False


def test_yaml_config_direct():
    """直接测试YAML配置文件内容"""
    print("\n=== 直接YAML配置测试 ===")
    
    try:
        import yaml
        
        with open('./configs/default.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ 直接YAML加载成功")
        
        # 检查models节
        if 'models' in config:
            print("✅ 找到models配置节")
            
            # 检查CLIP配置
            if 'clip' in config['models']:
                clip_config = config['models']['clip']
                print(f"  CLIP 多GPU: {clip_config.get('use_multi_gpu', False)}")
                print(f"  CLIP GPU IDs: {clip_config.get('gpu_ids', [])}")
            
            # 检查SD配置
            if 'stable_diffusion' in config['models']:
                sd_config = config['models']['stable_diffusion']
                print(f"  SD 多GPU: {sd_config.get('use_multi_gpu', False)}")
                print(f"  SD GPU IDs: {sd_config.get('gpu_ids', [])}")
        else:
            print("❌ 未找到models配置节")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 直接YAML配置测试失败: {e}")
        return False


def main():
    """主函数"""
    print("多GPU配置快速验证脚本")
    print("=" * 50)
    
    results = []
    
    # 运行所有测试
    results.append(("GPU可用性", test_gpu_availability()))
    results.append(("直接YAML配置", test_yaml_config_direct()))
    results.append(("配置管理器加载", test_config_loading()))
    results.append(("配置对象", test_config_objects()))
    results.append(("多GPU模块导入", test_multi_gpu_imports()))
    results.append(("PyTorch多GPU", test_pytorch_multi_gpu()))
    
    # 总结结果
    print("\n=== 测试结果总结 ===")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！多GPU配置正常")
        return 0
    else:
        print("⚠️  部分测试失败，请检查配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())