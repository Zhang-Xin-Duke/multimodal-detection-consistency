#!/usr/bin/env python3
"""
Setup script for Multi-Modal Retrieval Defense
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import sys
import subprocess
import json
from pathlib import Path

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Remove comments and empty lines
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Setup CUDA environment and hardware detection after installation
        self._setup_cuda_environment()
        self._setup_hardware_detection()
    
    def _setup_cuda_environment(self):
        """Setup CUDA environment and configuration"""
        try:
            import torch
            if torch.cuda.is_available():
                print("\n🚀 Setting up CUDA environment...")
                
                # Set CUDA environment variables
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                os.environ['TORCH_CUDA_ARCH_LIST'] = '6.0;6.1;7.0;7.5;8.0;8.6'
                
                # Check CUDA devices
                device_count = torch.cuda.device_count()
                print(f"✅ Detected {device_count} CUDA device(s)")
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory // 1024**3
                    print(f"   Device {i}: {props.name}, Memory: {memory_gb}GB")
                
                # Set default device
                torch.cuda.set_device(0)
                print(f"✅ Default CUDA device set to: {torch.cuda.current_device()}")
                
                # Test CUDA functionality
                try:
                    test_tensor = torch.randn(10, 10).cuda()
                    torch.cuda.empty_cache()
                    print("✅ CUDA functionality test passed")
                except Exception as e:
                    print(f"⚠️  CUDA test failed: {e}")
                
                print("✅ CUDA environment setup completed")
            else:
                print("⚠️  CUDA not available, using CPU mode")
        except ImportError:
            print("⚠️  PyTorch not installed, skipping CUDA setup")
        except Exception as e:
            print(f"⚠️  CUDA setup failed: {e}")
    
    def _setup_hardware_detection(self):
        """Setup hardware detection and generate dynamic configuration"""
        try:
            print("\n🔍 Running hardware detection and configuration...")
            
            # Create necessary directories
            config_dir = Path("configs/dynamic")
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Import hardware detection modules
            sys.path.insert(0, os.path.abspath('.'))
            from src.utils.hardware_detector import HardwareDetector
            from src.utils.dynamic_config import DynamicConfigManager
            
            # Run hardware detection
            detector = HardwareDetector()
            hardware_info = detector.detect_hardware()
            
            print(f"✅ Hardware detection completed:")
            print(f"   GPU Count: {hardware_info['gpu_count']}")
            print(f"   Total GPU Memory: {hardware_info['total_gpu_memory']:.1f}GB")
            print(f"   System Memory: {hardware_info['system_memory']:.1f}GB")
            print(f"   CPU Cores: {hardware_info['cpu_cores']}")
            
            # Generate dynamic configuration
            config_manager = DynamicConfigManager()
            config = config_manager.generate_config(hardware_info)
            
            # Save configuration
            config_path = config_dir / "auto_generated_config.yaml"
            config_manager.save_config(config, str(config_path))
            
            print(f"✅ Dynamic configuration saved to: {config_path}")
            print(f"   Selected Profile: {config['deployment']['profile']}")
            print(f"   Batch Size: {config['stable_diffusion']['batch_size']}")
            print(f"   Concurrent Generations: {config['stable_diffusion']['concurrent_generations']}")
            
            # Ask user if they want to run quick start
            try:
                response = input("\n🚀 Would you like to run quick start now? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    print("\n🚀 Starting quick deployment...")
                    from quick_start import QuickStartManager
                    quick_start = QuickStartManager()
                    quick_start.run(mode='start', config_path=str(config_path))
                else:
                    print("\n📝 You can run quick start later with: python quick_start.py")
                    print("   Or use the command: mm-deploy")
            except (KeyboardInterrupt, EOFError):
                print("\n📝 You can run quick start later with: python quick_start.py")
                print("   Or use the command: mm-deploy")
            
        except ImportError as e:
            print(f"⚠️  Hardware detection modules not available: {e}")
            print("   Please ensure all dependencies are installed")
        except Exception as e:
            print(f"⚠️  Hardware detection setup failed: {e}")
            print("   You can run it manually later with: python quick_start.py")


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        # Setup CUDA environment and hardware detection after development installation
        post_install = PostInstallCommand(self.distribution)
        post_install._setup_cuda_environment()
        post_install._setup_hardware_detection()

setup(
    name="multimodal-retrieval-defense",
    version="0.1.0",
    author="Zhang Xin",
    author_email="zhang.xin@duke.edu",
    description="Multi-Modal Retrieval Defense via Text Variant Consistency Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangxin-duke/multimodal-retrieval-defense",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "all": [
            # All dependencies are now included in the main requirements.txt
            # Use: pip install -e .[all] for complete installation
        ],
    },
    entry_points={
        "console_scripts": [
            "mm-defense=experiments.run_experiments:main",
            "mm-attack=experiments.run_attacks:main",
            "mm-deploy=deploy:main",
            "mm-quick-start=quick_start:main",
            "mm-auto-deploy=auto_deploy:main",
            "mm-hardware-detect=src.utils.hardware_detector:main",
            "mm-config-gen=src.utils.dynamic_config:main",
        ],
    },
    include_package_data=True,
    package_data={
        "multimodal_retrieval_defense": [
            "configs/*.yaml",
            "configs/**/*.yaml",
        ],
    },
    zip_safe=False,
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    keywords=[
        "multimodal",
        "retrieval",
        "adversarial",
        "defense",
        "computer vision",
        "natural language processing",
        "machine learning",
        "deep learning",
        "pytorch",
        "clip",
        "stable diffusion",
    ],
    project_urls={
        "Bug Reports": "https://github.com/zhangxin-duke/multimodal-retrieval-defense/issues",
        "Source": "https://github.com/zhangxin-duke/multimodal-retrieval-defense",
        "Documentation": "https://github.com/zhangxin-duke/multimodal-retrieval-defense/blob/main/README.md",
    },
)