#!/bin/bash
# setup.sh - 环境安装脚本
# 多模态检测一致性实验代码环境配置
# Author: ZHANG XIN <zhang.xin@duke.edu>
# Duke University

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查操作系统
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_error "此脚本仅支持Linux系统"
        exit 1
    fi
    
    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3未安装，请先安装Python 3.8+"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    # 使用Python进行版本比较，避免bc依赖
    version_check=$(python3 -c "import sys; print(1 if (sys.version_info.major, sys.version_info.minor) >= (3, 8) else 0)")
    if [[ $version_check -eq 0 ]]; then
        log_error "Python版本过低，需要3.8+，当前版本: $python_version"
        exit 1
    fi
    
    log_success "Python版本检查通过: $python_version"
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        log_success "检测到CUDA版本: $cuda_version"
    else
        log_warning "未检测到NVIDIA GPU，将使用CPU模式"
    fi
    
    # 检查内存
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_mem -lt 16 ]]; then
        log_warning "系统内存较少($total_mem GB)，建议至少32GB用于大规模实验"
    else
        log_success "内存检查通过: ${total_mem}GB"
    fi
    
    # 检查磁盘空间
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -lt 100 ]]; then
        log_warning "磁盘空间不足($available_space GB)，建议至少2TB用于数据集和模型"
    else
        log_success "磁盘空间检查通过: ${available_space}GB可用"
    fi
}

# 检查并安装conda
install_conda() {
    if command -v conda &> /dev/null; then
        log_success "Conda已安装"
        return 0
    fi
    
    # 检查是否已有miniconda3目录
    if [[ -d "$HOME/miniconda3" ]]; then
        log_info "检测到现有Miniconda安装，初始化环境..."
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        conda init bash
        log_success "Miniconda环境初始化完成"
        return 0
    fi
    
    log_info "安装Miniconda..."
    
    # 下载Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # 安装Miniconda
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # 初始化conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    # 清理安装文件
    rm miniconda.sh
    
    log_success "Miniconda安装完成"
}

# 创建conda环境
create_conda_environment() {
    log_info "创建conda环境: mm_defense"
    
    # 检查环境是否已存在
    if conda env list | grep -q "mm_defense"; then
        log_warning "环境mm_defense已存在，是否重新创建? (y/n)"
        read -r response
        if [[ "$response" == "y" || "$response" == "Y" ]]; then
            conda env remove -n mm_defense -y
        else
            log_info "使用现有环境"
            return 0
        fi
    fi
    
    # 创建新环境
    conda create -n mm_defense python=3.9 -y
    
    log_success "Conda环境创建完成"
}

# 激活conda环境
activate_environment() {
    log_info "激活conda环境..."
    
    # 初始化conda（如果需要）
    if ! command -v conda &> /dev/null; then
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    fi
    
    # 激活环境
    conda activate mm_defense
    
    log_success "环境激活成功"
}

# 安装PyTorch和CUDA支持
install_pytorch() {
    log_info "安装PyTorch和相关依赖..."
    
    # 检测CUDA版本并安装对应的PyTorch
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
        
        if [[ "$cuda_version" == "11.8" ]] || [[ "$cuda_version" > "11.8" ]]; then
            log_info "安装PyTorch with CUDA 11.8支持"
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        elif [[ "$cuda_version" == "11.7" ]]; then
            log_info "安装PyTorch with CUDA 11.7支持"
            conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
        else
            log_warning "CUDA版本$cuda_version可能不完全兼容，安装CPU版本"
            conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        fi
    else
        log_info "安装PyTorch CPU版本"
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    log_success "PyTorch安装完成"
}

# 创建requirements.txt文件
create_requirements() {
    log_info "创建requirements.txt文件..."
    
    cat > requirements.txt << 'EOF'
# 核心深度学习框架
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
diffusers>=0.18.0
accelerate>=0.20.0

# CLIP和多模态模型
open-clip-torch>=2.20.0
clip-by-openai>=1.0
sentence-transformers>=2.2.0

# 计算机视觉
opencv-python>=4.8.0
Pillow>=9.5.0
albumentations>=1.3.0
imagecorruptions>=1.1.2

# 科学计算
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
wandb>=0.15.0
tensorboard>=2.13.0

# 数据处理
h5py>=3.9.0
pyarrow>=12.0.0
datasets>=2.13.0

# 网络和API
requests>=2.31.0
aiohttp>=3.8.0
fastapi>=0.100.0
uvicorn>=0.22.0

# 配置和工具
PyYAML>=6.0
omegaconf>=2.3.0
hydra-core>=1.3.0
tqdm>=4.65.0
rich>=13.4.0

# 测试和代码质量
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
flake8>=6.0.0
isort>=5.12.0
pre-commit>=3.3.0

# 性能优化
numba>=0.57.0
faiss-cpu>=1.7.4
faiss-gpu>=1.7.4

# 文本处理
nltk>=3.8.0
spacy>=3.6.0
textblob>=0.17.0

# 其他工具
psutil>=5.9.0
gpustat>=1.1.0
colorama>=0.4.6
click>=8.1.0
EOF

    log_success "requirements.txt创建完成"
}

# 安装Python依赖
install_python_dependencies() {
    log_info "安装Python依赖包..."
    
    # 升级pip
    pip install --upgrade pip setuptools wheel
    
    # 安装requirements.txt中的依赖
    pip install -r requirements.txt
    
    # 安装额外的开发工具
    pip install jupyter jupyterlab ipywidgets
    
    # 安装FAISS（根据CUDA支持选择版本）
    if command -v nvidia-smi &> /dev/null; then
        log_info "安装FAISS GPU版本"
        pip install faiss-gpu
    else
        log_info "安装FAISS CPU版本"
        pip install faiss-cpu
    fi
    
    log_success "Python依赖安装完成"
}

# 下载和配置模型
setup_models() {
    log_info "配置预训练模型..."
    
    # 创建模型目录
    mkdir -p models/clip
    mkdir -p models/stable_diffusion
    mkdir -p models/qwen
    
    # 下载CLIP模型（如果不存在）
    python3 -c "
import clip
import torch
print('下载CLIP ViT-B/32模型...')
model, preprocess = clip.load('ViT-B/32', device='cpu')
print('CLIP模型下载完成')
" || log_warning "CLIP模型下载失败，将在首次使用时自动下载"
    
    # 配置Hugging Face缓存目录
    export HF_HOME="./models/huggingface"
    mkdir -p "$HF_HOME"
    
    log_success "模型配置完成"
}

# 创建项目目录结构
create_project_structure() {
    log_info "创建项目目录结构..."
    
    # 创建主要目录
    mkdir -p src/{attacks,models,utils,evaluation}
    mkdir -p configs/{attack_configs,defense_configs}
    mkdir -p data/{coco,flickr30k,processed}
    mkdir -p experiments
    mkdir -p tests
    mkdir -p notebooks
    mkdir -p results/{figures,tables,logs}
    mkdir -p cache/{sd_references,text_variants}
    mkdir -p scripts
    
    # 创建__init__.py文件
    touch src/__init__.py
    touch src/attacks/__init__.py
    touch src/models/__init__.py
    touch src/utils/__init__.py
    touch src/evaluation/__init__.py
    
    log_success "项目目录结构创建完成"
}

# 创建配置文件
create_config_files() {
    log_info "创建配置文件..."
    
    # 创建默认配置文件
    cat > configs/default.yaml << 'EOF'
# 默认配置文件
project:
  name: "multimodal_retrieval_defense"
  version: "1.0.0"
  author: "ZHANG XIN"
  email: "zhang.xin@duke.edu"

# 模型配置
models:
  clip:
    model_name: "ViT-B/32"
    device: "cuda"
    batch_size: 256
  
  stable_diffusion:
    model_name: "runwayml/stable-diffusion-v1-5"
    device: "cuda"
    fp16: true
    guidance_scale: 7.5
    num_inference_steps: 20
  
  qwen:
    model_name: "Qwen/Qwen-7B-Chat"
    api_key: null  # 设置API密钥
    temperature: 0.7
    max_tokens: 512

# 数据配置
data:
  datasets:
    - name: "coco"
      path: "data/coco"
      split: "test"
    - name: "flickr30k"
      path: "data/flickr30k"
      split: "test"
  
  preprocessing:
    image_size: [224, 224]
    normalize: true
    augmentation: false

# 防御配置
defense:
  text_variants:
    num_variants: 10
    similarity_threshold: 0.85
    max_retries: 3
  
  retrieval:
    top_k: 20
    batch_size: 256
    similarity_metric: "cosine"
  
  sd_generation:
    images_per_variant: 3
    batch_size: 4
    enable_cache: true
    cache_dir: "cache/sd_references"
  
  detection:
    aggregation_method: "mean"
    threshold_method: "statistical"
    statistical_alpha: 1.5
    min_reference_size: 5

# 攻击配置
attacks:
  hubness:
    epsilon: 0.031372549  # 8/255
    alpha: 0.007843137    # 2/255
    iterations: 10
    lambda_balance: 0.3
    target_query_size: 100
  
  pgd:
    epsilon: 0.031372549
    alpha: 0.007843137
    iterations: 10
    random_start: true
  
  text:
    max_replace_ratio: 0.2
    alpha: 0.05
    iterations: 30

# 实验配置
experiment:
  random_seeds: [42, 123, 456, 789, 999]
  num_runs: 5
  batch_size: 32
  save_intermediate: false
  
  evaluation:
    metrics: ["recall@1", "recall@5", "recall@10", "map", "auc"]
    top_k_values: [1, 5, 10, 20]

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "results/logs/experiment.log"
  console: true

# 性能配置
performance:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  enable_amp: true  # 自动混合精度
  compile_model: false  # PyTorch 2.0编译
EOF

    log_success "配置文件创建完成"
}

# 创建环境变量文件
create_env_file() {
    log_info "创建环境变量文件..."
    
    cat > .env << 'EOF'
# 环境变量配置

# 项目路径
PROJECT_ROOT=/home/cw/zhangxin_workspace/多模态检测一致性实验代码

# 数据路径
DATA_ROOT=${PROJECT_ROOT}/data
CACHE_ROOT=${PROJECT_ROOT}/cache
RESULTS_ROOT=${PROJECT_ROOT}/results

# 模型配置
HF_HOME=${PROJECT_ROOT}/models/huggingface
TORCH_HOME=${PROJECT_ROOT}/models/torch
CLIP_CACHE=${PROJECT_ROOT}/models/clip

# CUDA配置
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 根据实际GPU数量调整
CUDA_LAUNCH_BLOCKING=1

# 性能优化
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
NUMBA_CACHE_DIR=${PROJECT_ROOT}/cache/numba

# API配置（需要用户填写）
QWEN_API_KEY=your_qwen_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# 日志配置
LOG_LEVEL=INFO
WANDB_PROJECT=multimodal_retrieval_defense
WANDB_ENTITY=your_wandb_entity_here
EOF

    log_success "环境变量文件创建完成"
}

# 安装项目包
install_project_package() {
    log_info "安装项目包..."
    
    # 创建setup.py文件
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-retrieval-defense",
    version="1.0.0",
    author="ZHANG XIN",
    author_email="zhang.xin@duke.edu",
    description="Multi-Modal Retrieval Defense via Text Variant Consistency Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangxin-duke/multimodal-retrieval-defense",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mm-defense=src.cli:main",
        ],
    },
)
EOF

    # 以开发模式安装项目
    pip install -e .
    
    log_success "项目包安装完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 测试Python导入
    python3 -c "
import torch
import torchvision
import transformers
import diffusers
import clip
import numpy as np
import matplotlib.pyplot as plt
print('所有核心依赖导入成功')
" || {
        log_error "依赖导入失败"
        exit 1
    }
    
    # 测试CUDA（如果可用）
    if command -v nvidia-smi &> /dev/null; then
        python3 -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA设备数量: {torch.cuda.device_count()}')
    print(f'当前设备: {torch.cuda.current_device()}')
    print(f'设备名称: {torch.cuda.get_device_name()}')
" || log_warning "CUDA测试失败"
    fi
    
    # 检查项目结构
    if [[ -d "src" && -d "configs" && -d "data" ]]; then
        log_success "项目结构验证通过"
    else
        log_error "项目结构不完整"
        exit 1
    fi
    
    log_success "安装验证完成"
}

# 创建快速测试脚本
create_test_script() {
    log_info "创建快速测试脚本..."
    
    cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
安装验证测试脚本
"""

import sys
import torch
import clip
import numpy as np
from pathlib import Path

def test_basic_imports():
    """测试基础导入"""
    try:
        import torch
        import torchvision
        import transformers
        import diffusers
        import clip
        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn
        import pandas as pd
        print("✓ 所有基础依赖导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_cuda_availability():
    """测试CUDA可用性"""
    if torch.cuda.is_available():
        print(f"✓ CUDA可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("⚠ CUDA不可用，将使用CPU模式")
        return False

def test_clip_model():
    """测试CLIP模型加载"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"✓ CLIP模型加载成功，设备: {device}")
        return True
    except Exception as e:
        print(f"✗ CLIP模型加载失败: {e}")
        return False

def test_project_structure():
    """测试项目结构"""
    required_dirs = [
        "src", "configs", "data", "experiments", 
        "tests", "notebooks", "results", "cache"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"✗ 缺少目录: {missing_dirs}")
        return False
    else:
        print("✓ 项目结构完整")
        return True

def main():
    """主测试函数"""
    print("=" * 50)
    print("多模态检测一致性实验代码 - 安装验证")
    print("=" * 50)
    
    tests = [
        ("基础依赖导入", test_basic_imports),
        ("CUDA可用性", test_cuda_availability),
        ("CLIP模型加载", test_clip_model),
        ("项目结构", test_project_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n测试: {test_name}")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境配置成功")
        return 0
    else:
        print("❌ 部分测试失败，请检查安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x test_installation.py
    
    log_success "测试脚本创建完成"
}

# 打印使用说明
print_usage_instructions() {
    log_info "安装完成！使用说明："
    
    echo -e ""
echo -e "${GREEN}=== 环境激活 ===${NC}"
echo -e "conda activate mm_defense"
echo -e ""
echo -e "${GREEN}=== 快速测试 ===${NC}"
echo -e "python test_installation.py"
echo -e ""
echo -e "${GREEN}=== 配置API密钥 ===${NC}"
echo -e "编辑 .env 文件，设置以下API密钥："
echo -e "- QWEN_API_KEY: Qwen大模型API密钥"
echo -e "- OPENAI_API_KEY: OpenAI API密钥（可选）"
echo -e "- HUGGINGFACE_TOKEN: Hugging Face访问令牌"
echo -e ""
echo -e "${GREEN}=== 下载数据集 ===${NC}"
echo -e "python scripts/download_coco.py"
echo -e "python scripts/download_flickr30k.py"
echo -e ""
echo -e "${GREEN}=== 运行实验 ===${NC}"
echo -e "python experiments/run_experiments.py --config configs/default.yaml"
echo -e ""
echo -e "${GREEN}=== 开发模式 ===${NC}"
echo -e "jupyter lab  # 启动Jupyter Lab"
echo -e "pytest tests/  # 运行单元测试"
echo -e ""
echo -e "${YELLOW}注意事项：${NC}"
echo -e "1. 首次运行会自动下载预训练模型，需要网络连接"
echo -e "2. 大规模实验需要充足的GPU内存和存储空间"
echo -e "3. 建议在运行实验前先执行快速测试验证环境"
echo -e ""
}

# 主函数
main() {
    echo -e "${BLUE}"
    echo "================================================"
    echo "  多模态检测一致性实验代码 - 环境安装脚本"
    echo "  Author: ZHANG XIN <zhang.xin@duke.edu>"
    echo "  Duke University"
    echo "================================================"
    echo -e "${NC}"
    
    # 执行安装步骤
    check_system_requirements
    install_conda
    create_conda_environment
    
    # 激活环境并继续安装
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate mm_defense
    
    install_pytorch
    create_requirements
    install_python_dependencies
    setup_models
    create_project_structure
    create_config_files
    create_env_file
    install_project_package
    create_test_script
    verify_installation
    
    log_success "环境安装完成！"
    print_usage_instructions
}

# 错误处理
trap 'log_error "安装过程中发生错误，请检查日志"' ERR

# 运行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi