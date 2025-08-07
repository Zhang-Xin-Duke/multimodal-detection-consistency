#!/usr/bin/env python3
"""
GitHub推送准备脚本
清理项目，删除测试文件和不必要的文件，只保留实验相关的核心文件

作者: 张昕 (ZHANG XIN)
邮箱: zhang.xin@duke.edu
学校: Duke University
"""

import os
import shutil
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_project(project_root: str):
    """
    清理项目，删除不必要的文件
    
    Args:
        project_root: 项目根目录
    """
    root = Path(project_root)
    
    # 要删除的文件和目录列表
    files_to_remove = [
        # 测试文件
        "test_dataset_loading.py",
        "experiment_log.txt",
        
        # 备份文件
        "src/utils/data_loader.py.backup",
        
        # 临时文件和缓存
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        
        # IDE文件
        ".vscode",
        ".idea",
        "*.swp",
        "*.swo",
        "*~",
        
        # 系统文件
        ".DS_Store",
        "Thumbs.db",
        
        # 日志文件
        "*.log",
        "logs/",
        
        # 临时实验结果（保留results目录结构但清空内容）
        "experiment_figures/",
    ]
    
    # 要保留的数据集示例文件（小样本用于演示）
    keep_sample_data = True
    
    logger.info(f"开始清理项目: {root}")
    
    removed_count = 0
    
    for pattern in files_to_remove:
        if pattern.endswith('/'):
            # 目录
            dir_path = root / pattern.rstrip('/')
            if dir_path.exists() and dir_path.is_dir():
                shutil.rmtree(dir_path)
                logger.info(f"删除目录: {dir_path}")
                removed_count += 1
        elif '*' in pattern:
            # 通配符模式
            for file_path in root.rglob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"删除文件: {file_path}")
                    removed_count += 1
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    logger.info(f"删除目录: {file_path}")
                    removed_count += 1
        else:
            # 具体文件
            file_path = root / pattern
            if file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"删除文件: {file_path}")
                    removed_count += 1
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    logger.info(f"删除目录: {file_path}")
                    removed_count += 1
    
    # 清理数据目录，但保留结构和小样本数据
    data_dir = root / "data"
    if data_dir.exists():
        logger.info("清理数据目录...")
        
        # 保留processed目录结构但清空内容
        processed_dir = data_dir / "processed"
        if processed_dir.exists():
            for subdir in processed_dir.iterdir():
                if subdir.is_dir():
                    # 保留目录结构，但清空内容
                    for item in subdir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    logger.info(f"清空processed子目录: {subdir}")
        
        # 对于raw目录，保留小样本数据用于演示
        raw_dir = data_dir / "raw"
        if raw_dir.exists() and keep_sample_data:
            logger.info("保留小样本数据用于演示...")
            
            # 限制每个数据集的样本数量
            dataset_limits = {
                "coco": {"images": 10, "annotations": True},
                "flickr30k": {"images": 10, "annotations": True},
                "cc3m": {"images": 10, "annotations": True},
                "visual_genome": {"images": 10, "annotations": True}
            }
            
            for dataset_name, limits in dataset_limits.items():
                dataset_dir = raw_dir / dataset_name
                if dataset_dir.exists():
                    # 限制图像数量
                    images_dirs = ["images", "train2017", "val2017", "flickr30k_images", "VG_100K"]
                    for img_dir_name in images_dirs:
                        img_dir = dataset_dir / img_dir_name
                        if img_dir.exists():
                            image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                            if len(image_files) > limits["images"]:
                                # 保留前N个文件，删除其余的
                                for img_file in image_files[limits["images"]:]:
                                    img_file.unlink()
                                logger.info(f"限制{dataset_name}/{img_dir_name}图像数量为{limits['images']}")
    
    logger.info(f"清理完成，共删除 {removed_count} 个文件/目录")

def create_gitignore(project_root: str):
    """
    创建或更新.gitignore文件
    
    Args:
        project_root: 项目根目录
    """
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
PIPFILE.lock

# PyTorch
*.pth
*.pt

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Data (keep structure but ignore large files)
data/raw/*/images/
data/raw/*/train2017/
data/raw/*/val2017/
data/raw/*/test2017/
data/raw/*/flickr30k_images/
data/raw/*/VG_100K/
data/raw/*/VG_100K_2/
data/processed/

# Results
results/
experiment_figures/
*.png
*.jpg
*.jpeg
*.pdf

# Temporary files
temp/
tmp/
*.tmp
*.bak
*.backup

# Model checkpoints
checkpoints/
models/
*.ckpt

# Weights & Biases
wandb/

# MLflow
mlruns/

# Test files
test_*.py
*_test.py
tests/temp/

# Coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/
"""
    
    gitignore_path = Path(project_root) / ".gitignore"
    
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write(gitignore_content.strip())
    
    logger.info(f"已创建/更新 .gitignore 文件: {gitignore_path}")

def create_readme_update(project_root: str):
    """
    更新README文件，添加数据集配置说明
    
    Args:
        project_root: 项目根目录
    """
    readme_path = Path(project_root) / "README.md"
    
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 添加数据集配置说明
        dataset_section = """

## 数据集配置

本项目支持以下数据集：

### 主要数据集（已验证）
- **MS COCO**: 图像-文本检索基准数据集
- **Flickr30K**: 多模态检索标准数据集

### 扩展数据集（可选）
- **Conceptual Captions (CC3M)**: 大规模图像描述数据集
- **Visual Genome**: 密集标注的视觉理解数据集

### 数据集准备

1. 下载数据集到 `data/raw/` 目录
2. 运行数据集修复脚本（如需要）：
   ```bash
   python scripts/fix_datasets.py
   ```
3. 验证数据集加载：
   ```bash
   python -c "from src.utils.data_loader import DataLoaderManager; print('数据集加载正常')"
   ```

### 目录结构
```
data/
├── raw/
│   ├── coco/
│   │   ├── annotations/
│   │   ├── train2017/
│   │   └── val2017/
│   ├── flickr30k/
│   │   ├── flickr30k_images/
│   │   └── results_20130124.token
│   ├── cc3m/
│   │   ├── images/
│   │   └── cc3m_annotations.tsv
│   └── visual_genome/
│       ├── images/
│       ├── region_descriptions.json
│       └── image_data.json
└── processed/
    └── (自动生成的处理后数据)
```
"""
        
        # 如果README中没有数据集配置说明，则添加
        if "## 数据集配置" not in content:
            content += dataset_section
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("已更新README文件，添加数据集配置说明")
        else:
            logger.info("README文件已包含数据集配置说明")

def main():
    """
    主函数
    """
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    logger.info(f"准备项目推送到GitHub: {project_root}")
    
    # 清理项目
    clean_project(str(project_root))
    
    # 创建/更新.gitignore
    create_gitignore(str(project_root))
    
    # 更新README
    create_readme_update(str(project_root))
    
    logger.info("🎉 项目清理完成，可以推送到GitHub了！")
    
    # 提供Git命令建议
    logger.info("")
    logger.info("建议的Git命令：")
    logger.info("git add .")
    logger.info("git commit -m 'feat: 完成多模态检索对抗防御系统核心实现'")
    logger.info("git push origin main")
    logger.info("")
    logger.info("注意：请确保已经配置了正确的Git远程仓库")

if __name__ == "__main__":
    main()