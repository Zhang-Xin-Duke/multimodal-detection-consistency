#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态检测一致性实验代码 - 安装脚本
Author: ZHANG XIN <zhang.xin@duke.edu>
Duke University
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements文件
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# 项目信息
setup(
    name="multi-modal-retrieval-defense",
    version="1.0.0",
    author="ZHANG XIN",
    author_email="zhang.xin@duke.edu",
    description="Multi-Modal Retrieval Defense via Text Variant Consistency Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangxin-duke/multi-modal-retrieval-defense",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.24.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mm-defense=core.pipeline:main",
            "mm-attack=attacks.hubness_attack:main",
            "mm-eval=utils.evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "multi-modal",
        "retrieval",
        "defense",
        "adversarial",
        "consistency",
        "text-variant",
        "computer-vision",
        "natural-language-processing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/zhangxin-duke/multi-modal-retrieval-defense/issues",
        "Source": "https://github.com/zhangxin-duke/multi-modal-retrieval-defense",
        "Documentation": "https://multi-modal-retrieval-defense.readthedocs.io/",
    },
)