# 攻击方法论文和参考资料

本文件夹包含了项目中实现的攻击方法的相关论文、代码和笔记。

## 文件夹结构

```
papers_and_references/
├── README.md                           # 本文件
├── FGSM_2015_Goodfellow/              # FGSM攻击相关资料
│   ├── paper/                         # 论文文件
│   ├── code/                          # 参考代码实现
│   └── notes/                         # 研究笔记
└── CW_2017_Carlini_Wagner/            # C&W攻击相关资料
    ├── paper/                         # 论文文件
    ├── code/                          # 参考代码实现
    └── notes/                         # 研究笔记
```

## 论文信息

### 1. FGSM攻击 (Fast Gradient Sign Method)

**论文标题**: "Explaining and Harnessing Adversarial Examples"  
**作者**: Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy  
**发表会议**: ICLR 2015  
**论文链接**: https://arxiv.org/abs/1412.6572  

**核心贡献**:
- 提出了快速梯度符号方法(FGSM)生成对抗样本
- 解释了神经网络对对抗样本脆弱性的线性假设
- 提出了对抗训练作为防御方法

**核心公式**:
```
x_adv = x + ε * sign(∇_x J(θ, x, y))
```
其中:
- x: 原始输入
- ε: 扰动幅度
- J: 损失函数
- θ: 模型参数

### 2. C&W攻击 (Carlini & Wagner)

**论文标题**: "Towards Evaluating the Robustness of Neural Networks"  
**作者**: Nicholas Carlini, David Wagner  
**发表会议**: IEEE Symposium on Security and Privacy (S&P) 2017  
**论文链接**: https://arxiv.org/abs/1608.04644  

**核心贡献**:
- 提出了基于优化的C&W攻击方法
- 证明了防御蒸馏等防御方法的无效性
- 设计了更强的攻击评估框架

**核心优化目标**:
```
minimize ||δ||_p + c * f(x + δ)
```
其中:
- δ: 扰动向量
- c: 平衡参数
- f: 攻击目标函数
- p: Lp范数(通常p=2)

## 实现状态

- ✅ **FGSM攻击**: 已在 `src/attacks/fgsm_attack.py` 中实现
- ✅ **C&W攻击**: 已在 `src/attacks/cw_attack.py` 中实现

## 使用说明

参考 `examples/attack_methods_demo.py` 查看如何使用这些攻击方法。

## 相关资源

### FGSM相关
- [原始论文实现](https://github.com/tensorflow/cleverhans)
- [PyTorch实现参考](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

### C&W相关
- [作者官方实现](https://github.com/carlini/nn_robust_attacks)
- [ART库实现](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

---

**维护者**: 张昕 (zhang.xin@duke.edu)  
**最后更新**: 2025年8月