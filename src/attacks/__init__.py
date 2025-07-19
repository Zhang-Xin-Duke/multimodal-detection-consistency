"""攻击模块

实现各种对抗性攻击方法。
"""

from .hubness_attack import (
    HubnessAttackConfig,
    HubnessAttacker,
    AdaptiveHubnessAttacker,
    create_hubness_attacker
)

from .pgd_attack import (
    PGDAttackConfig,
    PGDAttacker,
    create_pgd_attacker
)

from .text_attack import (
    TextAttackConfig,
    TextAttacker,
    create_text_attacker
)

__all__ = [
    # Hubness攻击
    'HubnessAttackConfig',
    'HubnessAttacker', 
    'AdaptiveHubnessAttacker',
    'create_hubness_attacker',
    
    # PGD图像攻击
    'PGDAttackConfig',
    'PGDAttacker',
    'create_pgd_attacker',
    
    # 文本攻击
    'TextAttackConfig',
    'TextAttacker',
    'create_text_attacker'
]