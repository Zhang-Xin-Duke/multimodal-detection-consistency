"""攻击模块

实现各种对抗性攻击方法。
"""

from .hubness_attack import (
    HubnessAttackConfig,
    HubnessAttack as HubnessAttacker,
    HubnessAttackPresets,
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

from .fgsm_attack import (
    FGSMAttackConfig,
    FGSMAttacker,
    FGSMAttackPresets,
    create_fgsm_attacker
)

from .cw_attack import (
    CWAttackConfig,
    CWAttacker,
    CWAttackPresets,
    create_cw_attacker
)

from .fsta_attack import (
    FSTAAttacker,
    FSTAAttackConfig,
    FSTAAttackPresets,
    create_fsta_attacker
)

from .sma_attack import (
    SMAAttacker,
    SMAAttackConfig,
    SMAAttackPresets,
    create_sma_attacker
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
    'create_text_attacker',
    
    # FGSM攻击
    'FGSMAttackConfig',
    'FGSMAttacker',
    'FGSMAttackPresets',
    'create_fgsm_attacker',
    
    # C&W攻击
    'CWAttackConfig',
    'CWAttacker',
    'CWAttackPresets',
    'create_cw_attacker',
    
    # FSTA攻击
    'FSTAAttacker',
    'FSTAAttackConfig',
    'FSTAAttackPresets',
    'create_fsta_attacker',
    
    # SMA攻击
    'SMAAttacker',
    'SMAAttackConfig',
    'SMAAttackPresets',
    'create_sma_attacker'
]