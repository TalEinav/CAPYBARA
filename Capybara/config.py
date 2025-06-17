#config.py
from dataclasses import dataclass

@dataclass
class Config:
    # Data conventions
    DATA_IN_LOG2: bool = True          # every value is log₂(HAI/5)

    # Model hyper-params
    DIAG_T: float  = 0.1
    BANDWIDTH: int = 20
    REG: float     = 1.0

    # CV settings
    SPLIT_FRAC: float = 0.2
    N_SPLITS: int     = 5
    RNG_SEED: int     = 42
    
