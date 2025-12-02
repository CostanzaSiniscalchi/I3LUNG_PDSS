from enum import Enum

class Mode(Enum):
    RWD = 'RWD'
    DP = 'DP'
    FMRAD = 'FMRAD'
    PYRAD = 'PYRAD'
    GEN = 'GEN'

class Outcome(Enum):
    OS_6 = 'OS_6'
    OS_24 = 'OS_24'
    DCR = 'DCR'

class Subanalysis(Enum):
    C23 = 'C23'
    C2 = 'C2'
    IO_ONLY = 'IO_ONLY'
    IO_CHT = 'IOCHT'
    LOW_PDL1 = 'LOW_PDL1'
    HIGH_PDL1 = 'HIGH_PDL1'
    SQUAMOUS = 'SQUAMOUS'
    ADENOCARCINOMA = 'ADENOCARCINOMA'
    INT = 'INT'

class Model(Enum):
    LR = 'LR'
    RF = 'RF'