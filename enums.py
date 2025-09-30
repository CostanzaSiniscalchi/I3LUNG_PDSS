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

class Subanalysis(Enum):
    CLASSIC = 'CLASSIC'
    IO_ONLY = 'IO_ONLY'
    IO_CT = 'IOCT'
    LOW_PDL1 = 'LOW_PDL1'
    HIGH_PDL1 = 'HIGH_PDL1'
    SQUAMOUS = 'SQUAMOUS'
    ADENOCARCINOMA = 'ADENOCARCINOMA'

class Model(Enum):
    LR = 'LR'
    RF = 'RF'
    XGB = 'XGB'