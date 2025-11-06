"""Module MIL pour l'apprentissage multi-instance."""

# Import des fonctions utilitaires de base
from .util import (
    getLoggingLevel,
    log, 
    prepare_multimodal_mixed_bags
)

# Import des composants principaux
from .project import Project
from .dataset import Dataset
from .mil import (
    mil_config,
    train_mil
)

# Version du module
__version__ = '0.1.0'

# Exposition des classes et fonctions principales
__all__ = [
    'Project',
    'Dataset',
    'mil_config',
    'build_model',
    'train_mil',
    'prepare_multimodal_mixed_bags',
    'log',
    'getLoggingLevel',
    'setLoggingLevel'
]