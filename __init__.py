# modules/__init__.py
"""
Data Tool - Modules principaux
Outil d'analyse de données et de modélisation ML
"""

__version__ = "1.0.0"
__author__ = "Data Tool Team"

# Imports pour faciliter l'accès aux modules
from . import data_loader
from . import eda
from . import preprocessing
from . import modeling
from . import evaluation
from . import reporting

__all__ = [
    "data_loader",
    "eda",
    "preprocessing",
    "modeling",
    "evaluation",
    "reporting"
]
