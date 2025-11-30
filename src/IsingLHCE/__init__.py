"""
IsingLHCE: A Python package fitting an Ising model to design LHCE.
"""

from .fit_model import fit_model
from .functions import functions
from .conc_factor_func import conc_factor_func
from .analysis import analysis

__all__ = [
    "fit_model",
    "functions",
    "conc_factor_func",
    "analysis",
]
__version__ = "0.1.0"