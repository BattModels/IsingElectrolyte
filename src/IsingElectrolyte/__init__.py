"""
IsingElectrolyte: a differentiable mean-field Ising model of Li+ solvation shell composition.

Paper: Zhao, Lin, Kelly & Viswanathan, "Differentiable Solvation Shell Model for
Rational Electrolyte Design", arXiv:2609.05671 (2026).
"""

__all__ = [
    "interactions",
    "conc_vol_correction",
    "model",
    "train",
    "pretrained",
    "analysis",
]
__version__ = "0.1.0"
