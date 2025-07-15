"""
PyGEP: Interface for GeneExpressionProgramming.jl

This wrapper to instantiate the subsequent mentioned objects
"""

__version__ = "0.1.0"

from .core.regressor import GepRegressor, GepTensorRegressor
from .core.julia_interface import install_julia_dependencies, check_julia_installation

__all__ = [
    "GepRegressor", 
    "GepTensorRegressor",
    "install_julia_dependencies",
    "check_julia_installation"
]

