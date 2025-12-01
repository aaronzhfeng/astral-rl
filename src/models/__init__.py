# src/models/__init__.py
"""
ASTRAL Model Components.

- AbstractionBank: K learnable abstraction slots with gating
- TemperatureScheduler: Anneal temperature from high to low
- FiLM: Feature-wise Linear Modulation
- ASTRALAgent: Full agent with GRU + abstractions + FiLM
- BaselineAgent: GRU-only baseline for comparison
"""

from .abstraction_bank import AbstractionBank, TemperatureScheduler
from .film import FiLM
from .astral_agent import ASTRALAgent, BaselineAgent, count_parameters

__all__ = [
    "AbstractionBank",
    "TemperatureScheduler",
    "FiLM",
    "ASTRALAgent",
    "BaselineAgent",
    "count_parameters",
]
