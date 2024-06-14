# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file
# LICENSE or copy at https://www.gnu.org/licenses/)

"""
.. include:: ../../README.md
"""

__docformat__ = "google"

__all__ = [
    "AOR",
    "AORCheby",
    "AORNet",
    "BaseModel",
    "decompose_matrix",
    "device",
    "evaluate_model",
    "generate_A_H_sol",
    "GS",
    "Jacobi",
    "model_iterations",
    "RI",
    "RINet",
    "SOR",
    "SORCheby",
    "SORChebyNet",
    "SORNet",
    "train_model",
]

from deep_unfolding.methods import (
    AOR,
    GS,
    RI,
    SOR,
    AORCheby,
    BaseModel,
    Jacobi,
    SORCheby,
    model_iterations,
)
from deep_unfolding.train_methods import (
    AORNet,
    RINet,
    SORChebyNet,
    SORNet,
    evaluate_model,
    train_model,
)
from deep_unfolding.utils import decompose_matrix, device, generate_A_H_sol
