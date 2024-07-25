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
    "_decompose_matrix",
    "device",
    "evaluate_model",
    "generate_A_H_sol",
    "GaussSeidel",
    "Jacobi",
    "model_iterations",
    "Richardson",
    "RichardsonNet",
    "SOR",
    "SORCheby",
    "SORChebyNet",
    "SORNet",
    "train_model",
]

from deep_unfolding.iterative_solvers import (
    AOR,
    GaussSeidel,
    Richardson,
    SOR,
    AORCheby,
    BaseModel,
    Jacobi,
    SORCheby,
    model_iterations,
)
from deep_unfolding.unfolding_solvers import (
    AORNet,
    RichardsonNet,
    SORChebyNet,
    SORNet,
    evaluate_model,
    train_model,
)
from deep_unfolding.utils import _decompose_matrix, device, gen_linear
