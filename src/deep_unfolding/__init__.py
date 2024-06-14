# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file LICENSE or copy
# at https://www.gnu.org/licenses/)

"""
.. include:: ../../README.md
"""

__docformat__ = "google"

__all__ = [
    "AOR",
    "AOR_CHEBY",
    "AORNet",
    "base_model",
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
    "SOR_CHEBY",
    "SOR_CHEBY_Net",
    "SORNet",
    "train_model",
]

from deep_unfolding.methods import (
    AOR,
    AOR_CHEBY,
    GS,
    RI,
    SOR,
    SOR_CHEBY,
    Jacobi,
    base_model,
    model_iterations,
)
from deep_unfolding.train_methods import (
    AORNet,
    RINet,
    SOR_CHEBY_Net,
    SORNet,
    evaluate_model,
    train_model,
)
from deep_unfolding.utils import decompose_matrix, device, generate_A_H_sol
