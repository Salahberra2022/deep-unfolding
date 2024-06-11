# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file LICENSE or copy
# at https://www.gnu.org/licenses/)

"""
.. include:: ../../README.md
"""

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

from unfolding_linear.methods import (
    AOR,
    AOR_CHEBY,
    base_model,
    GS,
    Jacobi,
    model_iterations,
    RI,
    SOR,
    SOR_CHEBY,
)
from unfolding_linear.train_methods import (
    AORNet,
    evaluate_model,
    RINet,
    SORNet,
    SOR_CHEBY_Net,
    train_model,
)
from unfolding_linear.utils import (
    decompose_matrix,
    device,
    generate_A_H_sol,
)