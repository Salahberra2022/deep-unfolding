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
    "IterativeModel",
    "evaluate_model",
    "gen_linear",
    "GaussSeidel",
    "Jacobi",
    "Richardson",
    "RichardsonNet",
    "SOR",
    "SORCheby",
    "SORChebyNet",
    "SORNet",
]

from deep_unfolding.iterative_solvers import (AOR, SOR, AORCheby, GaussSeidel,
                                              IterativeModel, Jacobi,
                                              Richardson, SORCheby)
from deep_unfolding.unfolding_solvers import (AORNet, RichardsonNet,
                                              SORChebyNet, SORNet)
from deep_unfolding.utils import gen_linear
