# Copyright (c) 2022-2023 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file LICENSE or copy
# at https://www.gnu.org/licenses/)

__author__ = "Salah Berra and contributors"
__copyright__ = "Copyright 2022-2023"
__license__ = "GNU GPL"
__version__ = "0.1.0"
__maintainer__ = "Salah Berra / Mohamed Nennouche"
__email__ = "."
__status__ = "dev"

# Note that each Alghorithm import from method module and run in the main code Iterative.
from .methods import device, generate_A_H_sol, decompose_matrix
from .train_methods import train_model, evaluate_model, SORNet, SOR_CHEBY_Net, AORNet, RINet
from .utils import model_iterations, GS, RI, Jacobi, SOR, SOR_CHEBY, AOR, AOR_CHEBY