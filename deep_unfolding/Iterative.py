# Copyright (c) 2022-2023 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file LICENSE or copy
# at https://www.gnu.org/licenses/)

"""This module contains the core functions."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from .methods import itr_list, norm_list_GS, norm_list_RI, norm_list_Jacobi, norm_list_SOR, norm_list_AOR, norm_list_AOR_CHEBY, norm_list_SOR_CHEBY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU, if not CPU
print(f"Code run on : {device}")

# Name and model correspondance for use
methods_dict = {
    "GS" : norm_list_GS,
    "RI" : norm_list_RI, 
    "Jacobi" : norm_list_Jacobi,
    "SOR" : norm_list_SOR,
    "ChebySOR" : norm_list_SOR_CHEBY,
    "AOR" : norm_list_AOR,
    "ChebyAOR" : norm_list_AOR_CHEBY
}

# for visualization
marker_list = ['^k-', 'sb-', 'og-', 'Xc-', 'hk-', 'sk-', '^c-', 'sr-', '^r:', '^b:', 'ok:', 'sg:', '^y:']

def main(list_iterative):
    """Plot the MSE for different iterative methods.

    Args:
        list_iterative (list): List of iterative methods.

    """

    norm_list = np.zeros((len(list_iterative), len(itr_list)))

    for k, iterative in enumerate(list_iterative):
        norm_list[k] = methods_dict[iterative]

    # visualization
    for k, iterative in enumerate(list_iterative):
        plt.semilogy(itr_list, norm_list[k], marker_list[k], label=iterative)

    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.show()