# Copyright (c) 2022-2023 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file LICENSE or copy
# at https://www.gnu.org/licenses/)

"""This module contains the core functions."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

device = torch.device('cpu')  # 'cpu' or 'cuda'


def main(list_iterative):
    """Plot the MSE for different iterative methods.

    Args:
        list_iterative (list): List of iterative methods.

    """
    from .methods import itr_list, norm_list_GS, norm_list_RI, norm_list_Jacobi, norm_list_SOR, norm_list_AOR, norm_list_AOR_CHEBY, norm_list_SOR_CHEBY

    norm_list = np.zeros((len(list_iterative), len(itr_list)))

    for k, iterative in enumerate(list_iterative):
        if iterative == 'GS':
            norm_list[k] = norm_list_GS
        elif iterative == 'RI':
            norm_list[k] = norm_list_RI
        elif iterative == 'Jacobi':
            norm_list[k] = norm_list_Jacobi
        elif iterative == 'SOR':
            norm_list[k] = norm_list_SOR
        elif iterative == 'ChebySOR':
            norm_list[k] = norm_list_SOR_CHEBY
        elif iterative == 'AOR':
            norm_list[k] = norm_list_AOR
        elif iterative == 'ChebyAOR':
            norm_list[k] = norm_list_AOR_CHEBY

    marker_list = ['^k-', 'sb-', 'og-', 'Xc-', 'hk-', 'sk-', '^c-', 'sr-', '^r:', '^b:', 'ok:', 'sg:', '^y:']
    for k, iterative in enumerate(list_iterative):
        plt.semilogy(itr_list, norm_list[k], marker_list[k], label=iterative)

    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.show()
