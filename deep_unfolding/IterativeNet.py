<<<<<<< HEAD
# Copyright (c) 2022-2023 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file LICENSE or copy
# at https://www.gnu.org/licenses/)

=======
>>>>>>> 49129962c65b00396d7c679ff04f60b715fb28ac
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
<<<<<<< HEAD


device = torch.device('cpu')  # 'cpu' or 'cuda'


def main(list_iterative):
    """Main function to plot the MSE for different iterative methods.

    Args:
        list_iterative (list): List of iterative methods.

    """
    from methods import itr_list, norm_list_GS, norm_list_RI, norm_list_Jacobi, norm_list_SOR, norm_list_AOR, norm_list_AOR_CHEBY, norm_list_SOR_CHEBY
    from train_methods import itr_list, norm_list_SorNet, norm_list_Sor_Cheby_Net, norm_list_AorNet, norm_list_rinet
    norm_list = np.zeros((len(list_iterative), len(itr_list)))
    
    for k, iterative in enumerate(list_iterative):
        if iterative == 'SORNet':
            norm_list[k] = norm_list_SorNet
        elif iterative == 'ChebySORNet':
            norm_list[k] = norm_list_Sor_Cheby_Net
        elif iterative == 'AORNet':
            norm_list[k] = norm_list_AorNet
        elif iterative == 'RINet':
            norm_list[k] = norm_list_rinet
        elif iterative == 'RI':
            norm_list[k] = norm_list_RI
        elif iterative == 'GS':
            norm_list[k] = norm_list_GS
        elif iterative == 'SOR':
            norm_list[k] = norm_list_SOR

    marker_list = ['^k-', 'sb-', 'og-', 'Xc-', 'hk-', 'sk-', '^c-', 'sr-', '^r:', '^b:', 'ok:', 'sg:', '^y:']

    for k, iterative in enumerate(list_iterative):
        plt.semilogy(itr_list, norm_list[k], marker_list[k], label=iterative)
=======
from train_methods import itr_list, norm_list_SorNet, norm_list_Sor_Cheby_Net, norm_list_AorNet, norm_list_rinet 
device = torch.device('cpu') # 'cpu' or 'cuda'
list_iterative = [ 'SORNet', 'ChebySORNet','AORNet','RINet']
def main(list_iterative):
    norm_list = np.zeros((len(list_iterative), len(itr_list)))
    
    for k,iterative in enumerate(list_iterative):
        if iterative == 'SORNet': 
           norm_list[k]=norm_list_SorNet
        elif iterative == 'ChebySORNet': 
             norm_list[k]=norm_list_Sor_Cheby_Net
        elif iterative == 'AORNet': 
             norm_list[k]=norm_list_AorNet
        elif iterative == 'RINet': 
             norm_list[k]=norm_list_rinet
   
    marker_list = ['^k-','sb-', 'og-', 'Xc-', 'hk-','sk-','^c-', 'sr-','^r:', '^b:', 'ok:', 'sg:', '^y:' ]

    for k,iterative in enumerate(list_iterative):
        plt.semilogy(itr_list, norm_list[k],marker_list[k],label =iterative)
>>>>>>> 49129962c65b00396d7c679ff04f60b715fb28ac

    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.show()
<<<<<<< HEAD


=======
#main ()    
>>>>>>> 49129962c65b00396d7c679ff04f60b715fb28ac
