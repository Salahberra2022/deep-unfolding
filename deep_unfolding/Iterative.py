import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from methods import itr_list, norm_list_GS, norm_list_RI, norm_list_Jacobi, norm_list_SOR, norm_list_AOR, norm_list_AOR_CHEBY, norm_list_SOR_CHEBY
device = torch.device('cpu') # 'cpu' or 'cuda'
list_iterative = ['RI','AOR','SOR']
def result(list_iterative):
    norm_list = np.zeros((len(list_iterative), len(itr_list)))

    for k,iterative in enumerate(list_iterative):
        if iterative == 'GS': 
           norm_list[k]=norm_list_GS
        elif iterative == 'RI':
             norm_list[k]=norm_list_RI
        elif iterative == 'Jacobi':
             norm_list[k]=norm_list_Jacobi
        elif iterative == 'SOR':
             norm_list[k]=norm_list_SOR
        elif iterative == 'ChebySOR':
             norm_list[k]=norm_list_SOR_CHEBY
        elif iterative == 'AOR':
             norm_list[k]=norm_list_AOR
        elif iterative == 'ChebyAOR': 
             norm_list[k]=norm_list_AOR_CHEBY

    marker_list = ['^k-','sb-', 'og-', 'Xc-', 'hk-','sk-','^c-', 'sr-','^r:', '^b:', 'ok:', 'sg:', '^y:' ]
    for k,iterative in enumerate(list_iterative):
        plt.semilogy(itr_list, norm_list[k],marker_list[k],label =iterative)
 
    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.show()
#result()
