import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
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

    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.show()
#main ()    
