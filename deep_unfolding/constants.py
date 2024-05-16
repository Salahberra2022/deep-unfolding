# Script to englobe all constants used in the different scripts

from methods import norm_list_GS, norm_list_RI, norm_list_Jacobi, norm_list_SOR, norm_list_AOR, norm_list_AOR_CHEBY, norm_list_SOR_CHEBY
from train_methods import norm_list_SorNet, norm_list_Sor_Cheby_Net, norm_list_AorNet, norm_list_rinet

methods_dict = {
    "GS" : norm_list_GS,
    "RI" : norm_list_RI, 
    "Jacobi" : norm_list_Jacobi,
    "SOR" : norm_list_SOR,
    "ChebySOR" : norm_list_SOR_CHEBY,
    "AOR" : norm_list_AOR,
    "ChebyAOR" : norm_list_AOR_CHEBY,
    "SORNet" : norm_list_SorNet,
    "ChebySORNet" : norm_list_Sor_Cheby_Net,
    "AORNet" : norm_list_AorNet,
    'RINet' : norm_list_rinet
}

# for visualization
marker_list = ['^k-', 'sb-', 'og-', 'Xc-', 'hk-', 'sk-', '^c-', 'sr-', '^r:', '^b:', 'ok:', 'sg:', '^y:']