import math
import random
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../../utils')
from ranking_evaluation import Pre, Rec
from initialization import initialization


n = 943
m = 1682
data = pd.read_csv('../../dataset/ml-100k/u.data', delim_whitespace=True, index_col=False, header=None)
data_length = data.index.size

def preprocess_data():
    return  

# def prediction(b_i, U_minusi_i_u, V_i_T):
#     return b_i + U_minusi_i_u @ V_i_T 

def main():
    alpha = 0.5
    alpha_w = alpha_v = beta_v = 0.01
    gamma = 0.01
    # length of A_u
    rho = 3
    d = 20

    print("FPMC")

    r_usr, I_u, I_u_com, I, r_te, I_te, U_te, b_usr, b_i, W, V = initialization(d)
    
    # T = 100, 500, 1000
    Ts = [100, 400, 500]
    k = 20
    sum_T = 0
    for T in Ts:
        sum_T += T
        print("T = " + str(sum_T))
        # I_re = FISM_auc(I, I_u, I_u_com, r_usr, b_i, W, V, alpha, alpha_v, alpha_w, beta_v, gamma, T, d, rho)

        pre_score = Pre(k, U_te, I_re, I_te)
        print("Pre@" + str(k) + ": " + str(pre_score))

        rec_score = Rec(k, U_te, I_re, I_te)
        print("Rec@" + str(k) + ": " + str(rec_score))


if __name__ == '__main__':
    main()