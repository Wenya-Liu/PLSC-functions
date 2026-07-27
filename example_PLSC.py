#%%
import numpy as np
import PLSC_func as plscfun

#%% loading data

X_matrix = np.load("Path to the imaging data")
Y_matrix = np.load("Path to the behavioral data")

#%% performing PLSC

Lx, Ly, corr_Lx_X, corr_Ly_Y, S, U, Vt = plscfun.PLSC(X_matrix, Y_matrix)
LC_pvals, Sp_vect = plscfun.permutation_test(X_matrix, Y_matrix, S, U, n_perm=10000)
n_sig = LC_pvals[LC_pvals<=0.05].shape[0]
results_bs = plscfun.bootstrap_test(X_matrix, Y_matrix, U, Vt.T, n_perm = 100, procrustas='U_only')
results_bootstrap_stats = plscfun.bootstrap_stats(results_bs)
