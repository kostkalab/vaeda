import numpy as np
import random

def sim_inflate(X, frac_doublets=None, seeds=[1234, 15232, 3060309]):
    if frac_doublets==None:
        num_doublets = 1 * X.shape[0]
    else:
        num_doublets = int(frac_doublets * X.shape[0])
    
    ind1 = np.arange(X.shape[0])
    ind2 = np.arange(X.shape[0])
    
    np.random.seed(seeds[0])
    np.random.shuffle(ind1)
    np.random.seed(seeds[1])
    np.random.shuffle(ind2)
    
    X1 = np.copy(X)[ind1,:]
    X2 = np.copy(X)[ind2,:]
    
    res = X1 + X2
    
    lib1 = np.sum(X1, axis=1)
    lib2 = np.sum(X2, axis=1)
    
    lib_sze = np.maximum.reduce([lib1,lib2])
    high = np.max(lib_sze)
    
    inflated_sze = np.zeros([len(lib_sze)])
    for i, low in enumerate(lib_sze): 
        random.seed(seeds[2])
        inflated_sze[i] = random.choice(lib_sze[lib_sze>=low])
    
    ls = np.sum(res, axis=1)
    sf = inflated_sze / ls
    res= np.multiply(res.T, sf).T
    
    return res[:num_doublets,:], ind1[:num_doublets], ind2[:num_doublets]

