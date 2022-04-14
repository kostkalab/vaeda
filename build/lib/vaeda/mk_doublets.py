import numpy as np
import random
import tensorflow as tf
import tensorflow_probability as tfp
tfd  = tfp.distributions

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
        #inflated_sze[i] = np.random.randint(low, high=high+1, dtype=int)
    
    ls = np.sum(res, axis=1)
    sf = inflated_sze / ls
    res= np.multiply(res.T, sf).T
    
    return res[:num_doublets,:], ind1[:num_doublets], ind2[:num_doublets]



def sim_sum(X, frac_doublets=None, seeds=[1234, 15232, 3060309]):
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
    
    return res[:num_doublets,:], ind1, ind2



def sim_avg(X, frac_doublets=None, seeds=[1234, 15232, 3060309]):
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
    
    res = (X1 + X2) / 2
    
    return res[:num_doublets,:], ind1[:num_doublets], ind2[:num_doublets]





def mk_doublets(X):
    
    X_s = np.copy(X)
    
    np.random.shuffle(X_s)
    mx_prop   = np.array(tfd.Sample(tfd.Independent(tfd.Uniform(low=0.0, high=0.5)),sample_shape=[X.shape[0],1]).sample())
    res = mx_prop*X  + (1.0-mx_prop)*X_s
    res = res + np.array(tfd.Sample(tfd.Independent(tfd.Normal(loc=0.0, scale=0.1)),sample_shape=[res.shape[0],res.shape[1]]).sample())
    res = (res.transpose() - res.mean(axis=1)).transpose()
    return (res, mx_prop)

