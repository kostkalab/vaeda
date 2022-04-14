import numpy as np
import scipy.sparse as scs
from scipy.stats import multinomial
import pathlib as pl
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import mmread

from os import listdir
from os.path import isfile, join
import os

import random

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import seaborn as sns

from sklearn.preprocessing import StandardScaler

import random
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, accuracy_score, average_precision_score


import tensorflow as tf

import os
import pickle

from sklearn.model_selection import train_test_split

from kneed import KneeLocator

from numpy.random import seed

import time
import scanpy as sc
import anndata
from scipy.signal import savgol_filter
import math




from plot_results import get_dbl_metrics
from vae import define_clust_vae
from PU import PU, epoch_PU, epoch_PU2
from classifier import define_classifier
from mk_doublets import sim_inflate, sim_avg, sim_sum
from cluster import cluster, fast_cluster







def vaeda(X, save_dir='',
          gene_thresh=.01, num_hvgs=2000,
          pca_comp=30, quant=0.25,
          enc_sze=5, max_eps_vae=1000, pat_vae=20, LR_vae=1e-3, clust_weight=20000, rate=-0.75,
          N=1, k_mult=2, max_eps_PU=250, LR_PU=1e-3,
          remove_homos=True, use_old=True,
          seeds=[42, 29503, 432809, 42, 132975, 9231996, 12883823, 9231996, 1234, 62938, 57203 ,109573, 23]):
    
    
    #-time each step
    time_names = ['total', 'simulation', 'HVGs', 'scaling1', 'knn', 'downsample', 'scaling2', 'cluster', 'vae', 'epoch_selection', 'PU_loop']
    tmp1 = np.copy(np.zeros((1, len(time_names))))
    time_df = pd.DataFrame(tmp1, index=['time'], columns=time_names)

    
    #Filter genes
    #thresh = np.floor(X.shape[0]) * gene_thresh
    #tmp    = np.sum((X>0), axis=0)>thresh
    # X = X[:,tmp]
    
    
    #######################################################
    ######################### SIM #########################
    #######################################################
    npz_sim_path  = save_dir + 'sim_doubs.npz'
    sim_ind_path  = save_dir + 'sim_ind.npy'
    
    npz_sim  = pl.Path(npz_sim_path)

    total_start = time.perf_counter()
    
    if (npz_sim.exists() & use_old):
        print('loading in sim npz')
        
        dat_sim = scs.load_npz(npz_sim)
        sim_ind = np.load(sim_ind_path)
        ind1 = sim_ind[0,:]
        ind2 = sim_ind[1,:]
        Xs = scs.csr_matrix(dat_sim).toarray()
        
    else:
        print('generating new sim npz')

        start = time.perf_counter()
        
        Xs, ind1, ind2 = sim_inflate(X)
        stop = time.perf_counter()

        time_df.simulation = stop - start
        
        dat_sim = scs.csr_matrix(Xs) 
        
        if(len(save_dir)>0):
            scs.save_npz(npz_sim_path,dat_sim) 
            np.save(sim_ind_path, np.vstack([ind1,ind2]))

    Y = np.concatenate([np.zeros(X.shape[0]), np.ones(Xs.shape[0])])
    X = np.vstack([X,Xs])
    
    #Filter genes
    thresh = np.floor(X.shape[0]) * gene_thresh
    tmp    = np.sum((X>0), axis=0)>thresh
    X = X[:,tmp]
    
    #- HVGs
    if(X.shape[1] > num_hvgs):
        start = time.perf_counter()

        var = np.var(X, axis=0)
        np.random.seed(3900362577)
        hvgs = np.argpartition(var, -num_hvgs)[-num_hvgs:]  

        stop = time.perf_counter()

        time_df.HVGs = stop - start

        X = X[:,hvgs]

    #######################################################
    ######################### KNN #########################
    #######################################################
    
    #HYPERPARAMS
    neighbors = int(np.sqrt(X.shape[0]))

    start = time.perf_counter()
    
    #SCALING
    temp_X = np.log2(X+1)
    np.random.seed(42)
    scaler = StandardScaler().fit(temp_X.T)
    np.random.seed(42)
    temp_X = scaler.transform(temp_X.T).T

    stop = time.perf_counter()
    time_df.scaling1 = stop - start
    
    
    #KNN
    start = time.perf_counter()
    
    np.random.seed(42)
    pca = PCA(n_components=pca_comp)
    pca_proj = pca.fit_transform(temp_X)
    del(temp_X)
    
    np.random.seed(42)
    knn = NearestNeighbors(n_neighbors=neighbors)
    knn.fit(pca_proj,Y)
    graph = knn.kneighbors_graph(pca_proj)
    knn_feature = np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors)) #sum across rows

    stop = time.perf_counter()
    time_df.knn = stop - start
    
    
    start = time.perf_counter()
    
    #estimate true faction of doublets 
    quantile = np.quantile(knn_feature[Y==1], quant)
    num = np.sum(knn_feature[Y==0]>=quantile)
    min_num = int(np.round((sum(Y==0) *0.05)))
    num = np.max([min_num, num])
    estimated_doub_frac = num / sum(Y==0)
    estimated_doub_num = num
    print('estimated number of doublets:', estimated_doub_num)
    
    prob = knn_feature[Y==1] / np.sum(knn_feature[Y==1])
    np.random.seed(seeds[0])
    ind = np.random.choice(np.arange(sum(Y==1)), size=num, p=prob, replace=False)
    
    #ind = sum(Y==0) + ind

    #downsample the simulated doublets
    enc_ind = np.concatenate([np.arange(sum(Y==0)), (sum(Y==0) + ind)])
    X = X[enc_ind,:]
    Y = Y[enc_ind]
    knn_feature = knn_feature[enc_ind]
    
    stop = time.perf_counter()
    time_df.downsample = stop - start

    
    start = time.perf_counter()
    
    #re-scale
    X = np.log2(X+1)
    np.random.seed(42)
    scaler = StandardScaler().fit(X.T)
    np.random.seed(42)
    X = scaler.transform(X.T).T
    
    stop = time.perf_counter()
    time_df.scaling2 = stop - start
    
    #######################################################
    ####################### CLUSTER #######################
    #######################################################
    
    start = time.perf_counter()
    
    if(X.shape[0]>=1000):
        clust = fast_cluster(X, comp=pca_comp)
    else:
        clust = cluster(X, comp=pca_comp)
        
    stop = time.perf_counter()
    time_df.cluster = stop - start
    
    
    if(remove_homos):
        c = clust[Y==0]

        hetero_ind = c[ind1] != c[ind2]
        hetero_ind = hetero_ind[ind] #becasue down sampled
        print('number of homos:', sum((hetero_ind*-1)+1))
        if(len(save_dir)>0):
            np.save(save_dir + 'which_sim_doubs.npy', ind[hetero_ind])
        
        hetero_ind = np.concatenate([np.full(sum(Y==0), True), hetero_ind])
        
        X = X[hetero_ind,:]
        Y = Y[hetero_ind]
        clust = clust[hetero_ind]
        knn_feature = knn_feature[hetero_ind]
        
    else:
        if(len(save_dir)>0):
            np.save(save_dir + 'which_sim_doubs.npy', ind)
        
    
    #######################################################
    ######################### VAE #########################
    #######################################################
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=12345)
    X_train, X_test, clust_train, clust_test = train_test_split(X, clust, test_size=0.1, random_state=12345)
    clust_train = tf.one_hot(clust_train, depth=clust.max()+1)
    clust_test = tf.one_hot(clust_test, depth=clust.max()+1)
    
    
    ngens = X.shape[1]
    
    #VAE
    vae_path_real = save_dir + 'embedding_real.npy'
    vae_path_sim = save_dir + 'embedding_sim.npy'
    if (pl.Path(vae_path_real).exists() & pl.Path(vae_path_sim).exists() & (use_old)):
        print('using old encoding')
        encoding_real = np.load(vae_path_real)
        encoding_sim = np.load(vae_path_sim)
        encoding = np.vstack([encoding_real, encoding_sim])
        made_new=False
    else:
        print('generating new VAE encoding')
        made_new=True
        
        start = time.perf_counter()
        
        tf.random.set_seed(seeds[1])
        vae = define_clust_vae(enc_sze, ngens, clust.max()+1, LR=LR_vae, clust_weight=clust_weight)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    mode = 'min',
                                                    min_delta=0, 
                                                    patience=pat_vae, 
                                                    verbose=True, 
                                                    restore_best_weights=False)

        def scheduler(epoch, lr):
            if epoch < 3:
                return lr
            else:
                return lr * tf.math.exp(rate)

        callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler)

        #tf.config.optimizer.set_jit(True)
        hist = vae.fit(x=[X_train],
                       y=[X_train, clust_train],
                       validation_data=([X_test], [X_test, clust_test]),
                       epochs=max_eps_vae, 
                       use_multiprocessing=True,
                       callbacks=[callback, callback2])

        encoder = vae.get_layer('encoder')
        tf.random.set_seed(seeds[2])
        encoding = np.array(tf.convert_to_tensor(encoder(X)))
        
        stop = time.perf_counter()
        time_df.vae = stop - start

        if(len(save_dir)>0):
            np.save(vae_path_real, encoding[Y==0,:])
            np.save(vae_path_sim, encoding[Y==1,:])


    #######################################################
    ######################### PU ##########################
    #######################################################
    
    if(len(save_dir)>0):
        np.save(save_dir + 'knn_feature_real.npy', knn_feature[Y==0])
        np.save(save_dir + 'knn_feature_sim.npy', knn_feature[Y==1])
        np.save(save_dir + 'clusters_real.npy', clust[Y==0])
        np.save(save_dir + 'clusters_sim.npy', clust[Y==1])
    
    encoding_keep = encoding
    encoding = np.vstack([knn_feature,encoding.T]).T
    
    #PU BAGGING
    U = encoding[Y==0,:]
    P = encoding[Y==1,:]

    num_cells = P.shape[0]*k_mult
    k = int(U.shape[0] / num_cells)
    if(k<2):
        k=2
    
    start = time.perf_counter()
    
    hist = epoch_PU2(U, P, k, N, 250, seeds=seeds[3:], puLR=LR_PU)
            
    y=np.log(hist.history['loss'])
    x=np.arange(len(y))
    yhat = savgol_filter(y, window_length=7, polyorder=1) 

    y=yhat
    x=np.arange(len(y))

    kneedle = KneeLocator(x, y, S=10, curve='convex', direction='decreasing')

    knee = kneedle.knee

    if knee==None:
        knee = 250
    
    elif(num < 500):#add epochs if ther aren't enough cells
        print('added 100')
        knee = knee+100
        
    elif knee<20:
        knee = 20
        
    elif knee>250:
        knee = 250

    print('KNEE:', knee)   
        
    stop = time.perf_counter()
    time_df.epoch_selection = stop - start

    start = time.perf_counter()
    
    ##new v
    #tf.config.optimizer.set_jit(True)
    preds, preds_on_P, hists, _, _, _ = PU(U, P, k, N, knee, seeds=seeds[3:], puLR=LR_PU)
    ##new ^
    
    stop = time.perf_counter()
    time_df.PU_loop = stop - start
    
    total_stop = time.perf_counter()
    
    time_df.total = total_stop - total_start
    
    print(time_df)
    
    if(len(save_dir)>0):
        print('saving time df ', time_df)
        time_df.to_csv(save_dir + 'time.csv')
    

    if(len(save_dir)>0):
        np.save(save_dir + 'scores.npy', preds)
        np.save(save_dir + 'scores_on_sim.npy', preds_on_P)

        
    #######################################################
    ####################### CALLS #########################
    #######################################################
    #doub_call_ind = np.argsort(preds)[(-1*estimated_doub_num):]
    #calls = np.full(len(preds), 'singlet')
    #calls[doub_call_ind] = 'doublet'
    #if(len(save_dir)>0):
    #    print('saving calls')
    #    np.save(save_dir + 'doublet_calls.npy', calls)
    
    maximum = np.max([np.max(preds), np.max(preds_on_P)])
    minimum = np.min([np.min(preds), np.min(preds_on_P)])
    
    thresholds = np.arange(minimum,maximum,0.001)
    
    n = len(preds)
    dbr = n/10**5
    dbl_expected = n*dbr
    dbr_sd = np.sqrt(n*dbr*(1-dbr))
    
    FNR = []
    FPR = []
    nll_doub=[]
    
    o_t = np.sum(preds>=thresholds[-1])
    norm_factor = -(log_norm(o_t, dbl_expected, dbr_sd))
    
    for thresh in thresholds:
        
        o_t = np.sum(preds>=thresh)
        
        FNR.append((np.sum(preds_on_P<thresh)/len(preds_on_P)))
        FPR.append((o_t/len(preds)))   
        nll_doub.append((-(log_norm(o_t, dbl_expected, dbr_sd)/norm_factor)))
        
            
    cost = np.array(FNR) + np.array(FPR) + np.array(nll_doub)**2
    
    t = thresholds[np.argmin(cost)]
    call_mask = preds > t
    
    calls = np.full(len(preds), 'singlet')
    calls[call_mask] = 'doublet'
    if(len(save_dir)>0):
        print('saving calls')
        np.save(save_dir + 'doublet_calls.npy', calls)
    
    return preds, preds_on_P, calls, encoding_keep, knn_feature

    


def log_norm(x, mean, sd):
    t1 = -np.log(sd*np.sqrt(2*math.pi))
    t2 = (-.5)*((x-mean)/sd)**2
    return t1+t2