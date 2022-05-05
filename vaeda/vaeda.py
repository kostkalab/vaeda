import sys
import numpy as np
import scipy.sparse as scs
import pathlib as pl
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
#import pickle
from sklearn.model_selection import train_test_split
from kneed import KneeLocator
from numpy.random import seed
from scipy.signal import savgol_filter
import math

from .vae import define_clust_vae
from .PU import PU, epoch_PU
from .classifier import define_classifier
from .mk_doublets import sim_inflate
from .cluster import cluster, fast_cluster



def vaeda(adata, filter_genes=True, verbose=0, save_dir=None,
          gene_thresh=.01, num_hvgs=2000,
          pca_comp=30, quant=0.25,
          enc_sze=5, max_eps_vae=1000, pat_vae=20, LR_vae=1e-3, clust_weight=20000, rate=-0.75,
          N=1, k_mult=2, max_eps_PU=250, LR_PU=1e-3,
          remove_homos=True, use_old=False,
          seed=None):
    """
    Parameters
    ----------
    adata: anndata object
        raw counts stored in adata.X
    filter_genes: bool, optional
        if True, vaeda will select the 2000 most variable genes. if False, no gene filtering occurs. If set to false, it is recomended that genes are filtered before calling vaeda. (default is True)
    verbose: 'auto', 0, 1, or 2, optional
        verbosity argument passed to tensorflow model.fit functions. (default is 0)
    save_dir: str, optional
        if not None, location to where intermediate steps of vaeda are saved. (use for debugging or when use_old=True) (default=None)
    gene_thresh: float, optional
        Threshold used to filter genes. Genes extressed in <= (gene_thresh*number_of_cells) cells are filtered out. (default is 0.01)
    num_hvgs: int, optional
        The number of highly variable genes that are selected to be used in doublet annotation. (default is 2000)
    pca_comp: int, optional
        Number of pricipal compontents used for PCA reductions. (default is 30)
    quant: float, optional
        Quantile used for initial estimate of number of doublets. (default is 0.25)
    enc_sze: int, optional
        Size of the encoding leaned by vaeda. (default is 5)
    max_eps_vae: int, optional
        Maximum number of epochs used to train the variational auto encoder. Specifically, the vae will train until teh early stopping criteria is met or max_eps_vae, whichever is first. (default is 1000)
    pat_vae: int, optional
        Number of epochs with no improvement after which VAE training will be stopped. (default is 20)
    LR_vae: float, optional
        Learning rate for the VAE (default is 1e-3)
    clust_weight: float, optional
        Weight for the binary cross entropy loss of cluster classifier in the total VAE loss. (default is 20000)
    rate: float, optional
        Rate at which the VAE loss decays after epoch 3. (default is -0.75)
    N: int, optional
        Number of times to repeat the PU loop. (default is 1)
    k_mult: float, optional
        Factor that dtermines the ratio of U and P sizes. U.shape[0] = P.shape[0]*k_mult (default is 2)
    max_eps_PU: int, optional
        Number of epochs to train the epoch selection round of PU learning. (default is 250)
    LR_PU: float, optional
        Learning rate for the doublet classifier. (default is 1e-3)
    remove_homos: bool, optional
        If True, simulated doublets made from cells of the same cluster are removed before training. If False, simulated doublets made from cells of the same cluster are left in. (default is True)
    use_old: bool, optional
        If true, data from previous runs will be used. save_dir must also be set to use this feature. (default is False)
    seed: int, optional
        seed for generating reproducable results. (default is None)
    
    Returns
    -------
    anndata
        the origional anndata object with the encoding stored in adata.obsm['vaeda_embedding'] and the 
        adata.obs['vaeda_scores'] = preds
    adata.obs['vaeda_calls']  = calls
    adata.obsm['vaeda_embedding'] = encoding[Y==0,:]
        doublet probabilities and calls.
    """
    
    
    if(seed is not None):
        np.random.seed(seed)
        seeds = np.random.randint(0, high=sys.maxsize, size=13)
    else:
        seeds = np.random.randint(0, high=sys.maxsize, size=13)
        
    if (save_dir is None):
        use_old = False
        
    X = np.array(adata.X.todense())    
    
    #######################################################
    ######################### SIM #########################
    #######################################################
    old_sim = False
    if (save_dir is not None):
        npz_sim_path  = save_dir + 'sim_doubs.npz'
        sim_ind_path  = save_dir + 'sim_ind.npy'
        npz_sim  = pl.Path(npz_sim_path)
        if (npz_sim.exists()):
            old_sim = True
    
    if(old_sim & use_old):#if old sim doubs exist and we want to use them...
        if(verbose!=0):
            print('loading in simulated doublets')
        
        dat_sim = scs.load_npz(npz_sim)
        sim_ind = np.load(sim_ind_path)
        ind1 = sim_ind[0,:]
        ind2 = sim_ind[1,:]
        Xs = scs.csr_matrix(dat_sim).toarray()
        
    else: #otherwise, make new
        if(verbose!=0):
            print('generating simulated doublets')        
        Xs, ind1, ind2 = sim_inflate(X)        
        dat_sim = scs.csr_matrix(Xs) 
        
        if(save_dir is not None):#if we are saving, then save
            scs.save_npz(npz_sim_path,dat_sim) 
            np.save(sim_ind_path, np.vstack([ind1,ind2]))

            
    Y = np.concatenate([np.zeros(X.shape[0]), np.ones(Xs.shape[0])])
    X = np.vstack([X,Xs])
    
    if (filter_genes):
        #Filter genes
        thresh = np.floor(X.shape[0]) * gene_thresh
        tmp    = np.sum((X>0), axis=0)>thresh
        if (np.sum(tmp) >= num_hvgs):
            X = X[:,tmp]

        #- HVGs
        if(X.shape[1] > num_hvgs):
            var = np.var(X, axis=0)
            #if(reproducable):
            #    np.random.seed(3900362577)
            np.random.seed(seeds[0]) #*
            hvgs = np.argpartition(var, -num_hvgs)[-num_hvgs:]  
            X = X[:,hvgs]

    #######################################################
    ######################### KNN #########################
    #######################################################
    
    #HYPERPARAMS
    neighbors = int(np.sqrt(X.shape[0]))
    
    #SCALING
    temp_X = np.log2(X+1)
    scaler = StandardScaler().fit(temp_X.T)
    temp_X = scaler.transform(temp_X.T).T
    
    np.random.seed(seeds[1]) #*
    pca = PCA(n_components=pca_comp)
    pca_proj = pca.fit_transform(temp_X)
    del(temp_X)
    
    np.random.seed(seeds[2]) #*
    knn = NearestNeighbors(n_neighbors=neighbors)
    knn.fit(pca_proj,Y)
    graph = knn.kneighbors_graph(pca_proj)
    knn_feature = np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors)) #sum across rows
    
    #estimate true faction of doublets 
    quantile = np.quantile(knn_feature[Y==1], quant)
    num = np.sum(knn_feature[Y==0]>=quantile)
    min_num = int(np.round((sum(Y==0) *0.05)))
    num = np.max([min_num, num, 1])
    estimated_doub_frac = num / sum(Y==0)
    estimated_doub_num = num
    
    prob = knn_feature[Y==1] / np.sum(knn_feature[Y==1])
    np.random.seed(seeds[3])
    ind = np.random.choice(np.arange(sum(Y==1)), size=num, p=prob, replace=False)
    
    #downsample the simulated doublets
    enc_ind = np.concatenate([np.arange(sum(Y==0)), (sum(Y==0) + ind)])
    X = X[enc_ind,:]
    Y = Y[enc_ind]
    knn_feature = knn_feature[enc_ind]
    
    #re-scale
    X = np.log2(X+1)
    np.random.seed(seeds[4]) #*
    scaler = StandardScaler().fit(X.T)
    np.random.seed(seeds[5]) #*
    X = scaler.transform(X.T).T
    
    #######################################################
    ####################### CLUSTER #######################
    #######################################################    
    if(X.shape[0]>=1000):
        clust = fast_cluster(X, comp=pca_comp)
    else:
        clust = cluster(X, comp=pca_comp)
    
    if(remove_homos):
        c = clust[Y==0]

        hetero_ind = c[ind1] != c[ind2]
        hetero_ind = hetero_ind[ind] #becasue down sampled
        if(save_dir is not None):
            np.save(save_dir + 'which_sim_doubs.npy', ind[hetero_ind])
        
        hetero_ind = np.concatenate([np.full(sum(Y==0), True), hetero_ind])
        
        X = X[hetero_ind,:]
        Y = Y[hetero_ind]
        clust = clust[hetero_ind]
        knn_feature = knn_feature[hetero_ind]
        
    else:
        if(save_dir is not None):
            np.save(save_dir + 'which_sim_doubs.npy', ind)
        
    
    #######################################################
    ######################### VAE #########################
    #######################################################
    X_train, X_test, clust_train, clust_test = train_test_split(X, clust, test_size=0.1, random_state=12345)
    clust_train = tf.one_hot(clust_train, depth=clust.max()+1)
    clust_test = tf.one_hot(clust_test, depth=clust.max()+1)
    
    ngens = X.shape[1]
    
    #VAE
    old_VAE = False
    if (save_dir is not None):
        vae_path_real = save_dir + 'embedding_real.npy'
        vae_path_sim = save_dir + 'embedding_sim.npy'
        if (pl.Path(vae_path_real).exists() & pl.Path(vae_path_sim).exists()):
            old_VAE = True
    
    if (old_VAE & use_old): #if the old files exist and we want to use them...
        if(verbose!=0):
            print('using existing encoding')
        encoding_real = np.load(vae_path_real)
        encoding_sim = np.load(vae_path_sim)
        encoding = np.vstack([encoding_real, encoding_sim])
        made_new=False
    else:#otherwise, make new
        if(verbose!=0):
            print('generating VAE encoding')
        made_new=True
        
        tf.random.set_seed(seeds[6])
        vae = define_clust_vae(enc_sze, ngens, clust.max()+1, LR=LR_vae, clust_weight=clust_weight)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    mode = 'min',
                                                    min_delta=0, 
                                                    patience=pat_vae, 
                                                    verbose=0, 
                                                    restore_best_weights=False)

        def scheduler(epoch, lr):
            if epoch < 3:
                return lr
            else:
                return lr * tf.math.exp(rate)

        callback2 = tf.keras.callbacks.LearningRateScheduler(scheduler)

        hist = vae.fit(x=[X_train],
                       y=[X_train, clust_train],
                       validation_data=([X_test], [X_test, clust_test]),
                       epochs=max_eps_vae, 
                       use_multiprocessing=True,
                       callbacks=[callback, callback2],
                       verbose=verbose)

        encoder = vae.get_layer('encoder')
        tf.random.set_seed(seeds[7])
        encoding = np.array(tf.convert_to_tensor(encoder(X)))

        if(save_dir is not None):
            np.save(vae_path_real, encoding[Y==0,:])
            np.save(vae_path_sim, encoding[Y==1,:])


    #######################################################
    ######################### PU ##########################
    #######################################################
    
    if(save_dir is not None):
        np.save(save_dir + 'knn_feature_real.npy', knn_feature[Y==0])
        np.save(save_dir + 'knn_feature_sim.npy', knn_feature[Y==1])
        np.save(save_dir + 'clusters_real.npy', clust[Y==0])
        np.save(save_dir + 'clusters_sim.npy', clust[Y==1])
    
    encoding_keep = encoding
    encoding = np.vstack([knn_feature,encoding.T]).T
    
    #PU BAGGING
    if(verbose!=0):
        print('starting PU Learning')
    U = encoding[Y==0,:]
    P = encoding[Y==1,:]

    num_cells = P.shape[0]*k_mult
    k = int(U.shape[0] / num_cells)
    if(k<2):
        k=2
    
    hist = epoch_PU(U, P, k, N, max_eps_PU, seeds=seeds[8:], puLR=LR_PU, verbose=verbose)#seeds 8-12
            
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
        knee = knee+100
        
    elif knee<20:
        knee = 20
        
    elif knee>250:
        knee = 250
    
    preds, preds_on_P, hists, _, _, _ = PU(U, P, k, N, knee, seeds=seeds[8:], puLR=LR_PU, verbose=verbose)

    if(save_dir is not None):
        np.save(save_dir + 'scores.npy', preds)
        np.save(save_dir + 'scores_on_sim.npy', preds_on_P)
  
    #######################################################
    ####################### CALLS #########################
    #######################################################
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
    if(save_dir is not None):
        if(verbose!=0):
            print('saving calls')
        np.save(save_dir + 'doublet_calls.npy', calls)
    
    adata.obs['vaeda_scores'] = preds
    adata.obs['vaeda_calls']  = calls
    adata.obsm['vaeda_embedding'] = encoding[Y==0,:]
    
    #return preds, preds_on_P, calls, encoding_keep, knn_feature
    return adata

def log_norm(x, mean, sd):
    t1 = -np.log(sd*np.sqrt(2*math.pi))
    t2 = (-.5)*((x-mean)/sd)**2
    return t1+t2