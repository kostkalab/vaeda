#- PU
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from .classifier import define_classifier

def PU(U, P, k, N, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1, stop_metric='ValAUC', verbose=0):
    
    if (seeds):
        random_state = seeds[0]
        rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)
    else:
        rkf = RepeatedKFold(n_splits=k, n_repeats=N)

    preds= np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    hists = np.zeros([N*k, cls_eps])

    val_hists = np.zeros([N*k, cls_eps])
    auc_hists = np.zeros([N*k, cls_eps])
    val_auc = np.zeros([N*k, cls_eps])

    i = 0
    for test, train in rkf.split(U):
        
        i = i + 1
        if(verbose!=0):
            print('')
            print(str(i) + '/' + str(N*k) + ' itterations')
        
        X = np.vstack([U[train,:], P])
        Y = np.concatenate([np.zeros([len(train)]),
                               np.ones([P.shape[0]])])

        x = U[test,:]

        if (clss=='NN'):
            #DEFINE MODEL
            np.random.seed(1)
            if(seeds):
                tf.random.set_seed(seeds[1])
            classifier = define_classifier(X.shape[1], num_layers=num_layers)

            #shuffle training data
            ind = np.arange(X.shape[0])
            if(seeds):
                np.random.seed(seeds[2])
            np.random.shuffle(ind)
            
            auc = tf.keras.metrics.AUC(curve='PR', name='auc')
            classifier.compile(optimizer = tf.optimizers.Adam(learning_rate=puLR),
                          loss = 'binary_crossentropy', 
                          metrics=[auc])
            
            
            if ((X.shape[0] * 0.1) >= 50 ):

                if(seeds):
                    tf.random.set_seed(seeds[3])
                hist = classifier.fit(x=X,
                                      y=Y,
                                      epochs=cls_eps, 
                                      verbose=False,
                                      use_multiprocessing=True)

            else:
                    
                ind = np.arange(X.shape[0])
                if(seeds):
                    np.random.seed(seeds[2])
                np.random.shuffle(ind)
                
                if(seeds):
                    tf.random.set_seed(seeds[3])
                hist = classifier.fit(x=X[ind,:], 
                                      y=Y[ind], 
                                      epochs=cls_eps, 
                                      verbose=False)
            
            hists[i-1,:len(hist.history['loss'])]= hist.history['loss']
            auc_hists[i-1,:len(hist.history['auc'])]= hist.history['auc']
            
            if(seeds):
                tf.random.set_seed(seeds[4])
            preds[test] = preds[test] + np.array(classifier(x)).flatten()
            if(seeds):
                tf.random.set_seed(seeds[4])
            preds_on_P = preds_on_P + np.array(classifier(P)).flatten()

            
            
        if (clss == 'knn'):
            neighbors = int(np.sqrt(X.shape[0]))
            knn = NearestNeighbors(n_neighbors=neighbors)
            knn.fit(X,Y)
            
            graph = knn.kneighbors_graph(x)
            preds[test] = preds[test] + np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors))

            graph = knn.kneighbors_graph(P)
            preds_on_P = preds_on_P + np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors))
            

    preds = preds / ((i/k)*(k-1))
    preds_on_P = preds_on_P / ((i/k)*(k-1))

    return preds, preds_on_P, hists, val_hists, auc_hists, val_auc





def epoch_PU(U, P, k, N, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1, stop_metric='ValAUC', verbose=0):
    
    if(seeds):
        random_state = seeds[0]
        rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)
    else:
        rkf = RepeatedKFold(n_splits=k, n_repeats=N)
        
    preds= np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    Y_keep=np.concatenate([np.zeros(U.shape[0]), np.ones(P.shape[0])])

    i = 0
    for test, train in rkf.split(U):
        
        i = i + 1
        if(verbose!=0):
            print('')
            print(str(i) + '/' + str(N*k) + ' itterations')

        X = np.vstack([U[train,:], P])
        Y = np.concatenate([np.zeros([len(train)]),
                               np.ones([P.shape[0]])])

        #DEFINE MODEL
        if(seeds):
            np.random.seed(1)
            tf.random.set_seed(seeds[1])
        classifier = define_classifier(X.shape[1], num_layers=num_layers)

        #shuffle training data
        ind = np.arange(X.shape[0])
        if(seeds):
            np.random.seed(seeds[2])
        np.random.shuffle(ind)

        auc = tf.keras.metrics.AUC(curve='PR', name='auc')
        classifier.compile(optimizer = tf.optimizers.Adam(learning_rate=puLR),
                           loss = 'binary_crossentropy',
                           metrics=[auc])

        if(seeds):
            tf.random.set_seed(seeds[3])
        hist = classifier.fit(x=X,
                              y=Y,
                              epochs=cls_eps, 
                              verbose=False,
                              use_multiprocessing=True)
        
        break
            
    return hist

