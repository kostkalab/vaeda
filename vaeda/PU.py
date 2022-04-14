#- PU
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from sklearn.model_selection import RepeatedKFold
import sklearn

from classifier import define_classifier
from sklearn.neighbors import NearestNeighbors
from sklearn import svm

from sklearn.model_selection import train_test_split

#def PU(U, P, true, k, N, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1, stop_metric='ValAUC'):
def PU(U, P, k, N, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1, stop_metric='ValAUC'):
    #PU BAGGING
    
    if(seeds==None):
        print('generating seeds')
        seeds = [42, 42, 42, 42]
    
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    preds= np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    hists = np.zeros([N*k, cls_eps])
    ##new v
    val_hists = np.zeros([N*k, cls_eps])
    auc_hists = np.zeros([N*k, cls_eps])
    val_auc = np.zeros([N*k, cls_eps])
    ## new ^
    i = 0
    for test, train in rkf.split(U):
        
        i = i + 1
        print('')
        print(str(i) + '/' + str(N*k) + ' itterations')

        #t = true[true<2]
        #t = t[train]
        
        X = np.vstack([U[train,:], P])
        Y = np.concatenate([np.zeros([len(train)]),
                               np.ones([P.shape[0]])])

        x = U[test,:]
        #y = true.astype(int)[test]

        if (clss=='NN'):
            #DEFINE MODEL
            np.random.seed(1)
            tf.random.set_seed(seeds[1])
            classifier = define_classifier(X.shape[1], num_layers=num_layers)

            #shuffle training data
            ind = np.arange(X.shape[0])
            np.random.seed(seeds[2])
            np.random.shuffle(ind)
            
            
            ##new v
            print("LR: " + str(puLR))
            auc = tf.keras.metrics.AUC(curve='PR', name='auc')
            classifier.compile(optimizer = tf.optimizers.Adam(learning_rate=puLR),
                          loss = 'binary_crossentropy', 
                          metrics=[auc])
            
            
            if ((X.shape[0] * 0.1) >= 50 ):

                
                tf.random.set_seed(seeds[3])
                hist = classifier.fit(x=X,
                                      y=Y,
                                      #validation_data=(X[t,:], t),
                                      epochs=cls_eps, 
                                      verbose=False,
                                      use_multiprocessing=True)#,
                                      #callbacks=[callback])

                #val_hists[i-1,:len(hist.history['val_loss'])]= hist.history['val_loss']
                #val_auc[i-1,:len(hist.history['val_auc'])]= hist.history['val_auc']
            else:
                
                print('not enough cells for train test split')
                print("pat: " + str(puPat))
                if (stop_metric == 'ValAUC'):
                    print('using AUC')
                    
                    callback = tf.keras.callbacks.EarlyStopping(monitor='auc', 
                                                                mode = 'max',
                                                                min_delta=0.0, 
                                                                patience=puPat,#was 10
                                                                restore_best_weights=True)#was F
                    
                elif (stop_metric == 'ValLoss'):
                    print('using Loss')
                    
                    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                                mode = 'min',
                                                                min_delta=0.0, 
                                                                patience=puPat,#was 10
                                                                restore_best_weights=True)#was F
                ind = np.arange(X.shape[0])
                np.random.seed(seeds[2])
                np.random.shuffle(ind)
                
                tf.random.set_seed(seeds[3])
                #https://www.tensorflow.org/api_docs/python/tf/keras/Model
                hist = classifier.fit(x=X[ind,:], 
                                      y=Y[ind], 
                                      epochs=cls_eps, 
                                      verbose=False )
                                      #callbacks=[callback])
            
            hists[i-1,:len(hist.history['loss'])]= hist.history['loss']
            auc_hists[i-1,:len(hist.history['auc'])]= hist.history['auc']
            ##new ^
            
            ##old v
            #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=10)
            #tf.random.set_seed(seeds[3])
            #https://www.tensorflow.org/api_docs/python/tf/keras/Model
            #hist = classifier.fit(x=X[ind,:], y=Y[ind], epochs=cls_eps, verbose=False ,callbacks=[callback])

            #hists[i-1,:len(hist.history['loss'])]= hist.history['loss']
            #old ^
            
            tf.random.set_seed(seeds[4])
            preds[test] = preds[test] + np.array(classifier(x)).flatten()
            tf.random.set_seed(seeds[4])
            preds_on_P = preds_on_P + np.array(classifier(P)).flatten()
            
            
            
            #res = get_dbl_metrics(t[test], preds[test])
            #plt.show()
            #plt.close()
            
            
            
            
        if (clss == 'knn'):
            neighbors = int(np.sqrt(X.shape[0]))
            knn = NearestNeighbors(n_neighbors=neighbors)
            knn.fit(X,Y)
            
            graph = knn.kneighbors_graph(x)
            preds[test] = preds[test] + np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors)) #sum across rows

            graph = knn.kneighbors_graph(P)
            preds_on_P = preds_on_P + np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors))
            
        if (clss == 'SVM'):
            clf = svm.SVC(kernel='linear')
            clf.fit(X, Y)
            
            preds[test] = preds[test] + clf.predict(x)#.flatten()
            preds_on_P = preds_on_P + clf.predict(P)#.flatten()

    preds = preds / ((i/k)*(k-1))
    preds_on_P = preds_on_P / ((i/k)*(k-1))

    #return preds, preds_on_P, hists
    return preds, preds_on_P, hists, val_hists, auc_hists, val_auc




#def epoch_PU(U, P, true, k, N, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1, stop_metric='ValAUC'):
def epoch_PU(U, P, k, N, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1, stop_metric='ValAUC'):
    #PU BAGGING
    
    if(seeds==None):
        print('generating seeds')
        seeds = [42, 42, 42, 42]
    
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    preds= np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    #hists = np.zeros([N*k, cls_eps])

    i = 0
    for test, train in rkf.split(U):
        
        i = i + 1
        print('')
        print(str(i) + '/' + str(N*k) + ' itterations')

        #t = true[true<2]
        #t = t[train]
        
        X = np.vstack([U[train,:], P])
        Y = np.concatenate([np.zeros([len(train)]),
                               np.ones([P.shape[0]])])

        x = U[test,:]
        #y = true.astype(int)[test]

        #DEFINE MODEL
        np.random.seed(1)
        tf.random.set_seed(seeds[1])
        classifier = define_classifier(X.shape[1], num_layers=num_layers)

        #shuffle training data
        ind = np.arange(X.shape[0])
        np.random.seed(seeds[2])
        np.random.shuffle(ind)

        ##new v
        print("LR: " + str(puLR))
        classifier.compile(optimizer = tf.optimizers.Adam(learning_rate=puLR),
                      loss = 'binary_crossentropy')


        if ((X.shape[0] * 0.1) >= 50 ):

            tf.random.set_seed(seeds[3])
            hist = classifier.fit(x=X,
                                  y=Y,
                                  epochs=cls_eps, 
                                  verbose=False,
                                  use_multiprocessing=True)

        else:

            ind = np.arange(X.shape[0])
            np.random.seed(seeds[2])
            np.random.shuffle(ind)

            tf.random.set_seed(seeds[3])
            #https://www.tensorflow.org/api_docs/python/tf/keras/Model
            hist = classifier.fit(x=X[ind,:], 
                                  y=Y[ind], 
                                  epochs=cls_eps, 
                                  verbose=False )

        #hists[i-1,:len(hist.history['loss'])]= hist.history['loss']
        
        break
            

    #preds = preds / ((i/k)*(k-1))
    #preds_on_P = preds_on_P / ((i/k)*(k-1))

    #return preds, preds_on_P, hists
    return hist




#def epoch_PU2(U, P, true, k, N, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1, stop_metric='ValAUC'):
def epoch_PU2(U, P, k, N, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1, stop_metric='ValAUC'):
    #PU BAGGING
    
    if(seeds==None):
        print('generating seeds')
        seeds = [42, 42, 42, 42]
    
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    preds= np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    #hists = np.zeros([N*k, cls_eps])
    Y_keep=np.concatenate([np.zeros(U.shape[0]), np.ones(P.shape[0])])

    i = 0
    for test, train in rkf.split(U):
        
        i = i + 1
        print('')
        print(str(i) + '/' + str(N*k) + ' itterations')

        #t = true[true<2]
        #t = t[train]
        
        X = np.vstack([U[train,:], P])
        Y = np.concatenate([np.zeros([len(train)]),
                               np.ones([P.shape[0]])])

        #x = U[test,:]
        #y = true.astype(int)[test]
        #y = Y_keep[test]

        #DEFINE MODEL
        np.random.seed(1)
        tf.random.set_seed(seeds[1])
        classifier = define_classifier(X.shape[1], num_layers=num_layers)

        #shuffle training data
        ind = np.arange(X.shape[0])
        np.random.seed(seeds[2])
        np.random.shuffle(ind)

        ##new v
        print("LR: " + str(puLR))
        auc = tf.keras.metrics.AUC(curve='PR', name='auc')
        classifier.compile(optimizer = tf.optimizers.Adam(learning_rate=puLR),
                           loss = 'binary_crossentropy',
                           metrics=[auc])


        tf.random.set_seed(seeds[3])
        hist = classifier.fit(x=X,
                              y=Y,
                              #validation_data=(x, y),
                              epochs=cls_eps, 
                              verbose=False,
                              use_multiprocessing=True)
        
        break
            
    return hist


def epoch_PU3(U, P, k, N, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1, stop_metric='ValAUC'):
    #PU BAGGING
    
    if(seeds==None):
        print('generating seeds')
        seeds = [42, 42, 42, 42]
    
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    preds= np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    #hists = np.zeros([N*k, cls_eps])
    #Y_keep=np.concatenate([np.zeros(U.shape[0]), np.ones(P.shape[0])])

    i = 0
    for test, train in rkf.split(U):
        
        i = i + 1
        print('')
        print(str(i) + '/' + str(N*k) + ' itterations')

        #t = true[true<2]
        #t = t[train]
        
        X = np.vstack([U[train,:], P])
        Y = np.concatenate([np.zeros([len(train)]),
                               np.ones([P.shape[0]])])

        #x = U[test,:]
        #y = true.astype(int)[test]
        #y = Y_keep[test]

        #DEFINE MODEL
        np.random.seed(1)
        tf.random.set_seed(seeds[1])
        classifier = define_classifier(X.shape[1], num_layers=num_layers)

        #shuffle training data
        ind = np.arange(X.shape[0])
        np.random.seed(seeds[2])
        np.random.shuffle(ind)

        ##new v
        print("LR: " + str(puLR))
        auc = tf.keras.metrics.AUC(curve='PR', name='auc')
        classifier.compile(optimizer = tf.optimizers.Adam(learning_rate=puLR),
                           loss = 'binary_crossentropy',
                           metrics=[auc])


        tf.random.set_seed(seeds[3])
        hist = classifier.fit(x=X,
                              y=Y,
                              #validation_data=(x, y),
                              validation_split=0.1,
                              epochs=cls_eps, 
                              verbose=False,
                              use_multiprocessing=True)
        
        break
            
    return hist



#def noPU(U, P, true, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1):
def noPU(U, P, cls_eps, clss='NN', seeds=None, puPat=5, puLR=1e-3, num_layers=1):
                    
                
    X = np.vstack([U, P])
    Y = np.concatenate([np.zeros([U.shape[0]]),
                           np.ones([P.shape[0]])])


    if(clss=='NN'):
        #DEFINE MODEL
        np.random.seed(1)
        tf.random.set_seed(seeds[1])
        classifier = define_classifier(X.shape[1], num_layers=num_layers)

        #shuffle training data
        ind = np.arange(X.shape[0])
        np.random.seed(seeds[2])
        np.random.shuffle(ind)

        ##new v
        print("LR: " + str(puLR))
        auc = tf.keras.metrics.AUC(curve='PR', name='auc')
        classifier.compile(optimizer = tf.optimizers.Adam(learning_rate=puLR), loss = 'binary_crossentropy')

        tf.random.set_seed(seeds[3])
        hist = classifier.fit(x=X,
                              y=Y,
                              epochs=cls_eps, 
                              verbose=False,
                              use_multiprocessing=True)



        tf.random.set_seed(seeds[4])
        preds = np.array(classifier(U)).flatten()
        tf.random.set_seed(seeds[4])
        preds_on_P = np.array(classifier(P)).flatten()
    else:
        neighbors = int(np.sqrt(X.shape[0]))
        knn = NearestNeighbors(n_neighbors=neighbors)
        knn.fit(X,Y)

        graph = knn.kneighbors_graph(U)
        preds = np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors)) #sum across rows

        graph = knn.kneighbors_graph(P)
        preds_on_P = np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors))

    return preds, preds_on_P









'''def PU(U, P, true, k, N, cls_eps, clss='NN', seeds=None):
    #PU BAGGING
    
    if(seeds==None):
        print('generating seeds')
        seeds = [42, 42, 42, 42]
    
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    preds= np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    hists = np.zeros([N*k, cls_eps])
    val_hists = np.zeros([N*k, cls_eps])
    auc_hists = np.zeros([N*k, cls_eps])
    val_auc = np.zeros([N*k, cls_eps])
    i = 0
    for test, train in rkf.split(U):
        
        i = i + 1
        print('')
        print(str(i) + '/' + str(N*k) + ' itterations')

        X = np.vstack([U[train,:], P])
        Y = np.concatenate([np.zeros([len(train)]),
                               np.ones([P.shape[0]])])

        x = U[test,:]
        y = true.astype(int)[test]

        if (clss=='NN'):
            #DEFINE MODEL
            np.random.seed(1)
            tf.random.set_seed(seeds[1])
            classifier = define_classifier(X.shape[1])

            #shuffle training data
            ind = np.arange(X.shape[0])
            np.random.seed(seeds[2])
            np.random.shuffle(ind)
            
            auc = tf.keras.metrics.AUC(curve='PR', name='auc')
            classifier.compile(optimizer = tf.optimizers.Adam(learning_rate=1e-3),
                          loss = 'binary_crossentropy', 
                          metrics=[auc])
            
            
            if ((X.shape[0] * 0.1) >= 50 ):
            
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=12345)


                callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc',#val_loss
                                                            mode = 'max',#min
                                                            min_delta=0, 
                                                            patience=5, 
                                                            verbose=True, 
                                                            restore_best_weights=True)

                
                tf.random.set_seed(seeds[3])
                hist = classifier.fit(x=X_train,
                                      y=Y_train,
                                      validation_data=(X_test, Y_test),
                                      epochs=cls_eps, 
                                      verbose=False,
                                      use_multiprocessing=True,
                                      callbacks=[callback])

                val_hists[i-1,:len(hist.history['val_loss'])]= hist.history['val_loss']
                val_auc[i-1,:len(hist.history['val_auc'])]= hist.history['val_auc']
            else:
                
                print('not enough cells for train test split')
                
                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                            min_delta=0.0, 
                                                            patience=5,#was 10
                                                            restore_best_weights=True)#was F
                ind = np.arange(X.shape[0])
                np.random.seed(seeds[2])
                np.random.shuffle(ind)
                
                tf.random.set_seed(seeds[3])
                #https://www.tensorflow.org/api_docs/python/tf/keras/Model
                hist = classifier.fit(x=X[ind,:], 
                                      y=Y[ind], 
                                      epochs=cls_eps, 
                                      verbose=False ,
                                      callbacks=[callback])
            
            hists[i-1,:len(hist.history['loss'])]= hist.history['loss']
            auc_hists[i-1,:len(hist.history['auc'])]= hist.history['auc']
                        
            tf.random.set_seed(seeds[4])
            preds[test] = preds[test] + np.array(classifier(x)).flatten()
            tf.random.set_seed(seeds[4])
            preds_on_P = preds_on_P + np.array(classifier(P)).flatten()
            
        if (clss == 'knn'):
            neighbors = 100
            knn = NearestNeighbors(n_neighbors=neighbors)
            knn.fit(X,Y)
            
            graph = knn.kneighbors_graph(x)
            preds[test] = preds[test] + np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors)) #sum across rows

            graph = knn.kneighbors_graph(P)
            preds_on_P = preds_on_P + np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors))
            
        if (clss == 'SVM'):
            clf = svm.SVC(kernel='linear')
            clf.fit(X, Y)
            
            preds[test] = preds[test] + clf.predict(x)#.flatten()
            preds_on_P = preds_on_P + clf.predict(P)#.flatten()

    preds = preds / ((i/k)*(k-1))
    preds_on_P = preds_on_P / ((i/k)*(k-1))

    return preds, preds_on_P, hists, val_hists, auc_hists, val_auc




'''











'''
sce_vaeda
#- classifier
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from sklearn.model_selection import RepeatedKFold
import sklearn

from classifier import define_classifier
from sklearn.neighbors import NearestNeighbors
from sklearn import svm

def PU(U, P, true, k, N, cls_eps, clss='NN', seeds=None):
    #PU BAGGING
    
    if(seeds==None):
        print('generating seeds')
        seeds = [42, 42, 42, 42]
    
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    preds= np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    hists = np.zeros([N*k, cls_eps])
    i = 0
    for test, train in rkf.split(U):
        
        i = i + 1
        print('')
        print(str(i) + '/' + str(N*k) + ' itterations')

        X = np.vstack([U[train,:], P])
        Y = np.concatenate([np.zeros([len(train)]),
                               np.ones([P.shape[0]])])

        x = U[test,:]
        y = true.astype(int)[test]

        if (clss=='NN'):
            #DEFINE MODEL
            np.random.seed(1)
            tf.random.set_seed(seeds[1])
            classifier = define_classifier(X.shape[1])

            #shuffle training data
            ind = np.arange(X.shape[0])
            np.random.seed(seeds[2])
            np.random.shuffle(ind)
            
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=10)
            tf.random.set_seed(seeds[3])
            #https://www.tensorflow.org/api_docs/python/tf/keras/Model
            hist = classifier.fit(x=X[ind,:], y=Y[ind], epochs=cls_eps, verbose=False ,callbacks=[callback])

            hists[i-1,:len(hist.history['loss'])]= hist.history['loss']
                        
            tf.random.set_seed(seeds[4])
            preds[test] = preds[test] + np.array(classifier(x)).flatten()
            tf.random.set_seed(seeds[4])
            preds_on_P = preds_on_P + np.array(classifier(P)).flatten()
            
        if (clss == 'knn'):
            neighbors = 100
            knn = NearestNeighbors(n_neighbors=neighbors)
            knn.fit(X,Y)
            
            graph = knn.kneighbors_graph(x)
            preds[test] = preds[test] + np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors)) #sum across rows

            graph = knn.kneighbors_graph(P)
            preds_on_P = preds_on_P + np.squeeze(np.array(np.sum(graph[:,Y==1], axis=1) / neighbors))
            
        if (clss == 'SVM'):
            clf = svm.SVC(kernel='linear')
            clf.fit(X, Y)
            
            preds[test] = preds[test] + clf.predict(x)#.flatten()
            preds_on_P = preds_on_P + clf.predict(P)#.flatten()

    preds = preds / ((i/k)*(k-1))
    preds_on_P = preds_on_P / ((i/k)*(k-1))

    return preds, preds_on_P, hists



'''