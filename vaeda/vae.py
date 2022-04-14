#- vae
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
  


def define_clust_vae(enc_sze, ngens, num_clust, LR=1e-3, clust_weight=10000):
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions

    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
            reinterpreted_batch_ndims=1)
    
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[ngens]),
        tfkl.Dense(256, activation='relu'),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[0])),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[1])),
        tfpl.IndependentNormal(
            enc_sze,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
        )
    ], name='encoder')

    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.Dense(256, activation='relu'),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[2])),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[3])),
        tfpl.IndependentNormal(ngens)
    ], name='decoder')

    clust_classifier = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.BatchNormalization(),
        tfkl.Dense(num_clust, activation='sigmoid')#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
    ], name='clust_classifier')
    
    IPT     = tfk.Input(shape = ngens)
    z       = encoder(IPT)
    OPT1    = decoder(z)
    OPT2    = clust_classifier(z)

    vae = tfk.Model(inputs=[IPT],
                      outputs=[OPT1, OPT2])
    
    def nll(x, rv_x): 
        rec = rv_x.log_prob(x)
        return -tf.math.reduce_sum(rec, axis=-1) 
    
    vae.compile(optimizer = tf.optimizers.Adamax(learning_rate=LR),#Adam, 1e-3
                  loss=[nll, 'categorical_crossentropy'], loss_weights=[1,clust_weight])
  
    #1e-3
    return vae


def plt_loss(hist):
    x = np.arange(len(hist.history['loss']))
    keysv = list(hist.history.keys())
    
    fig, axes = plt.subplots(1, 4, sharex=True, figsize=(20,5))

    plt.subplot(1, 4, 1)
    plt.plot(x, np.array(hist.history['loss']), label='loss')
    plt.plot(x, np.array(hist.history['val_loss']), label='val_loss')
    plt.title('total loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.subplot(1, 4, 2)
    plt.plot(x, np.array(hist.history['decoder_loss']), label='decoder_loss')
    plt.plot(x, np.array(hist.history['val_decoder_loss']), label='val_decoder_loss')
    plt.title('vae loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
                     
    plt.subplot(1, 4, 3)
    plt.plot(x, np.array(hist.history['clust_classifier_loss']), label='clust_classifier_loss')
    plt.plot(x, np.array(hist.history['val_clust_classifier_loss']), label='val_clust_classifier_loss')
    plt.title('classifier loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.subplot(1, 4, 4)
    plt.plot(x, np.array(hist.history['lr']), label='lr')
    plt.title('Learning Rate')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('LR')
    
    
    
    
    

    
#vae_embeddings_schedules_adamax_5
def define_vae(enc_sze, ngens):
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions

    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
            reinterpreted_batch_ndims=1)
    
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[ngens]),
        tfkl.Dense(256, activation='relu'),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[0])),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[1])),
        tfpl.IndependentNormal(
            enc_sze,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
        )
    ], name='encoder')

    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.Dense(256, activation='relu'),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[2])),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[3])),
        tfpl.IndependentNormal(ngens)
    ], name='decoder')


    IPT     = tfk.Input(shape = ngens)
    z       = encoder(IPT)
    OPT1    = decoder(z)

    vae = tfk.Model(inputs=[IPT],
                      outputs=[OPT1])
    
    def nll(x, rv_x): 
        rec = rv_x.log_prob(x)
        return -tf.math.reduce_sum(rec, axis=-1) 
    
    vae.compile(optimizer = tf.optimizers.Adamax(learning_rate=1e-3),#Adam
                  loss=nll)
    #1e-3
    return vae


def plt_loss(hist):
    x = np.arange(len(hist.history['loss']))
    keysv = list(hist.history.keys())
    
    for k in keysv[:-1]:
        plt.plot(x, np.array(hist.history[k]), label = k)
    plt.title('loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')    
    
    
    
    
    

    
    
'''
#sce_vaeda_out
def define_vae(enc_sze, ngens):
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions

    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
            reinterpreted_batch_ndims=1)
    
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[ngens]),
        tfkl.Dense(256, activation='relu'),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[0])),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[1])),
        tfpl.IndependentNormal(
            enc_sze,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
        )
    ], name='encoder')

    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.Dense(256, activation='relu'),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[2])),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[3])),
        tfpl.IndependentNormal(ngens)
    ], name='decoder')


    IPT     = tfk.Input(shape = ngens)
    z       = encoder(IPT)
    OPT1    = decoder(z)

    vae = tfk.Model(inputs=[IPT],
                      outputs=[OPT1])
    
    def nll(x, rv_x): 
        rec = rv_x.log_prob(x)
        return -tf.math.reduce_sum(rec, axis=-1) 
    
    vae.compile(optimizer = tf.optimizers.Adam(learning_rate=1e-3),
                  loss=nll)
    
    return vae


def plt_loss(hist):
    x = np.arange(len(hist.history['loss']))
    keysv = list(hist.history.keys())
    plt.figure(0)
    plt.plot(x, np.array(hist.history['loss']), label = 'loss')
    plt.title('loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

'''