#- vae
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def define_clust_vae(enc_sze, ngens, num_clust, LR=1e-3, clust_weight=10000):
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions

    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
            reinterpreted_batch_ndims=1)
    
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[ngens]),
        tfkl.Dense(256, activation='relu'),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None),
        tfpl.IndependentNormal(
            enc_sze,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
        )
    ], name='encoder')

    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.Dense(256, activation='relu'),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None),
        tfpl.IndependentNormal(ngens)
    ], name='decoder')

    clust_classifier = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.BatchNormalization(),
        tfkl.Dense(num_clust, activation='sigmoid')
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
  
    return vae

    
def define_vae(enc_sze, ngens):
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions

    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
            reinterpreted_batch_ndims=1)
    
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[ngens]),
        tfkl.Dense(256, activation='relu'),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None),
        tfpl.IndependentNormal(
            enc_sze,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
        )
    ], name='encoder')

    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.Dense(256, activation='relu'),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None),
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

    return vae



    
    
    
