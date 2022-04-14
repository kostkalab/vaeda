#- classifier
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
    

    
    
def define_classifier(ngens, seed=1, num_layers=1): 
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions
    
    if(num_layers==1):
        classifier = tfk.Sequential([
            tfkl.InputLayer(input_shape=[ngens]),
            tfkl.BatchNormalization(),
            tfkl.Dense(1, activation='sigmoid')#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
        ])
    if(num_layers==2):
        print('using 2 layers in classifier')
        classifier = tfk.Sequential([
            tfkl.InputLayer(input_shape=[ngens]),
            tfkl.BatchNormalization(),
            tfkl.Dense(3, activation='relu'),
            tfkl.Dense(1, activation='sigmoid')#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
        ])

    model = tfk.Model(inputs=classifier.inputs,
                      outputs=classifier.outputs[0])

    return model
    
    
    
    
    
'''def define_classifier(ngens, seed=1, num_layers=1): 
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions
    
    if(num_layers==1):
        classifier = tfk.Sequential([
            tfkl.InputLayer(input_shape=[ngens]),
            tfkl.BatchNormalization(),
            tfkl.Dense(1, activation='sigmoid')#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
        ])
    if(num_layers==2):
        print('using 2 layers in classifier')
        classifier = tfk.Sequential([
            tfkl.InputLayer(input_shape=[ngens]),
            tfkl.BatchNormalization(),
            tfkl.Dense(3, activation='relu'),
            tfkl.Dense(1, activation='sigmoid')#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
        ])

    model = tfk.Model(inputs=classifier.inputs,
                      outputs=classifier.outputs[0])

    return model    
    '''
    
    
    
    
    
    
'''    
def define_classifier(ngens, seed=1): 
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions
    
    classifier = tfk.Sequential([
        tfkl.InputLayer(input_shape=[ngens]),
        tfkl.BatchNormalization(),
        tfkl.Dense(1, activation='sigmoid')#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
    ])

    model = tfk.Model(inputs=classifier.inputs,
                      outputs=classifier.outputs[0])

    return model


'''    
    
    
    
    
    
'''
    
    
#sce_vaeda
def define_classifier(ngens, seed=1): 
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions
    
    classifier = tfk.Sequential([
        tfkl.InputLayer(input_shape=[ngens]),
        tfkl.BatchNormalization(),
        tfkl.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
    ])

    model = tfk.Model(inputs=classifier.inputs,
                      outputs=classifier.outputs[0])

    model.compile(optimizer = tf.optimizers.Adam(learning_rate=1e-3),
                  #loss = 'mse')
                  loss = 'binary_crossentropy')

    return model

'''