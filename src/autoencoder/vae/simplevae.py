from __future__ import print_function
from plotload import load_polyp_data
from plotload import plot_1_to_255
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


from keras import backend as K
from keras import metrics
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D,AveragePooling2D, UpSampling2D, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model



img_rows = 720//4#240 # Original is ~720 
img_cols = 576//4#192 # Original is ~576
channels = 3   # RGB        
img_shape=(img_rows,img_cols,channels)
latent_dim=100
batch_size=100
epsilon_std=1.

print("loading data")
data=load_polyp_data(img_shape)

def sampling(args):
           #sampling used for lambda layer, #TODO find out what it does.
           z_mean, z_log_sigma = args
           epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                     mean=0., stddev=epsilon_std)
           return z_mean + K.exp(z_log_sigma) * epsilon
      

input_img = Input(shape=(img_shape))
x = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
x = LeakyReLU()(x)
x = AveragePooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = AveragePooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
x = AveragePooling2D((3, 3), padding='same')(x)
h = Flatten()(x) 

#divide network in 2
z_mean = Dense(latent_dim)(h) 
z_log_var = Dense(latent_dim)(h)
encoded=Lambda(sampling, output_shape=(latent_dim,))([z_mean,z_log_var])
x = Dense(540)(encoded)
x = Reshape((720//48,576//48,3))(x)
x = Conv2D(1, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = BatchNormalization(momentum=0.8)(x)
x = UpSampling2D((3, 3))(x)
x = Conv2D(8, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = BatchNormalization(momentum=0.8)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
 
vae=Model(input_img, decoded)

xent_loss = img_cols * img_rows * channels * metrics.binary_crossentropy(input_img,decoded)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()
vae.train_on_batch(data, data)