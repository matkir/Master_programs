from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# input image dimensions
img_rows, img_cols, channels = 720//4, 576//4, 3

if K.image_data_format() == 'channels_first':
    original_img_size = (channels, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, channels)
output_shape = original_img_size
latent_dim = 100
intermediate_dim = 128
epsilon_std = 1.0
epochs = 5





x = Input(shape=original_img_size)

conv_1 = Conv2D(8,kernel_size=(2, 2),padding='same', activation='tanh')(x)

conv_2 = Conv2D(16,kernel_size=(2, 2),padding='same', activation='relu',strides=(2, 2))(conv_1)

conv_3 = Conv2D(32,kernel_size=(2, 2),padding='same', activation='relu',strides=1)(conv_2)

conv_4 = Conv2D(32,kernel_size=(2, 2),padding='same', activation='relu',strides=1)(conv_3)

flat = Flatten()(conv_4)

hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])






# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(img_cols*img_rows, activation='relu')
decoder_reshape = Reshape((img_rows,img_cols,1))
decoder_deconv_1 = Conv2DTranspose(32,kernel_size=3,padding='same',strides=1,activation='relu')
decoder_deconv_2 = Conv2DTranspose(16,kernel_size=3,padding='same',strides=1,activation='relu')
decoder_deconv_3_upsamp = Conv2DTranspose(8,kernel_size=(2, 2),strides=1,padding='same',activation='tanh')
decoder_mean_squash = Conv2D(3,kernel_size=2,padding='same',activation='tanh')



hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# instantiate VAE model
vae = Model(x, x_decoded_mean_squash)

"""# Compute VAE loss
xent_loss = img_rows * img_cols * channels * metrics.binary_crossentropy(
    K.flatten(x),
    K.flatten(x_decoded_mean_squash))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
#vae.add_loss(vae_loss)
"""
def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)

    return recon + kl

vae.compile(optimizer='adam',loss=vae_loss)
vae.summary()


from plotload import load_polyp_data
from plotload import plot_1_to_255
from tqdm import tqdm
x_train=load_polyp_data((img_rows, img_cols, channels))
vae.fit(x_train, x_train, batch_size=10, nb_epoch=5)
"""for i in tqdm(range(100)):
    idx = np.random.randint(0, x_train.shape[0], 10)
    imgs = x_train[idx]     
    vae.train_on_batch(imgs,imgs)#, [[1],[0],[0],[0],[1],[0],[0],[1],[1],[1]])
"""

print("trained")























# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
"""x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
"""
# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 100)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()