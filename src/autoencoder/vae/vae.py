'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function
from plotload import load_polyp_data
from plotload import plot_1_to_255
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D,AveragePooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import tensorflow as tf

batch_size = 100
original_dim = 784

intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

class VAE():
    def __init__(self):
        self.img_rows = 720//4#240 # Original is ~720 
        self.img_cols = 576//4#192 # Original is ~576
        self.channels = 3   # RGB        
        self.img_shape=(self.img_rows,self.img_cols,self.channels)
        self.latent_dim=(100,)
        #self.latent_dim=tf.placeholder(tf.float32, shape=self.latent_dim, name="latentdim")
        self.latent_dim_int=self.latent_dim[0] #only works on flat latent_dims (for now!!)
        
    
        (input_encoder,output_encoder,self.encoder,z_mean,z_log_var)=self.build_encoder(self.img_shape,self.latent_dim_int)
        (input_decoder,output_decoder,self.decoder)=self.build_decoder(output_decoder)

        
        vae=Model(input_encoder,output_decoder)
        
        # Compute VAE loss
        xent_loss = self.img_rows*self.img_cols*self.channels * metrics.binary_crossentropy(input_encoder, output_decoder)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        
        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')
        vae.summary()        




    def build_encoder(self,img_shape,latent_dim):
        
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                      mean=0., stddev=epsilon_std)
            return z_mean + K.exp(z_log_sigma) * epsilon            
        
        input_img = Input(shape=(img_shape))
        x = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.5)(x)
        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Conv2D(1, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = AveragePooling2D((3, 3), padding='same')(x)
        h = Flatten()(x)  
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)
        encoded=Lambda(sampling, output_shape=self.latent_dim)([z_mean,z_log_var])
        Encoder=Model(input_img,encoded,name='encoder')
        return input_img,encoded,Encoder,z_mean,z_log_var        
        
    def build_decoder(self,encoded):
        """
        Decoder
        :param encoded: the layer that is the input of the decoder (typically the encoded layer from the encoder)
        :return: returns "input_code" wich is the input layer, "decoded" which is the last layer, and the model.
        """
        if type(encoded)==type((1,1)):        
            s=encoded
        else:
            s=encoded.get_shape().as_list()[1:]
        
        input_code=Input(shape=s)
        x = Dense(540)(input_code)
        x = Reshape((720//48,576//48,3))(x)
        x = Conv2D(1, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D((3, 3))(x)
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        #x = BatchNormalization(momentum=0.8)(x)
        #x = UpSampling2D((2, 2))(x)
        #x = Dropout(0.25)(x)
        #x = Conv2D(8, (3, 3), padding='same')(x)
        #x = LeakyReLU()(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
        Decoder=Model(input_code,decoded,name='decoder')
        return input_code,decoded,Decoder


    def build_VAE(inp,out,opt='adam',l='mse'):
        #AE=Sequential()
        #AE.add(e)
        #AE.add(d)
        #AE.compile(optimizer=opt, loss=l)
        return  Model(inp, out)


    def train(self, epochs=20, batch_size=32, save_interval=5):
        """
        Trainer: uses the self.autoencoder and the inputed dataset to train the wights
        It does also save a sample every save interval
        :param epochs: number of epochs run
        :param batch_size: how many imgs in each batch
        :param save_interval: how many epochs between each save

        """
        X_train=load_polyp_data(self.img_shape)
        loss=100
        for epoch in tqdm(range(epochs)):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx] 

            if (epoch+2) % save_interval == 0 and loss<0.05:
                img=np.clip((np.random.normal(imgs,0.1)),-1,1)
            else:
                img=imgs
            loss=self.autoencoder.train_on_batch(imgs, img)
            if epoch % save_interval == 0:
                idx2 = np.random.randint(0, X_train.shape[0], batch_size)
                imgs2 = X_train[idx]
                loss2=self.autoencoder.test_on_batch(imgs2, imgs2)
                print(loss,loss2)
                self.save_imgs(epoch,imgs[0:3,:,:,:]) 
                self.decoder.save_weights(f"decoder_weights_{epoch}.h5")
                self.encoder.save_weights(f"encoder_weights_{epoch}.h5")
                self.autoencoder.save_weights(f"ae_weights_{epoch}.h5")
        # encode and decode some digits
        # note that we take them from the *test* set
        print("saving")
        self.decoder.save("new_decoder.h5")
        self.encoder.save("new_encoder.h5")
        self.autoencoder.save("new_ae.h5")
        self.decoder.save_weights("decoder_weights.h5")
        self.encoder.save_weights("encoder_weights.h5")
        self.autoencoder.save_weights("ae_weights.h5")


a=VAE()
a.train()
sys.exit()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

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
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()