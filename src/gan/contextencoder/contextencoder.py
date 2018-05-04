from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

class ContextEncoder():
    def __init__(self):
        self.img_rows = 576#8*64//2#32
        self.img_cols = 720#8*64//2#32
        self.mask_width = 208#350 #self.img_rows//4 #8*16//2#8
        self.mask_height = 280#300 #self.img_cols//4#8*16//2#8
        self.channels = 3
        self.num_classes = 2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_width, self.mask_height, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator_img_size()
        self.generator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

        if '-weights' in sys.argv:
            self.generator.load_weights("saved_model/generator_weigths.h5")
            self.discriminator.load_weights("saved_model/discriminator_weigths.h5")
        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator) takes
        # masked_img as input => generates missing image => determines validity
        self.combined = Model(masked_img , [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)
        self.generator.save("saved_model/generator.h5")
        self.discriminator.save("saved_model/discriminator.h5")

    def build_generator_img_size(self):


        model = Sequential()

        # Encoder
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dropout(0.5))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=3, strides=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(512, kernel_size=1, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        # Decoder
        #model.add(UpSampling2D(size=(3,3)))
        model.add(Conv2D(256, kernel_size=3))
        #model.add(LeakyReLU(alpha=0.2))
        model.add(Activation('relu'))
        
        model.add(UpSampling2D(size=(3,3)))
        model.add(Conv2D(256, kernel_size=3))
        #model.add(LeakyReLU(alpha=0.2))
        model.add(Activation('relu'))
        
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3))
        #model.add(LeakyReLU(alpha=0.2))
        model.add(Activation('relu'))
       
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3))
        #model.add(LeakyReLU(alpha=0.2))
        model.add(Activation('relu'))
        
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Dropout(0.5))
        model.add(Conv2D(32, kernel_size=3))
        #model.add(LeakyReLU(alpha=0.2))
        model.add(Activation('relu'))        
        
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3))
        model.add(Activation('tanh'))

        model.summary()

        masked_img = Input(shape=self.img_shape)
        gen_missing = model(masked_img)

        return Model(masked_img, gen_missing)


    def build_generator(self):


        model = Sequential()

        # Encoder
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dropout(0.5))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=3, strides=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(512, kernel_size=1, strides=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        # Decoder
        model.add(UpSampling2D(size=(3,3)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        
        model.add(UpSampling2D(size=(3,3)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
       
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Dropout(0.5))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation('tanh'))

        model.summary()

        masked_img = Input(shape=self.img_shape)
        gen_missing = model(masked_img)

        return Model(masked_img, gen_missing)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.missing_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
       
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.missing_shape)
        validity = model(img)

        return Model(img, validity)

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_cols - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width
	
        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty((imgs.shape[0],self.mask_width,self.mask_height, self.channels))
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            missing_parts[i] = masked_img[_x1:_x2, _y1:_y2, :].copy()
            masked_img[_x1:_x2, _y1:_y2, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs, missing_parts, (y1, y2, x1, x2)


    def train(self, epochs, batch_size=128, sample_interval=50):
        half_batch = int(batch_size / 2)
        import plotload
        """
        X_train=plotload.load_polyp_data(self.img_shape)
        """
        import sys
        soft= True if '-soft' in sys.argv else False
        numtimes=np.zeros(batch_size*5)
        for epoch in range(epochs):


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            if epoch%100==0:
                print(f"most used picture was traned on {max(numtimes)} times")
                numtimes=np.zeros(batch_size*5)        
                X_train=plotload.load_polyp_batch(self.img_shape, batch_size*5)
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            numtimes[idx]+=1 #to count num of times each pic was trained on
            
            masked_imgs, missing, _ = self.mask_randomly(imgs)

            # Generate a half batch of new images
            gen_missing = self.generator.predict(masked_imgs)

            if soft:
                valid = 0.2*np.random.random_sample((half_batch,1))+0.9
                fake = 0.1*np.random.random_sample((half_batch,1))
            else:
                valid = np.ones((half_batch, 1))
                fake = np.zeros((half_batch, 1))

            if epoch%120==0:
                #small shakeup to get out of local minimas
                placeholder=valid
                valid=fake
                fake=placeholder
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(missing, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(1):
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
        
                masked_imgs, missing_parts, _ = self.mask_randomly(imgs)
        
                # Generator wants the discriminator to label the generated images as valid
                valid = np.ones((batch_size, 1))
        
                # Train the generator
                g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)
            if epoch % (sample_interval*5) == 0:
                self.save_model()   
    def sample_images(self, epoch, imgs):
        r, c = 3, 6

        masked_imgs, missing_parts, (y1, y2, x1, x2) = self.mask_randomly(imgs)
        gen_missing = self.generator.predict(masked_imgs)

        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_missing = 0.5 * gen_missing + 0.5

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0,i].imshow(imgs[i, :,:])
            axs[0,i].axis('off')
            axs[1,i].imshow(masked_imgs[i, :,:])
            axs[1,i].axis('off')
            filled_in = imgs[i].copy()
            filled_in[x1[i]:x2[i], y1[i]:y2[i], :] = gen_missing[i]
            axs[2,i].imshow(filled_in)
            axs[2,i].axis('off')
        fig.savefig("images/cifar_%d.png" % epoch)
        plt.close()

    def save_model(self):
        self.generator.save_weights("saved_model/generator_weigths.h5")
        self.discriminator.save_weights("saved_model/discriminator_weigths.h5")


if __name__ == '__main__':
    context_encoder = ContextEncoder()
    context_encoder.train(epochs=90000, batch_size=32, sample_interval=50)

