from __future__ import print_function, division

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
import plotload
import sys
from cc_weights import Weight_model
from selector import Selector
from masker import mask_from_template,mask_randomly_square,mask_green_corner,combine_imgs_with_mask
import matplotlib.pyplot as plt

import numpy as np

class CCgan():
    def __init__(self):
        self.img_rows = 576#8*64//2#32
        self.img_cols = 720#8*64//2#32
        # mask idealy 170 * 215  
        self.mask_width = 170#128#208
        self.mask_height = 215#160#280
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_width, self.mask_height, self.channels)
        self.model=Weight_model(self.img_rows, self.img_cols, self.mask_width,self.mask_height)

        optimizer = Adam(0.0002, 0.5)


        self.discriminator = self.model.build_discriminator()
        self.generator = self.model.build_generator()
       
        if '-weights' in sys.argv:
            print("loading old weights")
            self.generator.load_weights("saved_model/generator_weigths.h5")
            self.discriminator.load_weights("saved_model/discriminator_weigths.h5")
            
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=optimizer)

        masked_img = Input(shape=self.img_shape)
        gen_img = self.generator(masked_img)

        self.discriminator.trainable = False

        valid= self.discriminator(gen_img)

        self.combined = Model(masked_img , [gen_img, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
            optimizer=optimizer)        

        if "-save" in sys.argv:
            self.generator.save("saved_model/generator.h5")
            self.discriminator.save("saved_model/discriminator.h5")        


   

    def train(self, epochs, batch_size=2, save_interval=50):
        from tqdm import tqdm
        soft= True if '-soft' in sys.argv else False
        half_batch=batch_size//2
        for epoch in tqdm(range(epochs)):
            if epoch%100==0:
                X_train=plotload.load_polyp_batch(self.img_shape, batch_size*5)


            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            masked_imgs, missing, mask = mask_from_template(imgs)
            gen_fake = self.generator.predict(missing)
            gen_fake = combine_imgs_with_mask(gen_fake, imgs, mask)


            # SPESSSIELLE TING
            if soft:
                valid = 0.2*np.random.random_sample((half_batch,1))+0.9
                fake = 0.1*np.random.random_sample((half_batch,1))
            else:
                valid = np.ones((half_batch, 1))
                fake = np.zeros((half_batch, 1))

            if epoch%120==0 and epoch!=0:
                #small shakeup to get out of local minimas
                placeholder=valid
                valid=fake
                fake=placeholder


            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            #splising fake imgs
            d_loss_fake = self.discriminator.train_on_batch(gen_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            masked_imgs, missing_parts, _ = mask_from_template(imgs)

            valid = np.ones((batch_size, 1))

            # Train the generator
            g_loss = self.combined.train_on_batch(masked_imgs, [imgs, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))



            if epoch % save_interval == 0:
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)
            if epoch % (save_interval*5) == 0:
                self.save_model() 

    def sample_images(self, epoch, imgs):
        r, c = 3, 6

        masked_imgs, missing_parts, m = mask_from_template(imgs)   
        gen_fake1 = self.generator.predict(missing_parts)
        gen_fake = combine_imgs_with_mask(gen_fake1, imgs, m)
        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_fake = 0.5 * gen_fake + 0.5
        gen_fake1 = 0.5 * gen_fake1 + 0.5

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0,i].imshow(imgs[i, :,:])
            axs[0,i].axis('off')
            axs[1,i].imshow(gen_fake[i, :,:])
            axs[1,i].axis('off')
            axs[2,i].imshow(gen_fake1[i,:,:])
            axs[2,i].axis('off')
        fig.savefig("images/cc_%d.png" % epoch)
        plt.close()

    def save_model(self):
        self.generator.save_weights("saved_model/generator_weigths.h5")
        self.discriminator.save_weights("saved_model/discriminator_weigths.h5")



if __name__ == '__main__':
    cc = CCgan()
    cc.train(epochs=30000, batch_size=12, save_interval=100)




