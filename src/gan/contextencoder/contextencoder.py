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
import sys
from context_weights import Weight_model
import masker as ms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotload
import sys
import numpy as np

class ContextEncoder():
    def __init__(self):
        self.img_rows = 720//4#8*64//2#32
        self.img_cols = 576//4#8*64//2#32
        self.channels = 3
        self.img_shape = (self.img_cols, self.img_rows, self.channels)

        #to find mask size we need to make a dummy img based on input dims.
        if '-corner' in sys.argv:
            dummy=plotload.load_one_img(self.img_shape, dest='none',extra_dim=True)
            a, b, dims = ms.mask_green_corner(dummy)
            self.mask_width = dims[3]-dims[2]
            self.mask_height = dims[1]-dims[0]
            corner=True
        else:        
            self.mask_width = 62#208
            self.mask_height = 51#280
            corner=False
        
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)
        self.model=Weight_model(self.img_cols, self.img_rows, self.mask_height,self.mask_width,corner)

        optimizer = Adam(0.0002, 0.5)
        
        #Build and compile the discriminator
        self.discriminator = self.model.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.model.build_generator_img_size()
        self.generator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

        if '-weights' in sys.argv:
            print("loading old weights")
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
            loss_weights=[0.7, 0.3],
            optimizer=optimizer)
        if "-save" in sys.argv:
            self.generator.save("saved_model/generator.h5")
            self.discriminator.save("saved_model/discriminator.h5")



    def train(self, epochs, batch_size=128, sample_interval=50):
        half_batch = int(batch_size / 2)

        """
        X_train=plotload.load_polyp_data(self.img_shape)
        """
        soft= True if '-soft' in sys.argv else False
        corner= True if '-corner' in sys.argv else False
        numtimes=np.zeros(batch_size)
        from keras.callbacks import TensorBoard
        board=TensorBoard()
        board.set_model(self.discriminator)
        for epoch in range(epochs):


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
                
            if epoch%100==0:
                print(f"most used picture was traned on {max(numtimes)} times")
                numtimes=np.zeros(batch_size)        
                X_train=plotload.load_polyp_batch(self.img_shape, batch_size, data_type='none',crop=True)
            if epoch%50==0 and not corner:
                #after 50 itterations we flip the images, to make the set 2x times as large. sorry for not vectorizing
                for i in range(batch_size):
                    X_train[i]=np.fliplr(X_train[i])
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            numtimes[idx]+=1 #to count num of times each pic was trained on
            
            if corner:    
                masked_imgs, missing, _ = ms.mask_green_corner(imgs)
            else:
                masked_imgs, missing, _ = ms.mask_randomly_square(imgs, 
                    self.mask_height, 
                    self.mask_width)

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
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
    
            if corner:
                masked_imgs, missing_parts, _ = ms.mask_green_corner(imgs)
            else:
                masked_imgs, missing_parts, _ = ms.mask_randomly_square(imgs,self.mask_height,self.mask_width)
                
            # Generator wants the discriminator to label the generated images as valid
            valid = np.ones((batch_size, 1))
    
            # Train the generator
            g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Select a random half batch of images
                imgs=plotload.load_polyp_batch(self.img_shape, 4, 
                                         data_type='green', 
                                         crop=True)
                self.sample_images(epoch, imgs)
            if epoch % (sample_interval*5) == 0:
                self.save_model()   
    def sample_images(self, epoch, imgs):
        r, c = 3, 4

        #masked_imgs, missing_parts, (y1, y2, x1, x2) = ms.mask_randomly_square(imgs,self.mask_height,self.mask_width)
        masked_imgs, missing_parts, (y1, y2, x1, x2) =ms.mask_green_corner(imgs)   
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
            filled_in[x1:x2, y1:y2, :] = gen_missing[i]
            axs[2,i].imshow(filled_in)
            axs[2,i].axis('off')
        fig.savefig("images/cifar_%d.png" % epoch)
        plt.close()

    def save_model(self):
        self.generator.save_weights("saved_model/generator_weigths.h5")
        self.discriminator.save_weights("saved_model/discriminator_weigths.h5")


if __name__ == '__main__':
    context_encoder = ContextEncoder()
    context_encoder.train(epochs=2000, batch_size=320, sample_interval=100)

