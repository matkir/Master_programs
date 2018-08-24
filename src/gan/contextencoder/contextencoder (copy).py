from __future__ import print_function, division
from tqdm import tqdm

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
        """
        Initializes the contextencoder. 
        """
        self.set_training_info()
        globals().update(self.info)  
        self.threshold=threshold
        self.img_cols = 256 # Original is ~576
        self.img_rows = 356 # Original is ~720 
        self.channels = 3   # RGB 
        self.img_shape=(self.img_cols,self.img_rows,self.channels)
        self.combined=None
        self.discriminator=None
        self.pretrained=False
    def load_model(self,adress):
        """
        loads a model to the object instead of creating one. 
        :param adress: string of adress to the file of type h5.
        """
        if self.model!=None:
            print("Warning: overriding a loaded model")
        self.model=load_model(adress) 
    def load_model_weights(self,adress):
        """
        loads weights to the model. 
        :param model:  the model that is to get weights
        :param adress: string of adress to the file of type h5.
        :returns:      model with weights
        """
        if self.model==None:
            print("Error: no model in object")
        else:
            try:
                self.model.load_weights(adress)
                self.pretrained=True
            except e:
                print("Error: weights could not be loaded")
                print(e)
    
    def build_model(self):
        """
        builds a model to the object instead of loading one. 
        Uses AE_weights.py as model
        """
        if self.combined!=None:
            print("Warning: overriding a loaded model")
        wm=Weight_model(self.img_cols,self.img_rows)
        self.discriminator,self.generator,self.combined=wm.build_CE()  
        
    def set_training_info(self):
        self.info={}
        import sys
        try:
            if len(sys.argv)==1:
                choise=2
            else:
                choise=int(input("press 1 for last run or 2 for info.txt "))
        except:
            choise=False
        
        if choise==1:
            self.info=np.load("temp_info.npy").item()
            return
        elif choise==2:
            with open("info.txt") as f:
                for line in f:
                    (key, val) = line.split()
                    try:
                        self.info[key] = int(val)
                    except:
                        self.info[key] = float(val)
            np.save("temp_info.npy", self.info)
            return
        else:        
            self.info["mask"]=int(input("Mask [1] or corner [0]? "))
            if self.info['mask']==1:
                tmp=input("Mask adress? (default: /masks) ")
                self.info["mask_folder"]=tmp if isinstance(tmp, str) else "/masks"
            self.info["epochs"]=int(input("Number of epochs? "))
            self.info["batch_size"]=int(input("Batch size? "))
            self.info["save_interval"]=int(input("save interval? "))
            np.save("temp_info.npy", self.info)    
    
    
    
    
    
    
        


    def train(self):
        if self.info==None:
            print("Warning no info found, prompting for info")
            self.set_training_info()
        globals().update(self.info)
        if self.combined==None:
            print("Error: no model loaded")
            return
        if self.pretrained==True:
            print("Warning: model has pretrained weights")
        half_batch = int(batch_size / 2)
        for epoch in tqdm(range(epochs)):                
        
            # ---------------------
            #  Train Discriminator
            # ---------------------

                
            X_train=plotload.load_polyp_batch(self.img_shape, batch_size, data_type='none',crop=True)
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

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

            if g_loss[1]<self.threshold:
                self.threshold=g_loss[1]
                self.model.save(f"models/CE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.model.save_weights(f"models/CE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w.h5") 
                
    def extra():
        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            # Select a random half batch of images
            imgs=plotload.load_polyp_batch(self.img_shape, 4, 
                                         data_type='green', 
                                         crop=True)
            self.sample_images(epoch, imgs)
 
        
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

   

if __name__ == '__main__':
    context_encoder = ContextEncoder()
    context_encoder.build_model()
    context_encoder.train()
    
    #context_encoder.train(epochs=2000, batch_size=320, sample_interval=100)
    #imgs=plotload.load_polyp_batch((576//6,720//6,3), 4, data_type='green', crop=False)    
    #context_encoder.sample_images(2, imgs)
    #context_encoder.train(epochs=10000, batch_size=1024, sample_interval=100)


