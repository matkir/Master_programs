from encdec.EncDec import build_AE,build_decoder,build_encoder,build_discriminator,build_GAN
from plotload.PlotLoad import plot_1_to_255, load_polyp_data
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import stats
from keras.optimizers import Adam
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class GAN(): 
    def __init__(self):
        self.img_rows = 720//4#240 # Original is ~720 
        self.img_cols = 576//4#192 # Original is ~576
        self.channels = 3   # RGB        
        self.img_shape=(self.img_rows,self.img_cols,self.channels)
        self.latent_dim=(100,)
        self.latent_dim_int=self.latent_dim[0] #only works on flat latent_dims (for now!!)
        
        optimizer = Adam(0.0002, 0.5)
        
        (I_gen,O_gen,self.generator)=build_decoder(self.latent_dim)
        (I_disc,O_disc,self.discriminator)=build_discriminator(self.img_shape)
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.combined=build_GAN(self.generator,self.discriminator,opt=optimizer,l='binary_crossentropy')
        
        #adding the ae to decode the img,for fun
        _,_,self.tester=build_decoder(self.latent_dim)
        #self.tester.load_weights("../../autoencoder/dcae/decoder_weights.h5")
        
        if '-w' in sys.argv:
            self.discriminator.load_weights("discriminator_weights.h5")
            self.generator.load_weights("generator_weights.h5")
            #self.combined.load_weights("combined_weights.h5")
           
               

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(180//4*144*3, activation="tanh", input_shape=noise_shape))
        model.add(Reshape((180//2, 144//2, 3)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size=2, padding="same"))
        model.add(Activation("tanh"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=4, padding="same"))
        model.add(Dropout(0.5))
        model.add(Activation("tanh"))
        if '-s' in sys.argv:
            model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        r= Model(noise, img)
        return r
    
    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=1, input_shape=img_shape, padding="same"))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D())
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(MaxPooling2D())

        model.add(Dense(1, activation='sigmoid'))

        if '-s' in sys.argv:
            model.summary()

        img = Input(shape=img_shape)
        validity = model(img)
        
        r = Model(img, validity)
        return r
   
    
    def train(self, epochs, batch_size=128, save_interval=50):
        X_train=load_polyp_data(self.img_shape)
        half_batch=batch_size
        

        for epoch in tqdm(range(epochs)):

            #  Train Discriminator
            self.trainable(self.discriminator, True)

            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            
            noise = np.random.normal(0, 1, (half_batch,self.latent_dim_int))
            gen_imgs = self.generator.predict(noise)
    
           
           
            # Train the discriminator (real classified as ones and generated as zeros) 
            soft= True if '-soft' in sys.argv else False
            if soft:
                d_loss_real = self.discriminator.train_on_batch(imgs, 0.5*np.random.random_sample((half_batch,1))+0.8)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, 0.3*np.random.random_sample((half_batch,1)))
            else:
                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Train Generator
            self.trainable(self.discriminator, False)
            for _ in range(1):
                noise = np.random.normal(0, 1, (half_batch*2, self.latent_dim_int))
                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, np.ones((half_batch*2,1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch,imgs)
        print("saving")
        self.discriminator.save("new_discriminator.h5")
        self.generator.save("new_generator.h5")
        self.combined.save("new_combined.h5")
        self.discriminator.save_weights("discriminator_weights.h5")
        self.generator.save_weights("generator_weights.h5")
        self.combined.save_weights("combined_weights.h5")


    def save_imgs(self, epoch, sample=None):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim_int))
        gen_imgs = self.generator.predict(noise)
        
        # Rescale images 0 - 1
        gen_imgs = np.clip(0.5 * gen_imgs + 0.5,0,1)
        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        cnt1 = 0
        
        gen2_imgs =self.tester.predict(noise)
        gen2_imgs = np.clip(0.5 * gen2_imgs + 0.5,0,1)
        if sample is None:
            gen3_imgs=gen_imgs
        else:
            gen3_imgs=np.clip(sample,0,1)
        try: 
            for i in range(r):
                for j in range(c):
                    if i == 0:
                        axs[i,j].imshow(gen2_imgs[cnt, :,:,:])
                        axs[i,j].axis('off')
                        cnt += 1
                    elif i==1:
                        axs[i,j].imshow(gen3_imgs[cnt1, :,:,:])
                        axs[i,j].axis('off')
                        cnt1 += 1
                    else:
                        axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                        axs[i,j].axis('off')
                        cnt += 1
            fig.savefig("images/DCGAN_%d.png" % epoch)
            plt.close()
        except Exception as e: 
            print(e)



    def trainable(self,model, status):
        for layer in model.layers:
            layer.trainable = status
       



if __name__ == '__main__':
    if len(sys.argv)==1:
        print("")
        print("")
        print("")
        print("")
        print("USAGE")
        print("First argument is num epochs, rest is:\n")
        print("-l    loads the npy data (saves time)")
        print("-w    loads the weights from last run")
        print("-soft sets soft bounderies")
        print("-img  shows random image (not run)")
        sys.exit()
        
    if "-img" in sys.argv:
        data=np.load("train_data.npy")
        #data = (data.astype(np.float32) - 127.5) / 127.5
        a = 0.5 * data + 0.5
        
       
        plt.imshow(a[np.random.randint(1),:,:,:])
        plt.show()
        sys.exit()
        
    obj = GAN()
    a=sys.argv[1]
    if int(a):
        a=int(a)
    else:
        a=50
    obj.train(epochs=a, batch_size=8, save_interval=5)
