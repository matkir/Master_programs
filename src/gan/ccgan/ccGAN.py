from __future__ import print_function, division
if __name__=='__main__':
    from cc_weights import Weight_model
else:
    from . import Weight_model
import keras.backend as K
import plotload
import sys
from selector import Selector
#from masker import mask_from_template,mask_randomly_square,mask_green_corner,combine_imgs_with_mask
import masker as ms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
class CCgan():
    def __init__(self,img_cols,img_rows):
        """
        Initializes the autoencoder. 
        """
        self.set_training_info()
        globals().update(self.info)  
        self.threshold=threshold
        self.img_cols = img_cols # Original is ~576
        self.img_rows = img_rows # Original is ~720 
        self.channels = 3   # RGB 
        self.img_shape=(self.img_cols,self.img_rows,self.channels)
        self.combined=None
        self.discriminator=None
        self.generator=None
        self.pretrained=False
    def load_model(self):
        """
        loads a model to the object instead of creating one. 
        :param adress: string of adress to the file of type h5.
        """
        if self.combined!=None:
            print("Warning: overriding a loaded model")
        self.generator=load_model(f"models/CCgan-gen-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
        self.discriminator=load_model(f"models/CCgan-dic-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
        self.combined=load_model(f"models/CCgan-com-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
    def load_model_weights(self):
        if self.combined==None:
            print("Error: no model in object")
        else:
            try:
                self.combined.load_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w.h5")
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
        self.discriminator,self.generator,self.combined=wm.build_model() 
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

    def train_model(self):
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
            X_train=plotload.load_polyp_batch(self.img_shape, batch_size, data_type='green')

            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            if not corner:
                masked_imgs, missing, mask = ms.mask_from_template(imgs)
            else:
                masked_imgs, missing, mask = ms.mask_green_corner(imgs)

            gen_fake = self.generator.predict(missing)
            gen_fake = ms.combine_imgs_with_mask(gen_fake, imgs, mask)


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
            d_loss_fake = self.discriminator.train_on_batch(gen_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            masked_imgs, missing_parts, _ = ms.mask_from_template(imgs)

            valid = np.ones((batch_size, 1))

            # Train the generator
            g_loss = self.combined.train_on_batch(masked_imgs, [imgs, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            if g_loss[1]<self.threshold:
                self.threshold=g_loss[1]
                self.generator.save(f"models/CCgan-gen-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.discriminator.save(f"models/CCgan-dic-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.combined.save(f"models/CCgan-com-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.combined.save_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w.h5") 
    def build_wrapper(self):
        """
        Returns a func that works as a complete preprocsess tool
        """
        if mask==1:
            if self.generator==None:
                print("no model loaded")
                assert False
            def ret(input_img):
                if not cutter.is_green(input_img):
                    return input_img
                img=input_img.copy()
                if len(img.shape)==3:
                    img=np.expand_dims(img, 0) 
                y1,y2,x1,x2=ContextEncoder.dims
                prediced=np.squeeze(self.generator.predict(img),0)
                img=np.squeeze(img,0)
                img[y1:y2,x1:x2]=prediced[y1:y2,x1:x2]
                return np.expand_dims(img,0)
        else:
            print("Not yet implimented")

        return ret                            
            

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

                



if __name__ == '__main__':
    cc = CCgan(256,256)
    cc.build_model()
    cc.train_model()
    cc.build_wrapper()
    




