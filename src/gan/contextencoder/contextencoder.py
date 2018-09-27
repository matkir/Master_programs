from __future__ import print_function, division
from tqdm import tqdm
from keras.models import load_model

import keras.backend as K
import sys
if __name__=='__main__':
    from context_weights import Weight_model 
else:
    from . import Weight_model

import masker as ms
import plotload
import sys
import cutter
import numpy as np

class ContextEncoder():
    def __init__(self,img_cols,img_rows):
        """
        Initializes the contextencoder. 
        """
        self.set_training_info()
        globals().update(self.info)  
        self.threshold=threshold
        self.img_cols = img_cols # Original is ~576
        self.img_rows = img_rows # Original is ~720 
        self.channels = 3   # RGB 
        self.img_shape=(self.img_cols,self.img_rows,self.channels)
        dummy=plotload.load_one_img(self.img_shape, dest='med/green',extra_dim=True)
        self.dims =cutter.find_square_coords(dummy)                  
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
        self.generator=load_model(f"models/CE-gen-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
        self.discriminator=load_model(f"models/CE-dic-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
        self.combined=load_model(f"models/CE-com-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
    def load_model_weights(self):
        """
        loads weights to the model. 
        :param model:  the model that is to get weights
        :param adress: string of adress to the file of type h5.
        :returns:      model with weights
        """
        if self.combined==None:
            print("Error: no model in object")
        else:
            try:
                self.combined.load_weights(f"models/CE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w.h5")
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
        wm=Weight_model(self.img_cols,self.img_rows,self.dims)
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
    
    
    
    
    
    
        


    def train_model(self):
        def t(m,bol):
            for layer in m.layers:
                layer.trainable=bol        
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

            for _ in range(1):
                X_train = plotload.load_polyp_batch(self.img_shape, batch_size, data_type='none',crop=False)
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
            for _ in range(1):
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
                t(self.discriminator,False)
                g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])
                t(self.discriminator,True)

            # Plot the progress
            if epoch%15==0:
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            if g_loss[1]<self.threshold:
                self.threshold=g_loss[1]
                self.generator.save(f"models/CE-gen-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.discriminator.save(f"models/CE-dic-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.combined.save(f"models/CE-com-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.combined.save_weights(f"models/CE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w.h5") 
                

 
        
    def sample_images(self, epoch, imgs):
        r, c = 3, 4
        imgs=plotload.load_polyp_batch(self.img_shape, 4, data_type='green', crop=True)
        
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

   
    def build_wrapper(self):
        """
        Returns a func that works as a complete preprocsess tool
        """
        if mask==0:
            if self.generator==None:
                print("no model loaded")
                assert False
            def ret(input_img):
                if not cutter.is_green(input_img):
                    return input_img
                img=input_img.copy()
                if len(img.shape)==3:
                    img=np.expand_dims(img, 0) 
                y1,y2,x1,x2=self.dims
                prediced=np.squeeze(self.generator.predict(img),0)
                img=np.squeeze(img,0)
                img[y1:y2,x1:x2]=prediced#[y1:y2,x1:x2]
                return np.expand_dims(img,0)
        else:
            print("Not yet implimented")

        return ret        
    def sort_folder(self,w):
        import os
        import cv2
        from tqdm import tqdm
        from shutil import copyfile
        import sys
        
        if path is not None:
            dirs_i=[]
            dirs_o=[]
            d=next(os.walk(path))[1]
            for i in d:
                if i =='none' or i=='green' or i=='preprocessed':
                    continue
                dirs_o.append(path+'preprocessed/'+i)
                dirs_i.append(path+i)
            for i in dirs_o:
                if not os.path.exists(i):
                    os.makedirs(i)                    
        else:
            polyps='polyps'
            ulcerative_colitis='ulcerative-colitis'
            dirs=[polyps,ulcerative_colitis]
        
            if not os.path.exists(polyps_prep):
                os.makedirs(polyps_prep)    
            if not os.path.exists(ulcerative_colitis_prep):
                os.makedirs(ulcerative_colitis_prep)    
        
        for i,o in tqdm(zip(dirs_i,dirs_o)):
            for img_name in os.listdir(i):
                path=os.path.join(i,img_name)
                img=plotload.load_one_img((self.img_cols,self.img_rows), dest=path, 
                                     extra_dim=True)
                if cutter.is_green(img):
                    tmp=cv2.imwrite(os.path.join(o,img_name), cv2.cvtColor(127.5*w(img)[0]+127.5,cv2.COLOR_RGB2BGR))
                else:
                    tmp=cv2.imwrite(os.path.join(o,img_name), cv2.cvtColor(127.5*img[0]+127.5,cv2.COLOR_RGB2BGR))
                

if __name__ == '__main__':
    a=ContextEncoder(256,256)
    a.build_model()
    a.train_model()    
    #a.load_model()
    #a.load_model_weights()
    w=a.build_wrapper()
    a.sort_folder(w)
