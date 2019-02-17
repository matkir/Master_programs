from __future__ import print_function, division
if __name__=='__main__':
    from cc_weights import Weight_model
else:
    from . import Weight_model
from keras.models import load_model
import keras.backend as K
import plotload
import sys
from selector import Selector
#from masker import mask_from_template,mask_randomly_square,mask_green_corner,combine_imgs_with_mask
import masker as ms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cutter
import masker
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
        if not mask:
            dummy=plotload.load_polyp_batch(self.img_shape,20,data_type='med/stool-inclusions',crop=False)
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
        self.generator=load_model(f"models/CCgan-gen-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
        self.discriminator=load_model(f"models/CCgan-dic-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
        self.combined=load_model(f"models/CCgan-com-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
    def load_model_weights(self):
        if self.combined==None:
            print("Error: no model in object")
        else:
            try:
                self.combined.load_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w-com.h5")
                self.discriminator.load_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w-dis.h5")
                self.generator.load_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w-gen.h5")
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
        half_batch = batch_size
        for epoch in tqdm(range(epochs)):
            X_train = plotload.load_polyp_batch(self.img_shape, batch_size, data_type='med/none',crop=False)
            

            if corner:
                masked_imgs, missing, mask = ms.mask_green_corner(X_train)
                m=np.zeros(shape=X_train.shape)
                for i in range(X_train.shape[0]):
                    m[i,mask[0]:mask[1],mask[2]:mask[3]]=missing[i]
                missing=m
            else:
                masked_imgs, missing, mask = ms.mask_from_template(X_train)            
            
            if soft:
                valid = 0.2*np.random.random_sample((half_batch,1))+0.9
                fake = 0.1*np.random.random_sample((half_batch,1))
            else:
                valid = np.ones((half_batch, 1))
                fake = np.zeros((half_batch, 1))
                
            # ---------------------
            #  Train Generator
            # ---------------------

                

            
            valid = np.ones((batch_size, 1))
        
            # Train the generator
            t(self.discriminator,False)
            g_loss = self.combined.train_on_batch(masked_imgs, [X_train, valid])
            t(self.discriminator,True)
            
            # ---------------------
            #  Train discriminator
            # ---------------------
           

            gen_fake = self.generator.predict(masked_imgs)
            gen_fake = ms.combine_imgs_with_mask(gen_fake, X_train, mask)
            
            if epoch%120==0 and epoch!=0:
                #small shakeup to get out of local minimas
                fake, valid = valid , fake

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(X_train, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        






            # Plot the progress
            print ("[D: %f  G: %f, mse: %f]" % (d_loss[0], g_loss[0], g_loss[1]))
            if g_loss[1]<self.threshold:
                self.threshold=g_loss[1]
                self.generator.save(f"models/CCgan-gen-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.discriminator.save(f"models/CCgan-dic-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.combined.save(f"models/CCgan-com-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}.h5")   
                self.combined.save_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w-com.h5") 
                self.discriminator.save_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w-dis.h5") 
                self.generator.save_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w-gen.h5") 
        if g_loss[1]<self.threshold:
            self.threshold=g_loss[1]
            self.generator.save(f"models/CCgan-gen-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}_fin.h5")   
            self.discriminator.save(f"models/CCgan-dic-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}_fin.h5")   
            self.combined.save(f"models/CCgan-com-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}_fin.h5")   
            self.combined.save_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w-com_fin.h5") 
            self.discriminator.save_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w-dis_fin.h5") 
            self.generator.save_weights(f"models/CCgan-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if corner else 'n'}-w-gen_fin.h5")        
    def build_wrapper(self):
        """
        Returns a func that works as a complete preprocsess tool
        """
        if mask==1:
            def ret(input_img,mask=None):
                """
                Without a corner, a mask must be added
                """
                if not cutter.is_green(input_img):
                    return input_img
                if mask is None:
                    mask=plotload.load_single_template(input_img.shape,dest='med/green')
                img=input_img.copy()
                if len(img.shape)==3:
                    img=np.expand_dims(img, 0) 
                prediced=np.squeeze(self.generator.predict(img),0)
                img=masker.combine_imgs_with_mask(prediced, img, mask)
                return np.expand_dims(img,0)
        else:
            def ret(input_img):
                if not cutter.is_green(input_img):
                    return input_img
                img=input_img.copy()
                if len(img.shape)==3:
                    img=np.expand_dims(img, 0) 
                y1,y2,x1,x2=self.dims
                img, _, _ = ms.mask_green_corner(img)
                prediced=np.squeeze(self.generator.predict(img),0)
                img=np.squeeze(img,0)
                img[y1:y2,x1:x2]=prediced[y1:y2,x1:x2]
                return np.expand_dims(img,0)
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
    def sort_folder(self,w,path=None):
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
    cc = CCgan(256,256)
    #cc.build_model()
    #cc.train_model()
    cc.load_model()
    #cc.load_model_weights()
    w=cc.build_wrapper()
    root='/home/mathias/Documents/kvasir-dataset-v2/med/'
    cc.sort_folder(w,path=root)
    cc.sort_folder(w,path='/media/mathias/A_New_Hope/medico_test/')
    




