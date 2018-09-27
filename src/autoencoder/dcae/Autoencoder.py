from keras.models import load_model
if __name__=='__main__':
    from AE_weights import Weight_model
else:
    from . import Weight_model
import keras.backend as K
import tensorflow as tf
import plotload
import cutter
import numpy as np
import matplotlib.pyplot as plt
class Autoencoder():
    def __init__(self,img_cols,img_rows):
        """
        Initializes the autoencoder. 
        """
        self.set_training_info()
        globals().update(self.info)  
        self.threshold=threshold
        self.img_cols = img_cols#256 # Original is ~576
        self.img_rows = img_rows#256 # Original is ~720 
        self.channels = 3   # RGB 
        self.img_shape=(self.img_cols,self.img_rows,self.channels)
        dummy=plotload.load_one_img(self.img_shape, dest='med/green',extra_dim=True)
        self.dims =cutter.find_square_coords(dummy)          
        self.model=None
        self.pretrained=False
        
    def load_model(self,adress=None):
        """
        loads a model to the object instead of creating one. 
        :param adress: string of adress to the file of type h5.
        """
        if self.model!=None:
            print("Warning: overriding a loaded model")
        if adress is None:
            self.model=load_model(f"models/AE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if mask==0 else 'n'}.h5")   
        else:
            self.model=load_model(adress)
   
            
    def load_model_weights(self,adress=None):
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
                if adress is None:
                    self.model.load_weights(f"models/AE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if mask==0 else 'n'}-w.h5")                       
                else:
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
        if self.model!=None:
            print("Warning: overriding a loaded model")
        wm=Weight_model(self.img_shape)
        
        #def C_MSE(y_true, y_pred):
        #    y1,y2,x1,x2=cutter.find_square_coords(np.zeros(shape=(256,256,3)))
        #    return K.mean(K.square(tf.slice(y_pred,[0,y1,x1,0],[0,y2,x2,0]) - tf.slice(y_true,[0,y1,x1,0],[0,y2,x2,0])), axis=-1)  
        
        self.model=wm.build_AE()
        
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
        """
        Trainer: Trains the loaded autoencoder model
        :param epochs: number of epochs run
        :param batch_size: how many imgs in each batch
        :param save_interval: how many epochs between each save
        """
        if self.info==None:
            print("Warning no info found, prompting for info")
            self.set_training_info()
        globals().update(self.info)
        if self.model==None:
            print("Error: no model loaded")
            return
        if self.pretrained==True:
            print("Warning: model has pretrained weights")
        from tqdm import tqdm
        y1,y2,x1,x2=cutter.find_square_coords(plotload.load_polyp_batch(self.img_shape, batch_size))
        
        for epoch in tqdm(range(epochs)):
            X_train=plotload.load_polyp_batch(self.img_shape, batch_size,data_type='med/none')
            if mask==0:
                Y_train,X_train=cutter.add_green_suare(X_train)
            else:
                print("Not yet implimented")
            cur_loss=self.model.train_on_batch(X_train, Y_train)
            
            if epoch%10==0:
                self.save_img(epoch)
            
            if cur_loss<self.threshold:
                print(cur_loss)
                self.threshold=cur_loss
                self.model.save(f"models/AE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if mask==0 else 'n'}-fin.h5")   
                self.model.save_weights(f"models/AE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if mask==0 else 'n'}-w-fin.h5")   
        self.model.save(f"models/AE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if mask==0 else 'n'}-fin.h5")   
        #self.model.save_weights(f"models/AE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if mask==0 else 'n'}-w-fin.h5")   
        
            
    def build_wrapper(self):
        """
        Returns a func that works as a complete preprocsess tool
        input shape is [1,x,x,3]
        """
        if mask==0:
            if self.model==None:
                print("no model loaded")
                assert False
            def ret(input_img):
                if not cutter.is_green(input_img):
                    return input_img
                img=input_img.copy()
                y1,y2,x1,x2=cutter.find_square_coords(input_img)
                prediced=np.squeeze(self.model.predict(img),0)
                img=np.squeeze(img,0)
                img[y1:y2,x1:x2]=prediced[y1:y2,x1:x2]
                return np.expand_dims(img,0)
        else:
            print("Not yet implimented")

        return ret
    

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



    def save_img(self,epoch):
        test=plotload.load_one_img((256,256), dest='/home/mathias/Documents/kvasir-dataset-v2/med/stool-plenty/1.jpg', 
                                   extra_dim=True) 
        test2=A.model.predict(test)
        
        plt.subplot(121)
        plt.imshow(test[0]*0.5+0.5)
        plt.subplot(122)
        plt.imshow(test2[0]*0.5+0.5)
        plt.savefig(f"epoc_{epoch}.jpg")

        
if __name__=='__main__':
    A=Autoencoder(256,256)
    A.build_model()
    #A.load_model()
    A.train_model()    
    #A.load_model_weights()
    root='/home/mathias/Documents/kvasir-dataset-v2/med/'    
    w=A.build_wrapper()
    A.sort_folder(w,path=root)
