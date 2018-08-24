from keras.models import load_model
if __name__=='__main__':
    from AE_weights import Weight_model
else:
    from . import Weight_model
    
import plotload
import cutter
import numpy as np
class Autoencoder():
    def __init__(self,img_cols,img_rows):
        """
        Initializes the autoencoder. 
        """
        self.set_training_info()
        globals().update(self.info)  
        self.threshold=threshold
        self.img_cols = 256 # Original is ~576
        self.img_rows = 256 # Original is ~720 
        self.channels = 3   # RGB 
        self.img_shape=(self.img_cols,self.img_rows,self.channels)
        self.model=None
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
        if self.model!=None:
            print("Warning: overriding a loaded model")
        wm=Weight_model(self.img_shape)
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
        for epoch in tqdm(range(epochs)):
            X_train=plotload.load_polyp_batch(self.img_shape, batch_size)
            if mask==0:
                Y_train,X_train=cutter.add_green_suare(X_train)
            else:
                print("Not yet implimented")
            cur_loss=self.model.train_on_batch(X_train, Y_train)
            
            if cur_loss<self.threshold:
                self.threshold=cur_loss
                self.model.save(f"models/AE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if mask==0 else 'n'}.h5")   
                self.model.save_weights(f"models/AE-{self.img_shape[0]}-{self.img_shape[1]}-{'c' if mask==0 else 'n'}-w.h5")   
        
            
    def build_wrapper(self):
        """
        Returns a func that works as a complete preprocsess tool
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
    
