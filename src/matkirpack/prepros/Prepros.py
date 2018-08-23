import numpy as np
from keras.models import load_model


class Prepros():
    def __init__(self,img_size,algorithm=None):
        self.img_size=img_size
        if algorithm==None:
            print()
            print('Usage: \n The algorithm is a list with the following \n [type,filter] \n type: "AE" "CE" "CCGAN" \n filter: "c"-corner version "n"-template version' )
            assert False
        self.algorithm=algorithm[0]
        self.mask_filter=algorithm[1]
    
        try:
            self.model = load_model(f"models/{self.algorithm}-{self.img_size[0]}-{self.img_size[1]}-{self.mask_filter}.h5")
            self.model.load_weights(f"models/{self.algorithm}-{self.img_size[0]}-{self.img_size[1]}-{self.mask_filter}-w.h5")
        except:
            print()
            print('Model not found, make files named:')
            print(f"{self.algorithm}-{self.img_size[0]}-{self.img_size[1]}-{self.mask_filter}.h5")
            print(f"{self.algorithm}-{self.img_size[0]}-{self.img_size[1]}-{self.mask_filter}-w.h5")
       
    def __call__(img):
        if img.shape[0]!=self.img_size[0]:
            print(f"incomatible img size:\nGot {img.shape[0:2]}\nExpected {self.img_size}")
            assert False
        return self.model.predict(img)
        
Prepros((1,1),algorithm=["AE","c"])