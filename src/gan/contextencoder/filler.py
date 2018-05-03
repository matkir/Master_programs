from __future__ import print_function, division

from keras.models import load_model
import keras.backend as K
import sys
import numpy as np

class ContextEncoder():
    def __init__(self):
        self.img_rows = 576#8*64//2#32
        self.img_cols = 720#8*64//2#32
        self.mask_height = 208#300 #self.img_cols//4#8*16//2#8
        self.mask_width = 280#350 #self.img_rows//4 #8*16//2#8
        self.channels = 3
        self.num_classes = 2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_width, self.mask_height, self.channels)


        self.generator = load_model("saved_model/generator.h5")
        self.generator.load_weights("saved_model/generator_weigths.h5")
        #self.generator.compile(loss=['binary_crossentropy'],optimizer=optimizer)

        
        self.fill(None,self.img_shape) 
        
    def fill(self,img_adr,img_shape):
        import plotload as pl
        img,img_path=pl.load_one_img(img_shape, dest=img_adr)
        
        masked_img, _, (y1, y2, x1, x2) = self.mask_select(img,img_path)
        guess=np.squeeze(self.generator.predict(np.expand_dims(masked_img,axis=0)),axis=0) 
        
        
        masked_img[y1:y2, x1:x2,:] = guess
        import scipy.misc
        scipy.misc.toimage(masked_img, cmin=-1, cmax=1).save('outfile.png')        


    def mask_select(self,img,img_adr):
        import selector
        x1,x2,y1,y2 = selector.get_coord(adr=img_adr,scale=1)
        masked_img = np.empty_like(img)
        missing_parts = np.ndarray(shape=(x2-x1,y2-y1))
        masked_img = img.copy()
        missing_parts =masked_img[y1:y2, x1:x2,:].copy()
        masked_img[y1:y2, x1:x2,:] = 0
        if False:
            import matplotlib.pyplot as plt
            plt.imshow((0.5*masked_img+0.5))
            plt.show() 
            plt.imshow((0.5*missing_parts+0.5))
            plt.show()        
        return masked_img, missing_parts, (y1, y2, x1, x2)


if __name__ == '__main__':
    context_encoder = ContextEncoder()
