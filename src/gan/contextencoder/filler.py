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


        # Build and compile the generator
        self.generator = load_model("saved_model/generator.h5")
        self.generator.load_weights("saved_model/generator_weights.h5")
        self.generator.compile(loss=['binary_crossentropy'],
                               optimizer=optimizer)

        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)
        self.fill(sys.argv[1]) 
        
    def fill(img_adr):
        import plotload as pl
        img=pl.load_one_img(img_shape, dest=img_adr)
        
        masked_img, _, (y1, y2, x1, x2) = self.mask_select(img,img_adr)
        guess = self.generator.predict(masked_img) 
        
        masked_img[x1:x2, y1:y2,:] = guess
        import scipy.misc
        scipy.misc.toimage(masked_img, cmin=-1, cmax=1).save('outfile.png')        


    def mask_select(self,imgs,img_adr):
        import selector
        x1,y1,x2,y2 = selector.get_coord(adr=img_adr,scale=1)
        masked_img = np.empty_like(imgs)
        missing_parts = np.ndarray(shape=(x2-x1,y2-y1))
        masked_img = img.copy()
        missing_parts =masked_img[x1:x2, y1:y2,:]
        masked_img[x1:x2, y1:y2,:] = 0
        return masked_imgs, missing_parts, (y1, y2, x1, x2)


if __name__ == '__main__':
    context_encoder = ContextEncoder()
