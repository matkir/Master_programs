import scipy.misc
import plotload as pl
import numpy as np
from selector import gui
import os
from masker import mask_from_template
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

def main(adr='2.jpg',scale=1,img_cols=576,img_rows=720):
    channels = 3
    img_shape = (img_cols, img_rows, channels)
    generator = load_model("saved_model/generator.h5")
    generator.load_weights("saved_model/generator_weigths.h5")
    
  
    img,img_path=pl.load_one_img(img_shape, dest=adr)
    scipy.misc.toimage(img, cmin=-1, cmax=1).save('tmp.png')            
    global going 
    going=True
    for i in range(3):
        mask=gui(adr='tmp.png')
        np.save('/tmp/tmp_mask.npy', mask.T)        
        
        masked_imgs, missing_parts, m = mask_from_template(imgs,'../../../tmp')   
        gen_fake = self.generator.predict(missing_parts)
        gen_fake = combine_imgs_with_mask(gen_fake, imgs, m)
        scipy.misc.toimage(gen_fake, cmin=-1, cmax=1).save('tmp.png')            
        
main()
os.remove('tmp.png')
os.remove('/tmp/tmp_mask.npy')
