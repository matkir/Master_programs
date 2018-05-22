import scipy.misc
import plotload as pl
import numpy as np
from selector import gui
import os
from masker import mask_from_template,combine_imgs_with_mask
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

def main(adr='2.jpg',scale=1,img_cols=576,img_rows=720):
    channels = 3
    img_shape = (img_cols, img_rows, channels)
    generator = load_model("saved_model/generator.h5")
    generator.load_weights("saved_model/generator_weigths.h5")
    
  
    img,img_path=pl.load_one_img(img_shape,'1.jpg')
    scipy.misc.toimage(img, cmin=-1, cmax=1).save('working_img.png')            
    global going 
    going=True
    for i in range(3):
        mask=gui(adr='working_img.png').T
        #import matplotlib.pyplot as plt
        #plt.imshow(0.5*mask+0.5)
        #plt.show()
        np.save('/tmp/tmp_mask.npy', mask)        
        img,img_path=pl.load_one_img(img_shape,"working_img.png")
        masked_imgs, missing_parts, m = mask_from_template(img,'/tmp',rot=False,fliplr=False,flipud=False)   
        gen_fake = generator.predict(missing_parts)
        gen_fake = combine_imgs_with_mask(gen_fake, img, m)
        gen_fake = np.squeeze(gen_fake,axis=0)

        scipy.misc.toimage(gen_fake, cmin=-1, cmax=1).save('working_img.png')            
        
main()
os.remove('tmp.png')
os.remove('/tmp/tmp_mask.npy')
