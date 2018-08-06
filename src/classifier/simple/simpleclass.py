from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import keras.backend as K
import plotload
import sys
from weight_model import Weight_model
from selector import Selector
from masker import mask_from_template,mask_randomly_square,mask_green_corner,combine_imgs_with_mask
import matplotlib.pyplot as plt

import numpy as np


class Classification():
    def __init__(self):
        self.img_rows = 576//2#8*64//2#32
        self.img_cols = 720//2#8*64//2#32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        optimizer = Adam(lr=0.001)
        self.model=Weight_model(self.img_rows, self.img_cols)
        self.discriminator=self.model.build_discriminator()
        self.discriminator.compile(optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        if '-weights' in sys.argv:
            print("loading old weights")
            self.discriminator.load_weights("saved_model/discriminator_weigths.h5")
            
        if "-save" in sys.argv:
            self.generator.save("saved_model/generator.h5")
            self.discriminator.save("saved_model/discriminator.h5")        
            sys.exit()




   

    def train(self):
        from tqdm import tqdm
        X_train={}
        Y_train={}
        X_val={}
        Y_val={}
        X_test={}
        Y_test={}


        if True:
            polyps = plotload.load_polyp_batch(self.img_shape,1000,data_type='polyps',crop=True,rot=True)
            polyps = np.concatenate((polyps[0::4],polyps[1::4],polyps[2::4],polyps[3::4]))
            nonpolyps= plotload.load_polyp_batch(self.img_shape,1000,data_type='normal-cecum',crop=True,rot=True)
            nonpolyps = np.concatenate((nonpolyps[0::4],nonpolyps[1::4],nonpolyps[2::4],nonpolyps[3::4]))
                      
            np.save('polyps.npy', polyps)
            np.save('nonpolyps.npy', nonpolyps)
           
        else:
            polyps=np.load('polyps.npy')
            nonpolyps=np.load('nonpolyps.npy')
        
        
        totlen=len(polyps)//4
        trainlen=int(totlen*0.8)
        vallen=int(totlen*0.1)
        testlen=int(totlen*0.1) 
        
        polyp_set1=polyps[:totlen]
        
        polyp_set2=polyps[totlen:(2*totlen)]
        polyp_set2=polyp_set2[vallen+testlen:]
        polyp_set3=polyps[(2*totlen):(3*totlen)]
        polyp_set3=polyp_set3[vallen+testlen:]
        polyp_set4=polyps[(3*totlen):]
        polyp_set4=polyp_set4[vallen+testlen:]
        
        nonpolyp_set1=nonpolyps[:totlen]
        
        nonpolyp_set2=nonpolyps[totlen:(2*totlen)]
        nonpolyp_set2=nonpolyp_set2[vallen+testlen:]
        nonpolyp_set3=nonpolyps[(2*totlen):(3*totlen)]
        nonpolyp_set3=nonpolyp_set3[vallen+testlen:]
        nonpolyp_set4=nonpolyps[(3*totlen):]
        nonpolyp_set4=nonpolyp_set4[vallen+testlen:]
        
        
        X_test['polyp']=polyp_set1[:testlen]
        X_test['nonpolyp']=nonpolyp_set1[:testlen]
        X_val['polyp']=polyp_set1[testlen:testlen+vallen]
        X_val['nonpolyp']=nonpolyp_set1[testlen:testlen+vallen]
        
        X_train['polyp']=np.concatenate((polyp_set1[:trainlen],polyp_set2,polyp_set3,polyp_set4))
        X_train['nonpolyp']=np.concatenate((nonpolyp_set1[:trainlen],nonpolyp_set2,nonpolyp_set3,nonpolyp_set4))
        
        
        
        Y_train['polyp']    = np.ones(len(X_train['polyp'])).T
        Y_train['nonpolyp'] = np.zeros(len(X_train['polyp'])).T
        Y_val['polyp']     = np.ones(len(X_val['polyp'])).T
        Y_val['nonpolyp']  = np.zeros(len(X_val['polyp'])).T
        Y_test['polyp']     = np.ones(len(X_test['polyp'])).T
        Y_test['nonpolyp']  = np.zeros(len(X_test['polyp'])).T
        
        X_train['combined'] = np.concatenate((X_train['polyp'],X_train['nonpolyp']))
        Y_train['combined'] = np.concatenate((Y_train['polyp'],Y_train['nonpolyp']))
        X_val['combined']   = np.concatenate((X_val['polyp'],X_val['nonpolyp']))
        Y_val['combined']   = np.concatenate((Y_val['polyp'],Y_val['nonpolyp']))
        X_test['combined']  = np.concatenate((X_test['polyp'],X_test['nonpolyp']))
        Y_test['combined']  = np.concatenate((Y_test['polyp'],Y_test['nonpolyp']))

        if len(sys.argv)==2:
            name=sys.argv[1]
            self.discriminator.fit(x=X_train['combined'],
                                   y=Y_train['combined'], 
                                   epochs=10,
                                   validation_data=[X_val['combined'],
                                                    Y_val['combined']],
                                   callbacks=[TensorBoard(log_dir="./logs/"+name,
                                                          histogram_freq=2, batch_size=32)])
        else:
            self.discriminator.fit(x=X_train['combined'],
                                   y=Y_train['combined'], 
                                   epochs=10,
                                   validation_data=[X_val['combined'],
                                                    Y_val['combined']])            

        #print(self.discriminator.evaluate(x=X_test['polyp'], y=Y_test['polyp']))
        #print(self.discriminator.evaluate(x=X_test['nonpolyp'], y=Y_test['nonpolyp']))
        guess=self.discriminator.predict(X_test['combined'])
        print((np.sum(np.rint(guess.T) == Y_test['combined']))/len(guess))
        print(np.rint(guess.T))
        print(Y_test['combined'])
        for i,j in enumerate(guess):
            if np.rint(j) != Y_test['combined'][i]:
                print("guess=",j,"   ans=",Y_test['combined'][i])
                plt.imshow(X_test['combined'][i]*0.5+0.5)
                plt.show()
        
        
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == '__main__':
    clasifier=Classification()
    clasifier.train()
    print("done")


    