from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, Permute
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Cropping2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import numpy as np
import plotload
import masker
import cutter
class Weight_model():
    def __init__(self,img_rows,img_cols,corner=True):
        self.img_cols = img_cols 
        self.img_rows = img_rows 
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        if corner:
            dummy=plotload.load_one_img(self.img_shape, dest='green',extra_dim=True)
            self.dims =cutter.find_square_coords(dummy)
            self.mask_width = self.dims[3]-self.dims[2]
            self.mask_height = self.dims[1]-self.dims[0]
        else:        
            self.mask_width = 62#208
            self.mask_height = 51#280
        self.corner=corner
        
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)
   
    def build_generator_img_size(self):
         
        model = Sequential()

        # Encoder
        model.add(Conv2D(16, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(16, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(32, kernel_size=3, strides=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))   
        
        model.add(Conv2D(64, kernel_size=3, strides=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
       
        # Dropout
        model.add(Dropout(0.5))

        # Decoder
        
      
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation('relu'))        
        

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(Activation('tanh'))
        if self.corner:
            scale_y=int(np.ceil(self.mask_height/(model.outputs[0].get_shape()[1:3].as_list()[0])))
            scale_x=int(np.ceil(self.mask_width/(model.outputs[0].get_shape()[1:3].as_list()[1])))
            model.add(UpSampling2D((scale_y,scale_x)))
            height=model.outputs[0].get_shape()[1:3].as_list()[0]
            width=model.outputs[0].get_shape()[1:3].as_list()[1]
            target_height=(height-self.mask_height)/2
            target_width=(width-self.mask_width)/2
            
            #MIGHT BE WRONG, but probably not, lol, nevermind!
            model.add(Cropping2D((
                (int(np.ceil(target_height)),int(np.floor(target_height))),
                (int(np.ceil(target_width)),int(np.floor(target_width)))
            )))
        model.summary()
        model.add(Conv2D(32, kernel_size=5, strides=1, padding="same",activation='relu'))
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same",activation='relu'))
        model.add(Conv2D(16, kernel_size=3, strides=1, padding="same",activation='relu'))
        model.add(Conv2D(8, kernel_size=3, strides=1, padding="same",activation='relu'))
        model.add(Conv2D(self.channels, kernel_size=3, strides=1, padding="same",activation='tanh'))
        model.summary()

        masked_img = Input(shape=self.img_shape)
        gen_missing = model(masked_img)

        return Model(masked_img, gen_missing)




    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.missing_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
       
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.missing_shape)
        validity = model(img)

        return Model(img, validity)


    def build_CE(self):
        optimizer = Adam(0.0002, 0.5)
        
        #Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator_img_size()
        #self.generator.compile(loss=['binary_crossentropy'],
        #    optimizer=optimizer)

        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator) takes
        # masked_img as input => generates missing image => determines validity
        self.combined = Model(masked_img , [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.7, 0.3],
            optimizer=optimizer)        
        return self.discriminator,self.generator,self.combined

if __name__=='__main__':
    print("Usage: Same weights used by all programs")
    img_rows = 576
    img_cols = 720
    mask_width = 208
    mask_height = 280
    model=Weight_model(img_rows, img_cols, mask_width,mask_height)
    a=model.build_generator_img_size()
    print("")
