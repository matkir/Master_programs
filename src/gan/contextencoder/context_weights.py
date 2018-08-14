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

class Weight_model():
    def __init__(self,img_rows,img_cols,mask_width,mask_height,corner=False):
        self.img_rows = img_rows 
        self.img_cols = img_cols 
        self.mask_width = mask_width
        self.mask_height = mask_height 
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)
        self.corner=corner
   
    def build_generator_img_size(self):
        """
        def residual_block(x,channels_out, strides=(1, 1),):
            #NOTE, not in use
            shortcut=x
            x1=x(Conv2D(channels_out, kernel_size=3, strides=strides, padding="same")))
            x2=x1(LeakyReLU(alpha=0.2))
            x3=x2(BatchNormalization(momentum=0.8))
            x4 = layers.add([shortcut, x3])
            return x4
        """            
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
            scale_x=int(np.ceil(self.mask_height/(model.outputs[0].get_shape()[1:3].as_list()[1])))
            model.add(UpSampling2D((scale_y,scale_x)))
            height=model.outputs[0].get_shape()[1:3].as_list()[0]
            width=model.outputs[0].get_shape()[1:3].as_list()[1]
            target_height=(height-self.mask_height)/2
            target_width=(width-self.mask_width)/2
            
            #MIGHT BE WRONG
            model.add(Cropping2D((
                (int(np.ceil(target_height)),int(np.floor(target_height))),
                (int(np.ceil(target_width)),int(np.floor(target_width))))))

        model.add(Conv2D(32, kernel_size=5, strides=1, padding="same",activation='relu'))
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same",activation='relu'))
        model.add(Conv2D(16, kernel_size=3, strides=1, padding="same",activation='relu'))
        model.add(Conv2D(8, kernel_size=3, strides=1, padding="same",activation='relu'))
        model.add(Conv2D(self.channels, kernel_size=3, strides=1, padding="same",activation='tanh'))
        model.summary()

        masked_img = Input(shape=self.img_shape)
        gen_missing = model(masked_img)

        return Model(masked_img, gen_missing)


    def build_generator(self):


        model = Sequential()

        # Encoder
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dropout(0.5))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=3, strides=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(512, kernel_size=1, strides=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        # Decoder
        model.add(UpSampling2D(size=(3,3)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        
        model.add(UpSampling2D(size=(3,3)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
       
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Dropout(0.5))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation('tanh'))

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

    def sampling(self,args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
    
        # Returns:
            z (tensor): sampled latent vector
        """
    
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


if __name__=='__main__':
    print("Usage: Same weights used by all programs")
    img_rows = 576
    img_cols = 720
    mask_width = 208
    mask_height = 280
    model=Weight_model(img_rows, img_cols, mask_width,mask_height)
    a=model.build_generator_img_size()
    print("")
