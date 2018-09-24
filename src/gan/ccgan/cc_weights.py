"""
https://github.com/eriklindernoren/Keras-GAN/blob/master/ccgan/ccgan.py
"""
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import MaxPooling2D, AveragePooling2D, Concatenate, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

class Weight_model():
    def __init__(self,img_rows,img_cols):
        self.img_rows = img_rows 
        self.img_cols = img_cols 
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.gf=16
        self.df=16
        
        
    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            return u
        
        img = Input(shape=self.img_shape)
        #adding gausian noise, might not be the right way to do it?
        #noise = GaussianNoise(0.01)(img)
        
        
        # Downsampling
        d1 = conv2d(img, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*16)
    
        # Upsampling
        u1 = deconv2d(d4, self.gf*4,dropout_rate=0.25)
        u2 = deconv2d(u1, self.gf*2,dropout_rate=0.25)
        u3 = deconv2d(u2, self.gf*2)
        u4 = deconv2d(u3, self.gf)
    
        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, output_img)

    
    def build_discriminator(self):

        model = Sequential()
        model.add(Conv2D(16, kernel_size=4, strides=2, padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.8))
        model.add(Conv2D(32, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
        model.add(Conv2D(64, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())     
        model.add(Conv2D(64, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())         
        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
    
        model.summary()
    
        img = Input(shape=self.img_shape)
        features = model(img)
    
        label = Flatten()(features)
        validity = Dense(1, activation="sigmoid")(label)
    
        return Model(img,validity)

    def build_model(self):
        optimizer_generator = Adam()
        optimizer_discriminator = SGD()
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
    
        self.discriminator.compile(loss='binary_crossentropy',
                                optimizer=optimizer_discriminator,
                                metrics=['accuracy'])
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=optimizer_generator)
    
        masked_img = Input(shape=self.img_shape)
        gen_img = self.generator(masked_img)
    
        valid= self.discriminator(gen_img)
    
        self.combined = Model(masked_img , [gen_img, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                                  loss_weights=[0.10, 0.90],
                                  optimizer=optimizer_generator)        
        return self.discriminator,self.generator,self.combined

if __name__=='__main__':
    print("Usage: Same weights used by all programs")
    img_rows = 576
    img_cols = 720
    mask_width = 208
    mask_height = 280
    model=Weight_model(img_rows, img_cols, mask_width,mask_height)
    a=model.build_generator()
    a=model.build_discriminator()
