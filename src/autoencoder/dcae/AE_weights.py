from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import MaxPooling2D, AveragePooling2D, Concatenate, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

class Weight_model():
    def __init__(self,img_shape):
        self.img_shape = img_shape
        
        
    def build_AE(self,opt='adam',l='mse'):
     
            input_img = Input(shape=(self.img_shape)) 
            x = Conv2D(16, (2, 2), activation='tanh', padding='same')(input_img)
            x = AveragePooling2D((2, 2), padding='same')(x)
            x = Conv2D(64, (2, 2), padding='same')(x)
            x = LeakyReLU()(x)
            x = AveragePooling2D((2, 2), padding='same')(x)
            x = Conv2D(128, (2, 2), padding='same')(x)
            x = LeakyReLU()(x)
            x = AveragePooling2D((2, 2), padding='same')(x)
            x = Conv2D(128, (2, 2), padding='same')(x)
            x = LeakyReLU()(x)
            encoded = AveragePooling2D((2, 2), padding='same')(x)
            
            x = Conv2D(128, (2, 2), padding='same')(encoded)
            x = LeakyReLU()(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(96, (2, 2), padding='same')(x)
            x = LeakyReLU()(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(64, (2, 2), padding='same')(x)
            x = LeakyReLU()(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(16, (3, 3), padding='same')(x)
            x = LeakyReLU()(x)
            x = UpSampling2D((2, 2))(x)
            decoded = Conv2D(3, (2, 2), activation='tanh', padding='same')(x)  
            AE=Model(input_img, decoded, name="AE")            
            AE.summary()
            AE.compile(optimizer=opt, loss=l)
            return AE
    
    


if __name__=='__main__':
    print("Usage: Same weights used by all programs")
    img_rows = 576
    img_cols = 720
    model=Weight_model(img_rows, img_cols)
    model.build_AE()
    print("")
