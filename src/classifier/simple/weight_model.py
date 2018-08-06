from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import MaxPooling2D, AveragePooling2D, Concatenate, GaussianNoise
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

class Weight_model():
    def __init__(self,img_rows,img_cols):
        self.img_rows = img_rows 
        self.img_cols = img_cols 
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.gf=64
        self.df=64
        
    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=(3,3), strides=3, padding='same', input_shape=self.img_shape,activation='relu'))
        model.add(Conv2D(32, kernel_size=(3,3), strides=3, padding='same',activation='relu'))
        model.add(Conv2D(64, kernel_size=(4,4), strides=2, padding='same',activation='relu'))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same',activation='relu'))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding='same',activation='relu'))
        model.add(Conv2D(512, kernel_size=2, strides=1, padding='same',activation='relu'))
        model.add(Conv2D(512, kernel_size=2, strides=2, padding='same',activation='relu'))
        
        model.summary()
    
        img = Input(shape=self.img_shape)
        features = model(img)
    
        label = Flatten()(features)
        label = Dense(100, activation="relu")(label)
        validity = Dense(1, activation="sigmoid")(label)
    
        return Model(img,validity)

    


if __name__=='__main__':
    print("Usage: Same weights used by all programs")
    img_rows = 576
    img_cols = 720
    mask_width = 208
    mask_height = 280
    model=Weight_model(img_rows, img_cols)
    a=model.build_discriminator()
    print("done")
    print("done")
    print("done")
    
