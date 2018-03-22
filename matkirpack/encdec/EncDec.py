"""
To generate HTML documentation for this module issue the
command:

    pydoc -w foo


This is the encoder and decoder used in the master thesis. This is a package, and in the future, it will contain multiple different 
encoders and decoders.

"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model


def build_encoder(img_shape):
    """
    Encoder
    :param img_shape: The shape of the input image as a tuple containing shape
    :return: returns "input_img" wich is the input layer, "encoded" which is the last layer, and the model.
    """
    input_img = Input(shape=(img_shape)) 
    x = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(540, activation='relu')(x)
    Encoder=Model(input_img,encoded,name='encoder')
    return input_img,encoded,Encoder
def build_decoder(encoded):
    """
    Decoder
    :param encoded: the layer that is the input of the decoder (typically the encoded layer from the encoder)
    :return: returns "input_code" wich is the input layer, "decoded" which is the last layer, and the model.
    """    
    input_code=Input(shape=encoded.get_shape().as_list()[1:])
    x = Reshape((720//48,576//48,3))(input_code)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    Decoder=Model(input_code,decoded,name='decoder')
    return input_code,decoded,Decoder
def build_AE(e,d):
    """
    Autoencoder
    :param e: Model of encoder
    :param d: Model of decoder
    :return: Return the compiled model.
    """        
    AE=Sequential()
    AE.add(e)
    AE.add(d)
    AE.compile(optimizer='adam', loss='mse')
    return AE