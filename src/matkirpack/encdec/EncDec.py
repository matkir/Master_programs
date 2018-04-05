"""
To generate HTML documentation for this module issue the
command:

    pydoc -w foo


This is the encoder and decoder used in the master thesis. This is a package, and in the future, it will contain multiple different 
encoders and decoders.

"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D,AveragePooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model


def build_encoder(img_shape):
    """
    Encoder
    
    This encoder is suposed to be used for the GAN, but is trained with an AE (in the future)
    
    
    Important, image is in the range -1->1 so tanh is better.
    https://github.com/soumith/ganhacks talks about not using 0->1
    
    :param img_shape: The shape of the input image as a tuple containing shape
    :return: returns "input_img" wich is the input layer, "encoded" which is the last layer, and the model.
    """
    input_img = Input(shape=(img_shape)) 
    x = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = AveragePooling2D((3, 3), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(540, activation='tanh')(x)
    Encoder=Model(input_img,encoded,name='encoder')
    return input_img,encoded,Encoder
def build_decoder(encoded):
    """
    Decoder
    :param encoded: the layer that is the input of the decoder (typically the encoded layer from the encoder)
    :return: returns "input_code" wich is the input layer, "decoded" which is the last layer, and the model.
    """
    if type(encoded)==type((1,1)):        
        s=encoded
    else:
        s=encoded.get_shape().as_list()[1:]
    
    input_code=Input(shape=s)
    x = Dense(540)(input_code)
    x = Reshape((720//48,576//48,3))(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    #x = BatchNormalization(momentum=0.8)(x)
    #x = UpSampling2D((2, 2))(x)
    #x = Dropout(0.25)(x)
    #x = Conv2D(8, (3, 3), padding='same')(x)
    #x = LeakyReLU()(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    Decoder=Model(input_code,decoded,name='decoder')
    return input_code,decoded,Decoder
def build_AE(e,d,opt='adam',l='mse'):
    """
    Autoencoder
    :param e: Model of encoder
    :param d: Model of decoder
    :param opt: either optimizer or string
    :param l: loss function
    :return: Return the compiled model.
    """        
    AE=Sequential()
    AE.add(e)
    AE.add(d)
    AE.compile(optimizer=opt, loss=l)
    return AE

def build_GAN(e,d,opt='adam',l='binary_crossentropy'):
    GAN=Sequential()
    GAN.add(e)
    GAN.add(d)
    GAN.compile(optimizer=opt, loss=l)
    return GAN
    
def build_discriminator(shape):
    """
    discriminator so tell if an image is real or fake.
    :param shape: tuple containing the sape if the input, usually the shape of the image
    :return: model outputting true/false
    """
    input_img = Input(shape=(shape)) 
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    o = Dense(1,activation='sigmoid')(x)
    Discriminator=Model(input_img,o,name='discriminator')
    return input_img,o,Discriminator    
