from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os

class TL():
    def __init__(self):
        self.img_rows = 576//8
        self.img_cols = 720//8        
        self.channels = 3
        self.img_shape = (self.img_cols, self.img_rows, self.channels)
        model,l_out=self.make_model()
        self.VGG=Model(model.input,l_out)
        self.VGG.compile('adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        
        
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/kvasir-dataset-v2/Mediaeval/"        
        self.train_dir = folder
        self.val_dir   = folder
        
        self.batch_size = 20
        self.epochs = 20
        
    def train(self):
        
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            horizontal_flip = True,
            vertical_flip = True, 
            fill_mode = "nearest",
            zoom_range = 0.3,
            width_shift_range = 0.3,
            height_shift_range=0.3,
            rotation_range=30)
        
        train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            target_size = (self.img_cols, self.img_rows),
                                                            batch_size = self.batch_size, 
                                                            class_mode = "categorical")
        
        test_datagen = ImageDataGenerator(
            rescale = 1./255,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = "nearest",
            zoom_range = 0.3,
            width_shift_range = 0.3,
            height_shift_range=0.3,
            rotation_range=30)
        
        validation_generator = test_datagen.flow_from_directory(self.val_dir,
                                                                target_size = (self.img_cols, self.img_rows),
                                                                class_mode = "categorical")
        
        
        checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
        
        
        # Train the model 
        self.VGG.fit_generator(
            train_generator,
            samples_per_epoch = 5000,
            epochs = 1,
            #validation_data = validation_generator,
            #nb_val_samples = nb_validation_samples,
            callbacks = [checkpoint, early])
        
        
    def make_model(self):
        model = applications.VGG19(weights = "imagenet", include_top=False,input_shape = self.img_shape)
        #model.summary()
        
        #Freezing 
        for layer in model.layers[:10]:
            layer.trainable = False
        
        #adding custom layer
        l=model.output
        l=Flatten()(l)
        l=Dense(512, activation='relu')(l)
        l=Dropout(0.25)(l)
        l=Dense(512, activation='relu')(l)
        l_out=Dense(5,activation='softmax')(l) 
        return model,l_out



transfer=TL()
transfer.train()