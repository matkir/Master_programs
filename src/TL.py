from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os
import gan.contextencoder as auto



class TL():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256        
        self.channels = 3
        self.img_shape = (self.img_cols, self.img_rows, self.channels)
        model,l_out=self.make_model()
        o=Adam(lr=0.0003)
        self.VGG=Model(model.input,l_out)
        self.VGG.compile(o, loss='categorical_crossentropy', metrics=['accuracy']) 
        
        
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/mediaeval/Medico_2018_development_set/"        
        self.train_dir = folder
        self.val_dir   = folder
        
        self.batch_size = 20
        
    def train(self):
        a=auto.ContextEncoder(self.img_cols,self.img_rows)
        #a.build_model()
        #a.train_model()
        a.load_model()
        a.load_model_weights()
        prepro=a.build_wrapper()
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            horizontal_flip = True,
            vertical_flip = True, 
            fill_mode = "nearest",
            zoom_range = 0.3,
            width_shift_range = 0.3,
            height_shift_range=0.3,
            rotation_range=30,
            validation_split=0.1,
            preprocessing_function=prepro)
        
        val_datagen = ImageDataGenerator(
            rescale = 1./255,
            validation_split=0.9)    
        
        train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            target_size = (self.img_cols, self.img_rows),
                                                            batch_size = self.batch_size, 
                                                            class_mode = "categorical")
        
        val_generator = train_datagen.flow_from_directory(self.train_dir,
                                                                target_size = (self.img_cols, self.img_rows),
                                                                batch_size = self.batch_size, 
                                                                class_mode = "categorical")        

        path, dirs, files = next(os.walk('./logs'))
        file_count=len(files)
        #checkpoint = ModelCheckpoint(f"vgg16_{file_count}.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        #checkpoint = ModelCheckpoint(f"densenet_{file_count}.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        checkpoint = ModelCheckpoint(f"resNet50_{file_count}.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
        board = TensorBoard(f"./logs/run_{file_count}")
        
        # Train the model 
        self.VGG.fit_generator(
            train_generator,
            epochs = 10,
            validation_data=train_generator,
            callbacks = [checkpoint, early, board])

        
    def make_model(self):
        #model = applications.VGG19(weights = "imagenet", include_top=False,input_shape = self.img_shape)
        #model = applications.DenseNet201(weights='imagenet',include_top=False,input_shape=self.img_shape)     
        model = applications.ResNet50(include_top=False, input_shape=self.img_shape)
        model.summary()
        
        #Freezing 
        for layer in model.layers[:600]:
            layer.trainable = False
        
        #adding custom layer
        l=model.output
        l=Flatten()(l)
        l=Dense(512, activation='relu')(l)
        l=Dropout(0.25)(l)
        l=Dense(512, activation='relu')(l)
        l_out=Dense(16,activation='softmax')(l) 
        return model,l_out



transfer=TL()
transfer.train()
