from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os
#import ..autoencoder.dcae as auto

class TL():
    def __init__(self,folder,discriptor):
        self.img_rows = 256
        self.img_cols = 256        
        self.channels = 3
        self.img_shape = (self.img_cols, self.img_rows, self.channels)
        model,l_out=self.make_model()
        #o=Adam()
        self.lr=0.004 #0.004
        self.epoch=50
        self.patience=5
        self.otype="SGD"
        self.batch_size = 24
        o=SGD(lr=self.lr, nesterov=True)
        self.model=Model(model.input,l_out)
        self.model.compile(o, loss='categorical_crossentropy', metrics=['accuracy']) 
        
        
        #folder=os.path.expanduser("~")
        self.discriptor=discriptor
        #folder="/media/mathias/A_New_Hope/medicoTL/run_AE/medico/"        
        self.folder=folder
        self.train_dir = self.folder+"train"
        self.val_dir   = self.folder+"val"
        self.weight_dir= self.folder+"../"
        
    def train(self):
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            horizontal_flip = True,
            vertical_flip = True,
            )#rotation_range=30)
        
        val_datagen = ImageDataGenerator(
            rescale = 1./255)
        
        train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            target_size = (self.img_cols, self.img_rows),
                                                            batch_size = self.batch_size, 
                                                            class_mode = "categorical")
        
        val_generator = val_datagen.flow_from_directory(self.val_dir,
                                                                target_size = (self.img_cols, self.img_rows),
                                                                batch_size = self.batch_size, 
                                                                class_mode = "categorical")        

        path, dirs, files = next(os.walk('./logs'))
        file_count=len(files)
        #checkpoint = ModelCheckpoint(f"vgg16_{file_count}.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        #checkpoint = ModelCheckpoint(f"densenet_{file_count}.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        #checkpoint = ModelCheckpoint(f"resNet50_{file_count}.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        checkpoint = ModelCheckpoint(folder+f"InceptionResNetV2_{self.discriptor}_{file_count+1}.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=self.patience, verbose=1, mode='auto')
        board = TensorBoard(f"./logs/run_{self.discriptor}_{file_count+1}")
        f = open(f"training_info_{file_count+1}.txt","w+")
        f.write(f"\nLr={self.lr}\n")
        f.write(f"\nEpoch={self.epoch}\n")
        f.write(f"\nPatience={self.patience}\n")
        f.write(f"\nType={self.otype}\n")
        f.write(f"\nBatch={self.batch_size}\n")
        f.close()
        # Train the model 
        self.model.fit_generator(
            train_generator,
            epochs = self.epoch,
            validation_data=val_generator,
            callbacks = [checkpoint, early, board])

        
    def make_model(self):
        model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False,input_shape=self.img_shape) 

        #Freezing 
        #for layer in model.layers[:600]:
        #    layer.trainable = False
        
        #adding custom layer
        l=model.output
        l=GlobalAveragePooling2D()(l)
        l_out=Dense(16,activation='softmax')(l) 
        return model,l_out



folder="/media/mathias/A_New_Hope/medicoTL/run_vanilla/medico/"        
transfer=TL(folder,"Vanilla")
transfer.train()

folder="/media/mathias/A_New_Hope/medicoTL/run_CCGAN/medico/"        
transfer=TL(folder,"CC_GAN")
transfer.train()

folder="/media/mathias/A_New_Hope/medicoTL/run_ce/medico/"        
transfer=TL(folder,"Contextencoder")
transfer.train()

folder="/media/mathias/A_New_Hope/medicoTL/run_AE/medico/"        
transfer=TL(folder,"Autoencoder")
transfer.train()

folder="/media/mathias/A_New_Hope/medicoTL/run_clip/medico/"        
transfer=TL(folder,"Clip")
transfer.train()

