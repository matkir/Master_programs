from keras.models import load_model, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import applications
import os
import plotload
from tqdm import tqdm
import numpy as np
import cv2
"""
File for making the sub data.
"""
def load_data(img_shape, folder):
    
    data=np.ndarray(shape=(len(os.listdir(folder)), img_shape[0], img_shape[1], img_shape[2]),dtype=np.float32)
    i=0
    for img in tqdm(sorted(os.listdir(folder))):
        path=os.path.join(folder,img)
        save=cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)    
        data[i]=save
        i+=1
    data=np.true_divide(data,255)
    return data

def run(testing_name,testing_type,num):
    #import 
    folder=f"/media/mathias/A_New_Hope/medicoTL/{testing_type}/"        
    test_dir   = folder+"preprocessed_test/Medico_2018_test_set"
    weight_dir = f'/media/mathias/A_New_Hope/medicoTL/{testing_type}/medico/'
    weight_type=f"InceptionResNetV2_{testing_name}_{num}.h5"
    
    
    
    classes =  ['blurry-nothing','colon-clear','dyed-lifted-polyps','dyed-resection-margins',
                'esophagitis','instruments','normal-cecum','normal-pylorus',
                'normal-z-line','out-of-patient','polyps','retroflex-rectum',
                'retroflex-stomach','stool-inclusions','stool-plenty','ulcerative-colitis'
                ]
    
    
    
    imgs=load_data((256,256,3),test_dir)
    model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False,input_shape=(256,256,3)) 
    l=model.output
    l=GlobalAveragePooling2D()(l)
    l_out=Dense(16,activation='softmax')(l) 
    model=Model(model.input,l_out)
    model.load_weights(weight_dir+weight_type)
    
    l=model.predict(imgs)
    f= open(f"memed_groupname_detection{weight_type}_{num}_3712.txt","w+")
    l_names=sorted(os.listdir(test_dir))
    for i,data in enumerate(l):
        f.write(f"{l_names[i]},{classes[np.argmax(data)]},{np.max(data)}\n")
    
    print("done!")


n=9    
testing_name  = "CC_GAN"
testing_type  = "run_CCGAN"
run(testing_name,testing_type,n)
testing_name  = "Contextencoder"
testing_type  = "run_ce"
run(testing_name,testing_type,n)
testing_name  = "Vanilla"
testing_type  = "run_vanilla"
run(testing_name,testing_type,n)
testing_name  = "Autoencoder"
testing_type  = "run_AE"
run(testing_name,testing_type,n)
testing_name  = "Clip"
testing_type  = "run_clip"
run(testing_name,testing_type,n)
