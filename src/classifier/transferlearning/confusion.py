import numpy as np
import matplotlib.pyplot as plt
import plotload
from keras.models import load_model, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import applications
import os
from tqdm import tqdm
import cv2
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, f1_score
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

def guess_label(imgs,model,tresh):
    """
    out-of-patient
    instruments
    dyed-lifted-polyps
    dyed-resection-margins
    polyps
    esophagitis
    ulcerative-colitis
    retroflex-rectum
    retroflex-stomach
    normal-cecum
    normal-pylorus
    normal-z-line
    stool-plenty
    stool-inclusions
    colon-clear
    blurry-nothing    
    """
    #we only care about [2,3,9,10,11,12,15] where the mixup happens
    priority=np.array([15,14,2,3,5,1,9,10,11,0,4,7,8,13,12,6])
    label=model.predict(imgs)
    ret=np.zeros(shape=(imgs.shape[0]))[:,None]
    check=lambda a, b: any(i in b for i in a)
    for i,l in enumerate(label):
        get=np.argwhere(l > tresh)
        if check([2,3,9,10,11,12,15],np.squeeze(get)) and len(get)>1:
            ret[i]=np.squeeze(get)[np.argmin(priority[np.squeeze(get)])]
            print(ret[i-1]," becomes: ",ret[i])
        else:
            ret[i]=np.argmax(l)
    return ret  
        
            
    
    
def run(testing_name,testing_type,num):
    #testing_name  = "CC_GAN"
    #testing_type  = "run_CCGAN"
    folder        = f"/media/mathias/A_New_Hope/medicoTL/{testing_type}/"        
    test_dir      = folder+"preprocessed_test/Medico_2018_test_set"
    train_dir     = folder+"medico/train/"
    val_dir       = folder+"medico/val/"
    weight_dir    = folder+"medico/"
    weight_type   = f"InceptionResNetV2_{testing_name}_{num}.h5"
    
    model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False,input_shape=(256,256,3)) 
    l=model.output
    l=GlobalAveragePooling2D()(l)
    l_out=Dense(16,activation='softmax')(l) 
    model=Model(model.input,l_out)
    model.load_weights(weight_dir+weight_type)
    
    classes =  ['blurry-nothing','colon-clear','dyed-lifted-polyps','dyed-resection-margins',
                'esophagitis','instruments','normal-cecum','normal-pylorus',
                'normal-z-line','out-of-patient','polyps','retroflex-rectum',
                'retroflex-stomach','stool-inclusions','stool-plenty','ulcerative-colitis'
                ]
    
    cpt = sum([len(files) for r, d, files in os.walk(val_dir)])
    #(0, ) = True , (1, ) = prediceted
    confusion_data=np.zeros(shape=(cpt,2),dtype=np.int)
    
    cnt=0
    for class_num,class_name in enumerate(classes):
        path=f"{val_dir}{class_name}"
        imgs=load_data((256,256,3),path)
        #label=np.argmax(model.predict(imgs),axis=-1)
        label=guess_label(imgs,model,0.20)
        print()
        for l in label:
            confusion_data[cnt,0]=int(class_num)
            confusion_data[cnt,1]=int(l)
            cnt+=1
    print(cnt,cpt)
    f=open(f'{testing_name}_{weight_type}.txt',"w+")
    print(confusion_data)
    f.write("\nAccuracy_score\n")
    f.write(f"{accuracy_score(confusion_data[:,0], confusion_data[:,1])}")    
    f.write("\nf1_score\n")
    f.write(f"{f1_score(confusion_data[:,0], confusion_data[:,1],average='weighted')}")
    f.write("\nMatthews_corrcoef\n")
    f.write(f"{matthews_corrcoef(confusion_data[:,0], confusion_data[:,1])}")
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(confusion_data[:,0], confusion_data[:,1] )
    np.set_printoptions(precision=2)
    f.write("\nC_Matrix\n")
    f.write(f"{cnf_matrix}")
    f.close()
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(20,20))
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title=f'{testing_name},{weight_type}')
    plt.savefig(f'{testing_name}_{weight_type}.png')
    #plt.show()
    print()

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
