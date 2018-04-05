import sys,os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
def load_polyp_data(img_shape,data_type=None):
    """
    Loads the polyp data
    """
    if '-l' in sys.argv:
        return np.load("train_data.npy")
    if data_type==None:
        folder ='../../../../kvasir-dataset-v2/none' #TODO MAKE STATIC
    else:
        folder ='../../../../kvasir-dataset-v2/blanding' #TODO MAKE STATIC
    data=np.ndarray(shape=(len(os.listdir(folder)), img_shape[0], img_shape[1], img_shape[2]),dtype=np.int32)
    print(f"loading {len(os.listdir(folder))} images")
    i=0
    for img in tqdm(os.listdir(folder)):
        path=os.path.join(folder,img)
        save=cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        save=cv2.resize(save,(img_shape[1],img_shape[0]))
        data[i]=save#(np.roll(np.array(save),1,axis=-1))
        i+=1
    data = (data.astype(np.float32) - 127.5) / 127.5
    np.save("train_data.npy", data)
    return data

def plot_1_to_255(enc_img,dec_img,ae_img,real_img,epoch):
    """
    takes images as -1,1 and converts them to 0,1 format.
    Name is a misnomer
    
    """
    fig, axs = plt.subplots(3, 4) #3 of each picture
    dec_img=np.clip((dec_img*0.5)+0.5,0,1)
    if len(enc_img.shape)==2:
        enc_img=np.repeat(np.expand_dims((enc_img*0.5)+0.5,axis=-1),enc_img.shape[1]//2,axis=-1) 
    else:
        enc_img=np.squeeze((enc_img*0.5)+0.5)#remove silly dim
    ae_img=np.clip((ae_img*0.5)+0.5,0,1)
    real_img=np.clip((real_img*0.5)+0.5,0,1)
    cnt1=0
    cnt2=0
    cnt3=0
    cnt4=0
    for i in range(3):
        for j in range(4):
            if j==0:
                axs[i,j].imshow(dec_img[cnt1, :,:,:])
                axs[i,j].axis('off')
                cnt1 += 1
            elif j==1:
                if len(enc_img.shape)==3:
                    axs[i,j].imshow(enc_img[cnt2, :,:])
                else:
                    axs[i,j].imshow(enc_img[cnt2, :,:,:])
                axs[i,j].axis('off')
                cnt2 += 1
            elif j==2:
                axs[i,j].imshow(ae_img[cnt3, :,:,:])
                axs[i,j].axis('off')
                cnt3 += 1
            elif j==3:
                axs[i,j].imshow(real_img[cnt4, :,:,:])
                axs[i,j].axis('off')
                cnt4 += 1
            else:
                raise IndexError    

    plt.suptitle('decoded img | encoded img | encoded then decoded', fontsize=16)
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()               
