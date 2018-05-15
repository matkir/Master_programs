import sys,os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def _reduce_glare(input_img,tol=250,avg_mode=False):
    """
    Clips images, so that the highest values are lower
    """
    if not avg_mode:
        return np.clip(input_img, 0, tol)
    avg=np.average(input_img[(input_img.shape[0]//3):(2*input_img.shape[0]//3),(input_img.shape[1]//3):(2*input_img.shape[1]//3),0])
    avg1=np.average(input_img[(input_img.shape[0]//3):(2*input_img.shape[0]//3),(input_img.shape[1]//3):(2*input_img.shape[1]//3),1])
    avg2=np.average(input_img[(input_img.shape[0]//3):(2*input_img.shape[0]//3),(input_img.shape[1]//3):(2*input_img.shape[1]//3),2])
    input_img[input_img[:,:,0]>tol]=-avg
    input_img[input_img[:,:,1]>tol]=-avg1
    input_img[input_img[:,:,2]>tol]=-avg2
    return input_img
def _mask_around(input_img,tol):
    pass
    
    
def _crop_img(input_img,gray,tol=20,erosion=True):
    """
    Removes the black bars around images
    """
    if erosion:
        kernel = np.ones((5,5),np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)        
    mask = gray>tol
    return input_img[np.ix_(mask.any(1),mask.any(0))]

def load_polyp_data(img_shape,data_type=None,rot=False,crop=True,glare=False):
    """
    Loads the polyp data
    """
    if '-l' in sys.argv:
        return np.load("train_data.npy")
    import os
    if data_type==None:
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/kvasir-dataset-v2/none"
        #folder ='../../../../../kvasir-dataset-v2/none' #TODO MAKE STATIC
    else:
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/kvasir-dataset-v2/blanding"
        #folder ='../../../../../kvasir-dataset-v2/blanding' #TODO MAKE STATIC
    
    if rot:
        #Rot takes 4 times as many pics
        data=np.ndarray(shape=(len(os.listdir(folder))*4, img_shape[0], img_shape[1], img_shape[2]),dtype=np.int32)
        print(f"loading {len(os.listdir(folder))*4} images")
    else:
        data=np.ndarray(shape=(len(os.listdir(folder)), img_shape[0], img_shape[1], img_shape[2]),dtype=np.int32)
        print(f"loading {len(os.listdir(folder))} images")

    i=0
    for img in tqdm(os.listdir(folder)):
        path=os.path.join(folder,img)
        save=cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if crop:
            gray = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
            save=_crop_img(save,gray)
        save=cv2.resize(save,(img_shape[1],img_shape[0]))
        if rot:
            for r in [0,90,180,270]:
                M = cv2.getRotationMatrix2D((img_shape[1]/2,img_shape[0]/2),r,1)
                dst = cv2.warpAffine(save,M,(img_shape[1],img_shape[0]))
                data[i]=dst
                i+=1
        else:    
            data[i]=save
            i+=1
    #data=np.random.permutation(data)
    data = (data.astype(np.float32) - 127.5) / 127.5
    np.save("train_data.npy", data)
    return data


def load_polyp_batch(img_shape,batch_size,data_type=None,rot=False,crop=True,glare=False):
    """
    Loads the polyp data, in a for of random images from a batch
    """
    import os
    if data_type==None:
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/kvasir-dataset-v2/none"
        #folder ='../../../../../kvasir-dataset-v2/none' #TODO MAKE STATIC
    else:
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/kvasir-dataset-v2/blanding"
        #folder ='../../../../../kvasir-dataset-v2/blanding' #TODO MAKE STATIC
    
    data=np.ndarray(shape=(batch_size, img_shape[0], img_shape[1], img_shape[2]),dtype=np.int32)

    i=0
    imgs=np.random.choice(os.listdir(folder),batch_size,replace=True)
    for img in imgs:
        path=os.path.join(folder,img)
        save=cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if crop:
            gray = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
            save=_crop_img(save,gray)
        save=cv2.resize(save,(img_shape[1],img_shape[0]))
        
        r=np.random.choice([0,90,180,270],p=[0.7,0.1,0.1,0.1])
        if r !=0:
            M = cv2.getRotationMatrix2D((img_shape[1]/2,img_shape[0]/2),r,1)
            dst = cv2.warpAffine(save,M,(img_shape[1],img_shape[0]))
            data[i]=dst
            i+=1
        else:    
            data[i]=save
            i+=1

    data = (data.astype(np.float32) - 127.5) / 127.5
    np.save("train_data.npy", data)
    return data

def load_one_img(img_shape,dest=None,crop=True,glare=True):
    """
    Loads a spessific img, or random if non declared
    """
    import os
    if dest==None:
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/kvasir-dataset-v2/green/"
        img=folder+np.random.choice(os.listdir(folder),1)[0]
    else:
        img=dest
        
    
    save=cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2GRAY)
    if crop:
        save=_crop_img(save,gray)
    if glare:
        save=_reduce_glare(save)
    save=cv2.resize(save,(img_shape[0],img_shape[1]))
    data = (save.astype(np.float32) - 127.5) / 127.5
    return data,img

        

if __name__=='__main__':
    a=load_polyp_batch((500,500,3), 100, rot=True)
    a=load_one_img((1000,1000,3))
    print()