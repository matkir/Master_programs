import sys,os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
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
    
def _crop_center(img,crop_size):
    y,x,_= img.shape
    return img[0+crop_size:y-crop_size,0+crop_size:x-crop_size,:]    

def _crop_img(input_img,gray,tol=20,erosion=True,total=False):
    """
    Removes the black bars around images
    """
    if total:
        kernel = np.ones((20,20),np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)        
        mask=gray>20
        mask = cv2.morphologyEx(mask.astype(np.float32), cv2.MORPH_OPEN, kernel)      
        ret_img=input_img[np.ix_(mask.any(1),mask.any(0))]
        return _crop_center(ret_img,int(input_img.shape[0]*total))
    if erosion:
        kernel = np.ones((5,5),np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)        
    mask = gray>tol
    return input_img[np.ix_(mask.any(1),mask.any(0))]

def _find_folder(data_type):
    """
    Finds the folder used from the datatype
    :param data_type: path in kvasir
    :return: abs path to folder
    """
    if data_type==None:
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/kvasir-dataset-v2/none/"
    elif type(data_type) == str:
        if '.png' in data_type or '.jpg' in data_type:
            return data_type
        if data_type[0]=='/':
            return data_type
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/kvasir-dataset-v2/"+data_type+"/"
        if not os.path.isdir(folder):
            folder=os.path.expanduser("~")
            folder=folder+"/Documents/kvasir-dataset-v2/blanding/"
    else:
        folder=os.path.expanduser("~")
        folder=folder+"/Documents/kvasir-dataset-v2/blanding/"
    return folder


def load_polyp_data(img_shape,data_type=None,rot=False,crop=True,glare=False):
    """
    Loads the polyp data
    """
    if '-l' in sys.argv:
        return np.load("train_data.npy")
    
    folder=_find_folder(data_type)
    
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
    data = (data.astype(np.float32) - 127.5) / 127.5
    np.save("train_data.npy", data)
    return data


def load_polyp_batch(img_shape,batch_size,data_type=None,rot=False,crop=True,glare=False):
    """
    Loads the polyp data, in a for of random images from a batch
    """
   
    folder = _find_folder(data_type)
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

def load_one_img(img_shape,dest=None,crop=False,glare=False,total=0,printable=False,extra_dim=False):
    """
    Loads a spessific img, or random if non declared
    """
    
    folder =_find_folder(dest)
    if not '.png' in folder and not '.jpg' in folder:
        img=folder+np.random.choice(os.listdir(folder),1)[0]    
    else:
        img=folder
    
    save=cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2GRAY)
    if crop:
        save=_crop_img(save,gray,total=total)
    if glare:
        save=_reduce_glare(save)
    save=cv2.resize(save,(img_shape[1],img_shape[0]))
    data = (save.astype(np.float32) - 127.5) / 127.5
    if printable:
        data=data*0.5+0.5
        return data
    if extra_dim:
        data=np.expand_dims(data, axis=0)
        return data
    return data,img


def load_single_template(img_shape,dest='templates',fliplr=True,flipud=True,rot=True):
    folder=_find_folder(dest)
    if folder != dest:
        folder=folder+np.random.choice(os.listdir(folder),1)[0]
    else:
        folder=np.random.choice(glob.glob(folder+'/*.npy'),1)[0]
    arr=np.load(folder)
    #remember that numpy takes [x,y], but img_shape is [y,x,channel]
    if arr.shape[0]!=img_shape[1]:
        arr=cv2.resize(arr,(img_shape[1],img_shape[0]))
    if fliplr and np.random.choice([True, False]):
        arr=np.fliplr(arr) 
    if flipud and np.random.choice([True, False]):
        arr=np.flipud(arr)
    if rot:
        M = cv2.getRotationMatrix2D((img_shape[1]/2,img_shape[0]/2),np.random.randint(360),1)
        arr = cv2.warpAffine(arr,M,(img_shape[1],img_shape[0]))        
    return arr
        

       

if __name__=='__main__':
    a=load_one_img((576,720,3),dest='2.jpg',total=0.08,crop=True)
    """
    plt.imshow(0.5*a[0]+0.5)
    plt.show()    
    a=load_polyp_batch((576,720,3), 100, rot=True)
    plt.imshow(0.5*a[0]+0.5)
    plt.show()    
    a=load_polyp_data((576//2,720//2,3),data_type="polyps")
    """
    plt.imshow(0.5*a[0]+0.5)
    plt.show()    
    print()