import os
import cv2
from tqdm import tqdm
from shutil import copyfile
import sys
import uuid
"""
Navie sorting, just looking at pixel [520,50] to see if the px value is below a tresh



"""

def data(folders):
    
    if not os.path.exists(green_folder):
        os.makedirs(green_folder)    
    if not os.path.exists(none_folder):
        os.makedirs(none_folder)    

    for folder in folders:
        f=os.listdir(folder)
        for img in tqdm(f):
            path=os.path.join(folder,img)
            save=cv2.imread(path)
            save=cv2.resize(save,(576,720))
            if save[520,50,2] < 50 and save[520,50,1] >100 and save[520,50,0]>100:
                copyfile(path, os.path.join(green_folder,str(uuid.uuid1())))
            else:
                if '78f877f3-083c-40ef-a82c-65764f6ab285' in img:
                    print(img)
                copyfile(path, os.path.join(none_folder,str(uuid.uuid1())))




root='/home/mathias/Documents/kvasir-dataset-v2/med/'
folders=[]
folders.append(root+'blurry-nothing')
folders.append(root+'colon-clear')
folders.append(root+'dyed-lifted-polyps')
folders.append(root+'dyed-resection-margins')
folders.append(root+'instruments')
folders.append(root+'normal-cecum')
folders.append(root+'polyps')
folders.append(root+'stool-inclusions')
folders.append(root+'stool-plenty')
folders.append(root+'ulcerative-colitis')
none_folder =root+'none' 
green_folder =root+'green' 
data(folders)
