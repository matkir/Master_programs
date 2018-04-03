import os
import cv2
from tqdm import tqdm
from shutil import copyfile
"""
Navie sorting, just looking at pixel [520,50] to see if the px value is below a tresh



"""


def data():
    folder ='../../../kvasir-dataset-v2/blanding' 
    green_folder ='../../../kvasir-dataset-v2/green' 
    none_folder ='../../../kvasir-dataset-v2/none' 
    
    for img in tqdm(os.listdir(folder)):
        path=os.path.join(folder,img)
        save=cv2.imread(path)
        if save[520,50,2] < 50 and save[520,50,1] >100 and save[520,50,0]>100:
            copyfile(path, os.path.join(green_folder,img))
        else:
            if '78f877f3-083c-40ef-a82c-65764f6ab285' in img:
                print(img)
            copyfile(path, os.path.join(none_folder,img))
data()
