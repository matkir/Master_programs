import os
import cv2
from tqdm import tqdm
from shutil import copyfile
import sys
"""
Navie sorting, just looking at pixel [520,50] to see if the px value is below a tresh



"""


def data():
    
    if not os.path.exists(green_folder):
        os.makedirs(green_folder)    
    if not os.path.exists(none_folder):
        os.makedirs(none_folder)    
   
    for img in tqdm(os.listdir(folder)):
        path=os.path.join(folder,img)
        save=cv2.imread(path)
        if save[520,50,2] < 50 and save[520,50,1] >100 and save[520,50,0]>100:
            copyfile(path, os.path.join(green_folder,img))
        else:
            if '78f877f3-083c-40ef-a82c-65764f6ab285' in img:
                print(img)
            copyfile(path, os.path.join(none_folder,img))

if len(sys.argv)!=2:
    folder ='../../../kvasir-dataset-v2/blanding' 
    none_folder ='../../../kvasir-dataset-v2/none' 
    green_folder ='../../../kvasir-dataset-v2/green' 
    data()
    folder ='../../../kvasir-dataset-v2/ulcerative-colitis' 
    data()
else:
    folder ='../kvasir-dataset-v2/blanding' 
    none_folder ='../kvasir-dataset-v2/none' 
    green_folder ='../kvasir-dataset-v2/green' 
    data()
    folder ='../kvasir-dataset-v2/ulcerative-colitis' 
    data()
