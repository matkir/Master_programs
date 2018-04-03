from plotload import *
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from shutil import copyfile

def data():
    folder ='../../../kvasir-dataset-v2/polyps' 
    green_folder ='../../../kvasir-dataset-v2/green' 
    none_folder ='../../../kvasir-dataset-v2/none' 
    
    for img in tqdm(os.listdir(folder)):
        path=os.path.join(folder,img)
        save=cv2.imread(path)
        if save[520,50][2] < 3 and save[520,50][0]>140:
            copyfile(path, os.path.join(green_folder,img))
        else:
            copyfile(path, os.path.join(none_folder,img))
data()
