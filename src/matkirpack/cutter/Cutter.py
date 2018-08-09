import scipy.misc
import plotload as pl
import numpy as np
import pygame, sys 
from PIL import Image
import os, random
import cv2
import uuid
"""
Cutter does a couple of things
1) gather a database of squares contained in a subfolder
2) add a square to an input image, and returns both x and x~ 
3) takes an image with a square and replicate the opposite corner in the green corner

"""

def _make_square_from_green_img(input_img,show=False):
    top_left,bottom_right = _find_dims(input_img)
    print(top_left,bottom_right)
    img=input_img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0],:]
    if show:
        plt.imshow(img)
        plt.show()
    
    folder=os.path.expanduser("~")
    folder=folder+"/Documents/kvasir-dataset-v2/green_templates/"    
    filename = str(uuid.uuid4())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if(top_left[0]<50 and top_left[1]<400):
        cv2.imwrite(folder+filename+'.jpg', img*255)
    print()


def _find_dims(input_img,show=False):
    """
        finds the posisition of a green square located in an image.    
    """
    if len(input_img.shape)==4:
        img_cols=input_img.shape[1]
        img_rows=input_img.shape[2]
        channels=input_img.shape[3]
        one_image=False
        assert False
    else:
        img_cols=input_img.shape[0]
        img_rows=input_img.shape[1]
        channels=input_img.shape[2]
        one_image=True
       
    img_shape = (img_cols, img_rows, channels)
    template_shape = (int(img_shape[1]*0.3),int(img_shape[0]*0.3)) #template is approx 30% of the original shape
    
    
    template = cv2.imread('1.png')
    template = cv2.resize(template, template_shape)
    
    res = cv2.matchTemplate(input_img.astype(np.float32),template.astype(np.float32),cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)    
    
    top_left = max_loc
    bottom_right = (top_left[0] + template_shape[0], top_left[1] + template_shape[1]) 
    
    if show:
        a=input_img.copy()
        cv2.rectangle(a,top_left, bottom_right, 255, 1)
        plt.imshow(a,cmap = 'gray')
        plt.show()
    return top_left,bottom_right
    
def add_green_suare(input_img):
    if len(input_img.shape)==4:
        img_cols=input_img.shape[1]
        img_rows=input_img.shape[2]
        channels=input_img.shape[3]
        one_image=False
        assert False
    else:
        img_cols=input_img.shape[0]
        img_rows=input_img.shape[1]
        channels=input_img.shape[2]
        one_image=True
       
    img_shape = (img_cols, img_rows, channels)
    template_shape = (int(img_shape[1]*0.3),int(img_shape[0]*0.3)) #template is approx 30% of the original shape
    
    folder=os.path.expanduser("~")
    folder=folder+"/Documents/kvasir-dataset-v2/green_templates/"       
    template = cv2.imread(folder+random.choice(os.listdir(folder)))
    template = cv2.resize(template, template_shape)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    plt.imshow(template)
    plt.show()
    
    """
    (34, 382) (250, 554)
    in standard img if 576,720
    
    """
    output_img=input_img.copy()
    output_img[382:554,34:250]=template

    return input_img,output_img



if __name__=='__main__':
    import plotload
    
    import matplotlib.pyplot as plt
    #for i in range(10):
    #    a=plotload.load_one_img((576,720), dest='green', crop=False, glare=False,printable=True)
    #    _make_square_from_green_img(a)
    #_find_dims(a)
    b=plotload.load_one_img((576,720), dest='none', crop=False, glare=False,printable=True)
    c,d=add_green_suare(b)
    plt.imshow(c)
    plt.show()
    plt.imshow(d)
    plt.show()