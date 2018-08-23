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
        img_count=input_img.shape[0]
        img_cols=input_img.shape[1]
        img_rows=input_img.shape[2]
        channels=input_img.shape[3]
        one_image=False
    else:
        img_count=1
        img_cols=input_img.shape[0]
        img_rows=input_img.shape[1]
        channels=input_img.shape[2]
        one_image=True
    if one_image:
        input_img=np.expand_dims(input_img, 0)
            
    img_shape = (img_count, img_cols, img_rows, channels)
    template_shape = (int(img_shape[2]*0.3),int(img_shape[1]*0.3)) #template is approx 30% of the original shape
    
    folder=os.path.expanduser("~")
    folder=folder+"/Documents/kvasir-dataset-v2/green_templates/" 
    
    template=np.zeros(shape=(img_shape[0], template_shape[1], template_shape[0], img_shape[3]),dtype=np.float32)

    imgs=np.random.choice(os.listdir(folder),img_shape[0],replace=True)    
    for i,j in enumerate(imgs):
        img = cv2.cvtColor(cv2.imread(folder+j), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(template_shape[0],template_shape[1]))
        img = (img.astype(np.float32) - 127.5)
        img = img / 127.5
        template[i]=img
    
    """
    (34, 382) (250, 554)
    in standard img if 576,720
    
    """
    pos=[0,1,2,3] 
    pos[0]=int(382/576*img_cols)
    pos[1]=int(554/576*img_cols)
    pos[2]=int(34/720*img_rows)
    pos[3]=int(250/720*img_rows)
    
    output_img=input_img.copy()
    for i in range(img_count):
        output_img[i,pos[0]:pos[1],pos[2]:pos[3]]=template[i]

    return input_img,output_img



if __name__=='__main__':
    import plotload
    
    import matplotlib.pyplot as plt
    #for i in range(10):
    #    a=plotload.load_one_img((576,720), dest='green', crop=False, glare=False,printable=True)
    #    _make_square_from_green_img(a)
    #_find_dims(a)
    b=plotload.load_polyp_batch((576//2,720//2,3), 10, crop=False)
    c,d=add_green_suare(b)
    plt.imshow(c)
    plt.show()
    plt.imshow(d)
    plt.show()