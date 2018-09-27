import scipy.misc
import plotload as pl
import numpy as np
import pygame, sys 
from PIL import Image
import os, random
import cv2
import uuid
import matplotlib.pyplot as plt
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
def is_green(input_img):
    a,b,c,d = find_square_coords(input_img)
    if a==False:
        return False
    if len(input_img.shape)==4:
        input_img=np.squeeze(input_img,0)
    
    if(input_img[(a+b)//2,(c+d)//2,0]<0 and input_img[(a+b)//2,(c+d)//2,1]>0):
        return True
    return False
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


def memoize(f):
    memo = {(256, 256, 3): (168, 256, 0, 88)}
    memoize.has_run=False
    def helper(x):
        y=x.shape[-3:]
        if y not in memo:            
            memo[y] = f(x)
        else:
            if not memoize.has_run:
                print(f"using predetermined green for {y}:{memo[y]}")
                memoize.has_run=True
        return memo[y]
    return helper

 
        
def find_square_coords(input_imgs):
        """
        takes either one of multiple images and finds bottom right corner
        """
        img_num,img_cols,img_rows,img_channels=input_imgs.shape
        
        borderx=0.8*img_rows
        bordery=0.1*img_cols
        bottom_left=[]
        top_right=[]
        img2=np.squeeze(input_imgs[0].copy()*0.5+0.5)
        for input_img in input_imgs:
            #img2=np.squeeze(input_img.copy()*0.5+0.5)
            img=(input_img*127.5+127.5).astype(np.uint8)
            img=cv2.Canny(img, 0, 300)
            minLineLength=img_cols*0.18
            lines = cv2.HoughLinesP(image=img,rho=1,theta=np.pi/180, threshold=70,lines=np.array([]), minLineLength=minLineLength,maxLineGap=3)
            l2=[]
            #plt.imshow(img)
            #plt.show()
            if lines is None:
                continue
            lines=np.squeeze(lines,axis=1)
            for l in lines:
                """Really sorry for this ugly code! it finds the lines that are inside the area"""
                if l[0]<bordery and l[2]>bordery and l[0]<img_rows*0.6 and l[2]<img_rows*0.4 and abs(l[1]-l[3])<5:
                    img2=np.squeeze(input_img.copy()*0.5+0.5)
                    cv2.line(img2,(l[0],l[1]),(l[2],l[3]),0.5,4)
                    plt.imshow(img2)
                    l2.append(((l[2],l[3])))
                    l2.append(((l[0],l[1])))
                if l[1]>borderx and l[3]<borderx and l[1]>img_cols*0.4 and l[3]>img_cols*0.4:
                    cv2.line(img2,(l[0],l[1]),(l[2],l[3]),1,4)
                    l2.append(((l[0],l[1])))
                    l2.append(((l[2],l[3])))
            if len(l2)<2:
                continue
            bottom_left.append((min(l2, key = lambda t: t[0])[0],(max(l2, key = lambda t: t[1]))[1]))
            top_right.append(((max(l2, key = lambda t: t[0])[0]+1),((min(l2, key = lambda t: t[1]))[1])-1))

        if not bottom_left or not top_right:
            return False,False,False,False
        top_right=max(set(top_right), key=top_right.count)
        bottom_left=max(set(bottom_left), key=bottom_left.count)
        
        img2=np.squeeze(input_imgs[0].copy()*0.5+0.5)
        cv2.line(img2,bottom_left,top_right,(0,255,0),1) 

        print('img coords: ',top_right,bottom_left)
        return top_right[1], bottom_left[1], bottom_left[0], top_right[0]

find_square_coords=memoize(find_square_coords)
             
def add_green_suare(input_img):
    if len(input_img.shape)==4:
        img_count=input_img.shape[0]
        img_cols=input_img.shape[1]
        img_rows=input_img.shape[2]
        channels=input_img.shape[3]
        y1,y2,x1,x2 = find_square_coords(input_img[0])
        one_image=False
    else:
        img_count=1
        img_cols=input_img.shape[0]
        img_rows=input_img.shape[1]
        channels=input_img.shape[2]
        one_image=True    
        y1,y2,x1,x2 = find_square_coords(input_img)
    
    if one_image:
        input_img=np.expand_dims(input_img, 0)
            
    img_shape = (img_count, img_cols, img_rows, channels)
    template_shape = (int(x2-x1),int(y2-y1))
    
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

    output_img=input_img.copy()
    for i in range(img_count):
        output_img[i,y1:y2,x1:x2]=template[i]

    return input_img,output_img



if __name__=='__main__':
    import plotload
    
    import matplotlib.pyplot as plt
    #for i in range(10):
    #    a=plotload.load_one_img((576,720), dest='green', crop=False, glare=False,printable=True)
    #    _make_square_from_green_img(a)
    #_find_dims(a)
    b=plotload.load_polyp_batch((260,260,3), 10,data_type='green', crop=False)
    find_square_coords(b)
    
    
    c,d=add_green_suare(b)
    b=plotload.load_polyp_batch((260,260,3), 10,data_type='none', crop=False)
    c,d=add_green_suare(b)
    plt.imshow(c[0]*0.5+0.5)
    plt.show()
    plt.imshow(d[0]*0.5+0.5)
    plt.show()