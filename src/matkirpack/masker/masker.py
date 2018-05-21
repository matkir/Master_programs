import scipy.misc
import numpy as np
import pygame, sys 
from PIL import Image 


def mask_randomly_square(imgs,mask_width,mask_height,mask_val=0):
    """
    Mask randomly sqare takes a series of images (num images,image rows, image cols,channels) and masks an area with the chosen val
    :param imgs: list of imgs
    :param mask_width: width of the mask
    :param mask_height: height of the mask
    :param val: the value that the mask is filled with
    :return: (The masked imgs,the part that is removed,pos of the removed part)
    """
    
    img_num,img_rows,img_cols,img_channels=imgs.shape
    y1 = np.random.randint(0, img_cols - mask_height, img_num)
    y2 = y1 + mask_height
    x1 = np.random.randint(0, img_rows - mask_width, img_num)
    x2 = x1 + mask_width
    
    masked_imgs = np.empty_like(imgs)
    missing_parts = np.empty((imgs.shape[0],mask_width,mask_height, img_channels))
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
        missing_parts[i] = masked_img[_x1:_x2, _y1:_y2, :].copy()
        masked_img[_x1:_x2, _y1:_y2, :] = mask_val
        masked_imgs[i] = masked_img

    return masked_imgs, missing_parts, (y1, y2, x1, x2)
def mask_green_corner(imgs,val=0):
    """
    Mask the bottom area where the green sqare is. takes a series of images (num images,image rows, image cols,channels) and masks an area with the chosen val
    :param imgs: list of imgs
    :param mask_width: width of the mask
    :param mask_height: height of the mask
    :param val: the value that the mask is filled with
    :return: (The masked imgs,the part that is removed,pos of the removed part)
    """
    img_num,img_cols,img_rows,img_channels=imgs.shape
    box_coord=(int(0),int(0.35*img_rows),int(0.65*img_cols),int(img_cols))
    
    y1=box_coord[0]    
    y2=box_coord[1]    
    x1=box_coord[2]    
    x2=box_coord[3]    
    
    
    
    masked_imgs = np.empty_like(imgs)
    missing_parts = np.empty((img_num ,x2-x1 ,y2-y1 ,img_channels))
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        missing_parts[i] = masked_img[x1:x2, y1:y2, :].copy()
        masked_img[x1:x2, y1:y2, :] = val
        masked_imgs[i] = masked_img

    return masked_imgs, missing_parts, (y1, y2, x1, x2)

def mask_from_template(imgs,template_folder="templates",val=0,fliplr=True,flipud=True,rot=True):
    img_num,img_cols,img_rows,img_channels=imgs.shape
    mask=pl.load_single_template((img_cols,img_rows),template_folder,fliplr=True,flipud=True,rot=True)
    
    masked_imgs   = np.empty_like(imgs)
    missing_parts = np.empty_like(imgs)
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        rest       = img.copy()
        rest[:,:,0]       = np.multiply(masked_img[:,:,0],np.logical_not(mask))
        rest[:,:,1]       = np.multiply(masked_img[:,:,1],np.logical_not(mask))
        rest[:,:,2]       = np.multiply(masked_img[:,:,2],np.logical_not(mask))
        masked_img[:,:,0] = np.multiply(masked_img[:,:,0],mask)
        masked_img[:,:,1] = np.multiply(masked_img[:,:,1],mask)
        masked_img[:,:,2] = np.multiply(masked_img[:,:,2],mask)
        masked_imgs[i] = masked_img
        missing_parts[i] = rest

    return masked_imgs, missing_parts 


if __name__=='__main__':    
    import plotload as pl
    import matplotlib.pyplot as plt  
    img=pl.load_polyp_batch((576,720,3),5,data_type="green",rot=False)
    #a,b,c=mask_randomly_square(img, 100, 100)
    #a,b,c=mask_green_corner(img)
    a,b=mask_from_template(img)
    plt.imshow(0.5*b[0]+0.5)
    plt.show()
    plt.imshow(0.5*a[0]+0.5)
    plt.show()
