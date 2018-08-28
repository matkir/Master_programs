import scipy.misc
import numpy as np
import pygame, sys 
from PIL import Image 
import plotload as pl
import cutter

def mask_randomly_square(imgs,mask_height,mask_width,mask_val=0):
    """
    Mask randomly sqare takes a series of images (num images,image rows, image cols,channels) and masks an area with the chosen val
    :param imgs: list of imgs
    :param mask_width: width of the mask
    :param mask_height: height of the mask
    :param val: the value that the mask is filled with
    :return: (The masked imgs,the part that is removed,pos of the removed part)
    """
    
    img_num,img_cols,img_rows,img_channels=imgs.shape
    x1 = np.random.randint(0, img_cols - mask_height, img_num)
    x2 = x1 + mask_height
    y1 = np.random.randint(0, img_rows - mask_width, img_num)
    y2 = y1 + mask_width
    
    masked_imgs = np.empty_like(imgs)
    missing_parts = np.empty((imgs.shape[0],mask_height,mask_width, img_channels))
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
        tmp=masked_img.copy()
        missing_parts[i] = tmp[_x1:_x2, _y1:_y2, :]
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
    y1,y2,x1,x2=cutter.find_square_coords(imgs)
    
    
    
    masked_imgs = np.empty_like(imgs)
    missing_parts = np.empty((img_num ,y2-y1 ,x2-x1 ,img_channels))
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        missing_parts[i] = masked_img[y1:y2, x1:x2, :].copy()
        masked_img[y1:y2, x1:x2, :] = val
        masked_imgs[i] = masked_img

    return masked_imgs, missing_parts, (y1, y2, x1, x2)

def mask_from_template(imgs,template_folder="templates",val=0,fliplr=True,flipud=True,rot=True):
    if len(imgs.shape)==3:
        imgs=np.expand_dims(imgs, 0)
    img_num,img_cols,img_rows,img_channels=imgs.shape
    
    masked_imgs   = np.empty_like(imgs)
    missing_parts = np.empty_like(imgs)
    mask          = np.empty_like(imgs[:,:,:,1])
    for i in range(img_num):
        mask[i]=pl.load_single_template((img_cols,img_rows),template_folder,fliplr=fliplr,flipud=flipud,rot=rot)
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        rest       = img.copy()
        rest[:,:,0]       = np.multiply(masked_img[:,:,0],np.logical_not(mask[i]))
        rest[:,:,1]       = np.multiply(masked_img[:,:,1],np.logical_not(mask[i]))
        rest[:,:,2]       = np.multiply(masked_img[:,:,2],np.logical_not(mask[i]))
        masked_img[:,:,0] = np.multiply(masked_img[:,:,0],mask[i])
        masked_img[:,:,1] = np.multiply(masked_img[:,:,1],mask[i])
        masked_img[:,:,2] = np.multiply(masked_img[:,:,2],mask[i])
        masked_imgs[i] = masked_img
        missing_parts[i] = rest

    return masked_imgs, missing_parts, mask

def combine_imgs_with_mask(gen_img,org_img,mask):
    """
    Takes an image (org_img) and replaces parts of the image with the gen img part, where the mask is True
    If only one mask is given, every transformation will use the same mask.
    If as many masks as images, each mask will be used in order per image.
    
    :param gen_img: image array with only the parts needed to fill img (num_imgs,col,row,channel)
    :param org_img: image array that needs a new part (num_imgs,col,row,channel)
    :param mask: a boolean array of either (1,col,row,channel) or (num_imgs,col,row,channel).
    :return: Img array with the same dims as org_img
    """
    if isinstance(mask, tuple):
        org_copy=org_img.copy()
        for i in range(gen_img.shape[0]):
            org_copy[i,mask[0]:mask[1],mask[2]:mask[3]]=gen_img[i,mask[0]:mask[1],mask[2]:mask[3]]
        return org_copy        
    org_copy=org_img.copy()
    gen_copy=gen_img.copy()
    mask_copy=mask.copy()
    #only 2 dims adding and expending dims
    if len(org_copy.shape)==3:
        org_copy=np.expand_dims(org_copy, 0)
   
    #check if 3 dims and fist dim is 1
    if mask_copy.shape[0]==1:
        mask_copy=np.repeat(mask_copy, org_copy.shape[0], axis=0)
    
    #only 2 dims adding and expending dims
    if len(mask_copy.shape)==2:
        mask_copy=np.expand_dims(mask_copy, 0)
        mask_copy=np.repeat(mask_copy, org_copy.shape[0], axis=0)
    

    #adding color dim
    mask_copy=np.expand_dims(mask_copy,-1)
    mask_copy=np.repeat(mask_copy, 3, axis=-1)
    
            
    for i, img in enumerate(org_copy):
        mask_copy[i] = np.multiply(mask_copy[i],gen_copy[i])
        org_copy[i]  = np.multiply(np.logical_not(mask_copy[i]),org_copy[i])
        org_copy[i]  = np.add(mask_copy[i],org_copy[i])
    return org_copy        

if __name__=='__main__':    
    import plotload as pl
    import matplotlib.pyplot as plt  
    img=pl.load_polyp_batch((576,720,3),5,data_type="green",rot=False)
    #a,b,c=mask_randomly_square(img, 20, 100)
    a,b,c=mask_green_corner(img)
    plt.imshow(a[0]*0.5+0.5)
    plt.show()
    plt.imshow(b[0]*0.5+0.5)
    plt.show()
    #mask,missing,template=mask_from_template(img,rot=False)
    #plt.imshow(0.5*missing[0]+0.5)
    #plt.show()
    #plt.imshow(0.5*missing[0]+0.5)
    #plt.show()
    #img2=img[::-1,:,:,:]
    reconst=combine_imgs_with_mask(img2,img,template)
    #plt.imshow(0.5*reconst[0]+0.5)
    #plt.show()
