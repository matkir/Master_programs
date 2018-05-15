import scipy.misc
import plotload as pl
import numpy as np
import pygame, sys 
from PIL import Image 
"""
Code inspired from:
https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558
"""

def _gui_displayImage(screen, px,brush,mask_array,startmouse,prior):
     # ensure that the rect always has positive width, height
    width =  pygame.mouse.get_pos()[0] 
    height = pygame.mouse.get_pos()[1]
    done=True
    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = width, height
    if not (width and height):
        return current,mask_array,done
    if current == prior:
        return current,mask_array,done
    
    
    if brush==1:
        """
        Using Pen
        """

        box=(20,20)
        # draw transparent box and blit it onto canvas
        screen.blit(px, px.get_rect())
        im = pygame.Surface(box)
        im.fill((128, 128, 128))
        pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
        im.set_alpha(128)
        screen.blit(im, (width-box[0]//2, height-box[1]//2))
        mask_array[width-box[0]//2:width+box[0]//2, height-box[1]//2:height+box[1]//2]=1        
        
      
    
    
    if brush==2:
        """
        Using dragbox
        """
        x,y=startmouse
        
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)   

        screen.blit(px, px.get_rect())
        im = pygame.Surface((width-x, height-y))
        im.fill((128, 128, 128))
        pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
        im.set_alpha(128)
        screen.blit(im, (x, y))
        #pygame.display.flip() 
        done=2
        #drawing the mask on
        disp_mask=mask_array.copy()
        surf = pygame.surfarray.make_surface((disp_mask*255))
        surf.set_alpha(50)
        screen.blit(surf, (0, 0))       
        pygame.display.flip()        
        return (width,height),(x,width,y,height),done 
        #mask_array[x:width,y:height]=1
        
        
    if brush==3:
        box=(220,180)
        # draw transparent box and blit it onto canvas
        screen.blit(px, px.get_rect())
        im = pygame.Surface(box)
        im.fill((128, 128, 128))
        pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
        im.set_alpha(128)
        screen.blit(im, (width-box[0]//2, height-box[1]//2))
        #mask_array[width-box[0]//2:width+box[0]//2, height-box[1]//2:height+box[1]//2]=1
        done=3
        #drawing the mask on
        disp_mask=mask_array.copy()
        surf = pygame.surfarray.make_surface((disp_mask*255))
        surf.set_alpha(50)
        screen.blit(surf, (0, 0))   
        pygame.display.flip()        
        return (width,height),(width,height,box),done 
    
    #drawing the mask on
    disp_mask=mask_array.copy()
    surf = pygame.surfarray.make_surface((disp_mask*255))
    surf.set_alpha(50)
    screen.blit(surf, (0, 0))   

    pygame.display.flip()        
    return (width,height),mask_array,done    



def _displayImage(screen, px, prior, box=(220,180)):
    # ensure that the rect always has positive width, height
    width =  pygame.mouse.get_pos()[0] 
    height = pygame.mouse.get_pos()[1]
   
    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface(box)
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (width-box[0]//2, height-box[1]//2))
    pygame.display.flip()


    # return current box extents
    return ('a', 'b', width, height)    

def _displayImage1(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width =  pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()


    # return current box extents
    return (x, y, width, height)    
    
def _gui_setup(path,scale,buttons):
    button0,button1,button2,button3=buttons
    px = pygame.image.load(path)
    #px = pygame.transform.scale(px, list(map(lambda x:int(int(x)*scale),list(px.get_rect())[2:])))

    #makes space to the buttons
    screen_size=list(px.get_rect())[2:]
    screen_size[0]=screen_size[0]+100 
    screen = pygame.display.set_mode(screen_size)

 
    screen.blit(px, px.get_rect())
    pygame.draw.rect(screen, [100, 200, 100], button0)
    pygame.draw.rect(screen, [200, 200, 200], button1)
    pygame.draw.rect(screen, [100, 100, 200], button2)
    pygame.draw.rect(screen, [200, 200, 100], button3)
    pygame.display.flip()
    return screen, px



def _gui_mainLoop(screen, px,buttons,img_shape=(720,576)):
    button0,button1,button2,button3=buttons
    going=True
    startpos = endpos = prior = None
    n=0
    brushes={"pen":1,"rightSizeBox":2,"dragBox":3}
    brush=1
    mask_array=np.zeros(shape=img_shape[:2])
    done=True
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                going = False
                n=1
                return mask_array,going
            if event.type == pygame.MOUSEBUTTONUP:
                """
                Looking at the 4 buttons
                """
                mouse_pos = event.pos
                if mouse_pos[0]>px.get_rect()[2]:
                    if button0.collidepoint(mouse_pos):
                        brush=brushes["pen"]
                        continue
                    if button1.collidepoint(mouse_pos):
                        brush=brushes["rightSizeBox"]
                        continue
                    if button2.collidepoint(mouse_pos):
                        brush=brushes["dragBox"]
                        continue
                    if button3.collidepoint(mouse_pos):
                        going=False
                        return mask_array,going
                else:
                    if done==2:
                        mask_array[tmp_mask[0]:tmp_mask[1],tmp_mask[2]:tmp_mask[3]]=1
                        done=True
                    elif done==3: 
                        mask_array[tmp_mask[0]-tmp_mask[2][0]//2:tmp_mask[0]+tmp_mask[2][0]//2, tmp_mask[1]-tmp_mask[2][1]//2:tmp_mask[1]+tmp_mask[2][1]//2]=1
                        done=True  
                    disp_mask=mask_array.copy()
                    surf = pygame.surfarray.make_surface((disp_mask*255))
                    surf.set_alpha(50)
                    screen.blit(surf, (0, 0))   
                    pygame.display.flip()                        
                startpos=None
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos= event.pos
                if not startpos:
                    startpos = event.pos
            
            if pygame.mouse.get_pressed()[0]:
                mouse_pos= event.pos
                if mouse_pos[0]<px.get_rect()[2]:
                    #drawing
                    prior,tmp_mask,done = _gui_displayImage(screen, px, brush,mask_array,startpos,prior)
                    if done==True:
                        mask_array=tmp_mask
                else:
                    startpos=None
                    
    #return ( topleft + bottomright )
    return mask_array,going
def _setup(path,scale):
    px = pygame.image.load(path)
    #px = pygame.transform.scale(px, list(map(lambda x:int(int(x)*scale),list(px.get_rect())[2:])))

    screen = pygame.display.set_mode(list(px.get_rect())[2:])
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px
def _mainLoop(screen, px, box=(280,208)):
    going=True
    startpos = endpos = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                going = False
                endpos = (0,0,0,0)
                n=1
            if event.type == pygame.MOUSEBUTTONUP:
                if not startpos:
                    startpos = 1
                else:
                    endpos = event.pos
                    n=1
        if startpos:
            prior = _displayImage(screen, px, prior, box)
    #return ( topleft + bottomright )
    return (endpos[0]-box[0]//2,endpos[0]+box[0]//2,endpos[1]-box[1]//2,endpos[1]+box[1]//2,going)    

    
def get_coord(adr='2.jpg',scale=1,mask_width=0,mask_height=0):
    """
    Only used for contextencoder/filler.py
    """
    if type(adr)!=str:
        input_loc = '2.jpg'
    else:
        input_loc=adr

    pygame.init()
    screen, px = _setup(input_loc,scale)
    left, upper, right, lower, _ = _mainLoop(screen, px,(mask_width,mask_height))
    # ensure output rect always has positive width, height
    """
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    """
    pygame.display.quit()
    return int(left*scale),int(upper*scale),int(right*scale),int(lower*scale)


def get_coord_live(adr='2.jpg',scale=1,img_rows=576,img_cols=720,mask_height=208,mask_width=280):
    """
    Only used for contextencoder/live.py
    """
    channels = 3
    img_shape = (img_rows, img_cols, channels)
    missing_shape = (mask_width, mask_height, channels)
    from keras.models import load_model

    generator = load_model("saved_model/generator.h5")
    generator.load_weights("saved_model/generator_weigths.h5")
    
    import scipy.misc
    import plotload as pl
    import numpy as np

    img,img_path=pl.load_one_img(img_shape, dest=adr)
    scipy.misc.toimage(img, cmin=-1, cmax=1).save('tmp.png')            
    pygame.init()
    global going 
    going=True
    while going:
        img,img_path=pl.load_one_img(img_shape, dest='tmp.png')
        screen, px = _setup('tmp.png',1)
        left, upper, right, lower, going = _mainLoop(screen, px, (mask_width, mask_height))
        if not going:   
            scipy.misc.toimage(masked_img, cmin=-1, cmax=1).save('output.png')
            break
        (x1, x2, y1, y2)=(int(left*scale),int(upper*scale),int(right*scale),int(lower*scale))
        masked_img = np.empty_like(img)
        missing_parts = np.ndarray(shape=(x2-x1,y2-y1))
        masked_img = img.copy()
        missing_parts =masked_img[y1:y2, x1:x2,:].copy()
        masked_img[y1:y2, x1:x2,:] = -1
        guess=np.squeeze(generator.predict(np.expand_dims(masked_img,axis=0)),axis=0)  
        masked_img[y1:y2, x1:x2,:] = guess

        scipy.misc.toimage(masked_img, cmin=-1, cmax=1).save('tmp.png')  
        
    pygame.display.quit()

def gui(adr='2.jpg',scale=1,img_rows=720,img_cols=576,mask_height=208,mask_width=280):
    channels = 3
    img_shape = (img_rows, img_cols, channels)

    #from keras.models import load_model
    #generator = load_model("saved_model/generator.h5")
    #generator.load_weights("saved_model/generator_weigths.h5")
    

    img,img_path=pl.load_one_img(img_shape, dest=adr)
    scipy.misc.toimage(img, cmin=-1, cmax=1).save('tmp.png')            
    
    #starting window, and adding buttons
    pygame.init()
    button0=pygame.Rect(img_shape[0]+25,0 ,50,50)    
    button1=pygame.Rect(img_shape[0]+25,100,50,50)    
    button2=pygame.Rect(img_shape[0]+25,200,50,50)    
    button3=pygame.Rect(img_shape[0]+25,300,50,50)
    buttons=[button0,button1,button2,button3]
    global going 
    going=True
    while going:
        img,img_path=pl.load_one_img(img_shape, dest='tmp.png')
        screen, px = _gui_setup('tmp.png',1,buttons)            
        mask_array, going = _gui_mainLoop(screen, px,buttons,img_shape=img_shape)
        print(mask_array)
        scipy.misc.toimage(mask_array, cmin=-1, cmax=1).save('tmp.png') 
    pygame.quit()            
            
            
if __name__=='__main__':
    gui()
    