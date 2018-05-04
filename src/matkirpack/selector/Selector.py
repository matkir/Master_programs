import pygame, sys 
from PIL import Image 
"""
Code inspired from:
https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558
"""


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
    
def _setup(path,scale):
    px = pygame.image.load(path)
    #px = pygame.transform.scale(px, list(map(lambda x:int(int(x)*scale),list(px.get_rect())[2:])))

    screen = pygame.display.set_mode(list(px.get_rect())[2:])
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def _mainLoop(screen, px):
    going=True
    box=(280,208)
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

    
def get_coord(adr='2.jpg',scale=1):
    #################
    ###starts here###
    #################
    if type(adr)!=str:
        input_loc = '2.jpg'
    else:
        input_loc=adr

    pygame.init()
    screen, px = _setup(input_loc,scale)
    left, upper, right, lower, _ = _mainLoop(screen, px)
    # ensure output rect always has positive width, height
    """
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    """
    pygame.display.quit()
    return int(left*scale),int(upper*scale),int(right*scale),int(lower*scale)


def get_coord_live(adr='2.jpg',scale=1):
    img_rows = 576#8*64//2#32
    img_cols = 720#8*64//2#32
    mask_height = 208#300 #self.img_cols//4#8*16//2#8
    mask_width = 280#350 #self.img_rows//4 #8*16//2#8
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
        left, upper, right, lower, going = _mainLoop(screen, px)
        if not going:   
            scipy.misc.toimage(masked_img, cmin=-1, cmax=1).save('output.png')
            break
        (x1, x2, y1, y2)=(int(left*scale),int(upper*scale),int(right*scale),int(lower*scale))
        masked_img = np.empty_like(img)
        missing_parts = np.ndarray(shape=(x2-x1,y2-y1))
        masked_img = img.copy()
        missing_parts =masked_img[y1:y2, x1:x2,:].copy()
        masked_img[y1:y2, x1:x2,:] = 0
        guess=np.squeeze(generator.predict(np.expand_dims(masked_img,axis=0)),axis=0)  
        masked_img[y1:y2, x1:x2,:] = guess

        scipy.misc.toimage(masked_img, cmin=-1, cmax=1).save('tmp.png')  
        
    pygame.display.quit()


if __name__=='__main__':
    print(get_coord_live())
    