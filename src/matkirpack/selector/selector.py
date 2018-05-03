import pygame, sys 
from PIL import Image 
"""
Code inspired from:
https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558
"""
def displayImage(screen, px, prior, box=(220,180)):
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

def displayImage1(screen, px, topleft, prior):
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
    
def setup(path,scale):
    px = pygame.image.load(path)
    #px = pygame.transform.scale(px, list(map(lambda x:int(int(x)*scale),list(px.get_rect())[2:])))

    screen = pygame.display.set_mode(list(px.get_rect())[2:])
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def mainLoop(screen, px):
    box=(280,208)
    startpos = endpos = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not startpos:
                    startpos = 1
                else:
                    endpos = event.pos
                    n=1
        if startpos:
            prior = displayImage(screen, px, prior, box)
    #return ( topleft + bottomright )
    return (endpos[0]-box[0]//2,endpos[0]+box[0]//2,endpos[1]-box[1]//2,endpos[1]+box[1]//2)

def get_coord(adr='2.jpg',scale=1):
    if type(adr)!=str:
        input_loc = '2.jpg'
    else:
        input_loc=adr

    pygame.init()
    screen, px = setup(input_loc,scale)
    left, upper, right, lower = mainLoop(screen, px)
    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    pygame.display.quit()
    return int(left*scale),int(upper*scale),int(right*scale),int(lower*scale)
    

if __name__ == "__main__":
    print(get_coord('3.jpg',1))