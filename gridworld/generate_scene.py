import numpy as np

imagex=20
imagey=20

def print_image(image) :
    for row in image : 
        print(row)
        
def square(image,side,xpos,ypos,label=1):
    if not side%2 :
        print("side must have odd length")
        side+=1
    for i in range(-int(side/2),int(side/2)+1):
        for j in range(-int(side/2),int(side/2)+1):
            if xpos+i > 0 and xpos+i < imagex and ypos+j > 0 and ypos+j < imagey :
                image[xpos+i,ypos+j] = 1
        
    return image

## create blank image
image = np.zeros((imagex,imagey))
## put square at truth-level position
square(image,5,3,3)

