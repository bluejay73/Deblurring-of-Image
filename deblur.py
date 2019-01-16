import cv2
import numpy as np

def motionImgKernel(ang,dia,size=65):
    kernel=np.ones((1,dia),np.float32)
    cos=np.cos(ang)
    sin=np.sin(ang)
    x=np.float32([[cos,-sin,0],[sin,cos,0]])
    size2=size//2
    x[:,2]=(size2,size2)-np.dot(x[:,:2], ((dia-1)*0.5,0))
    kernel=cv2.warpAffine(kernel,x,(size,size), flags=cv2.INTER_CUBIC)
    return kernel
    
    
def defocusImgKernel(dia,size=65):
    kernel=np.zeros((size,size),np.uint8)
    cv2.circle(kernel,(size,size),dia,255,-1,cv2.LINE_AA,shift=1)
    kernel=np.float32(kernel)/255.0
    return kernel
    
    
def blur_border(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)
