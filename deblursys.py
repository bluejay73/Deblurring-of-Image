
import numpy as np
import cv2
import sys,getopt
import deblur
import skimage

opts, args = getopt.getopt(sys.argv[1:], '', ['def', 'ang=', 'd=', 'noise='])
opts = dict(opts)
 
try:
    myim=args[0]
except:
    print "As you didn't gave any image so using default one"
    myim='gray1.png'
             
win='deblurred'
img_bw=cv2.imread(myim,0)
img_rgb=cv2.imread(myim,1)
    
if img_bw is None and img_rgb is None:
    print('Error 404-Image Not Found!\n try again', myim)
    sys.exit(1)
    
img_r = np.zeros_like(img_bw)
img_g = np.zeros_like(img_bw)
img_b = np.zeros_like(img_bw) 
img_r = img_rgb[..., 0]
img_g = img_rgb[..., 1]
img_b = img_rgb[..., 2]

img_rgb = np.float32(img_rgb)/255.0
img_bw = np.float32(img_bw)/255.0
img_r = np.float32(img_r)/255.0
img_g = np.float32(img_g)/255.0
img_b = np.float32(img_b)/255.0
    
cv2.imshow('Original_Image', img_rgb)
    
img_r = deblur.blur_border(img_r)
img_g = deblur.blur_border(img_g)
img_b = deblur.blur_border(img_b)
    
    
IMG_R = cv2.dft(img_r, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    
IMG_G = cv2.dft(img_g, flags=cv2.DFT_COMPLEX_OUTPUT)
    
IMG_B = cv2.dft(img_b, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    
defocus = '--def' in opts
    
def update(_):
    ang = np.deg2rad( cv2.getTrackbarPos('angle', win) )
    dia = cv2.getTrackbarPos('d', win)
    noise = 10**(-0.1*cv2.getTrackbarPos('Noise', win))

    if defocus:
        psf = deblur.defocusImgKernel(dia)
    else:
        psf = deblur.motionImgKernel(ang, dia)
    cv2.imshow('psf', psf)

    psf /= psf.sum()
    psf_pad = np.zeros_like(img_bw)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    PSF = cv2.dft(psf_pad,flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
    PSF2 = (PSF**2).sum(-1)
    iPSF = PSF / (PSF2 + noise)[...,np.newaxis]

                                                                                                                                                                           
    RES_R = cv2.mulSpectrums(IMG_R, iPSF, 0)
    RES_G = cv2.mulSpectrums(IMG_G, iPSF, 0)
    RES_B = cv2.mulSpectrums(IMG_B, iPSF, 0)


        
    res_r = cv2.idft(RES_R, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
    res_g = cv2.idft(RES_G, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
    res_b = cv2.idft(RES_B, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )

    res_rgb = np.zeros_like(img_rgb)
    res_rgb[..., 0] = res_r
    res_rgb[..., 1] = res_g
    res_rgb[..., 2] = res_b

    res_rgb = np.roll(res_rgb, -kh//2, 0)
    res_rgb = np.roll(res_rgb, -kw//2, 1)
    cv2.imshow('Result', res_rgb)
    return res_rgb
cv2.namedWindow(win)
cv2.namedWindow('psf', 0)
cv2.createTrackbar('angle', win, int(opts.get('--angle', 135)), 180, update)
cv2.createTrackbar('d', win, int(opts.get('--d', 22)), 50, update)
cv2.createTrackbar('Noise', win, int(opts.get('--snr', 25)), 50, update)
update(None)


#cv2.imwrite('jay.png',jay)
print "Enter Q/q to Quit on any Image window when you get the enhanced Image"
print "Enter c/C to change the Kernel on any Image window"
while True:                       
    ch = cv2.waitKey() & 0xFF
    if ch == 81 or ch==113:
        jay=update(None)
        cv2.imwrite('newim.tif',jay);
        break
        
            
        
            
    if ch == 67 or ch==99:
        defocus = not defocus
        update(None)
        

    
    
    
    
