import numpy as np
import cv2
from matplotlib import pyplot as plt

def generateGaussianMask(width,height,sigma):
    # create a mask first, center square is 1, remaining all zeros
    crow,ccol = rows//2 , cols//2
    mask = np.zeros((height,width,2),np.float32)
    c1=2*np.pi*sigma*sigma
    c2=2*sigma*sigma
    for x in range(cols):
        for y in range(rows):
            t_x=x-ccol
            t_y=y-crow
            v=np.exp((-t_x**2-t_y**2)/c2)/c1
            mask[x,y]=(v,v)
    return mask

def generateSharpenMask(width,height,sigma):
    # create a mask first, center square is 1, remaining all zeros
    crow,ccol = rows//2 , cols//2
    mask = np.zeros((height,width,2),np.float32)
    c1=2*np.pi*sigma*sigma
    c2=2*sigma*sigma
    for x in range(cols):
        for y in range(rows):
            t_x=x-ccol
            t_y=y-crow
            v=2-np.exp((-t_x**2-t_y**2)/c2)/c1
            mask[x,y]=(v,v)
    return mask

def applyFilter(dft_shift,mask):
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back

def generateMask(width,height,radius):
    # create a mask first, center square is 1, remaining all zeros
    crow,ccol = rows//2 , cols//2
    mask = np.zeros((height,width,2),np.float32)
    for x in range(cols):
        for y in range(rows):
            if (x-crow)**2+(y-crow)**2<=radius*radius:
                mask[x,y]=(1,1)
            else:
                mask[x,y]=(0,0)
    return mask

def dft2d(img):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return dft_shift,magnitude_spectrum


img = cv2.imread('fractal.png',0)
dft_shift,magnitude_spectrum=dft2d(img)
rows, cols = img.shape
mask1=generateGaussianMask(cols,rows,5)
mask2=generateSharpenMask(cols,rows,8)
mask3=generateMask(cols,rows,100)
mask1_magnitude= (cv2.magnitude(mask1[:,:,0],mask1[:,:,1]))
mask2_magnitude= (cv2.magnitude(mask2[:,:,0],mask2[:,:,1]))
img_back1=applyFilter(dft_shift,mask1)
img_back2=applyFilter(dft_shift,mask2)
img_back3=applyFilter(dft_shift,mask3)
plt.subplot(241),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(242),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(243),plt.imshow(img_back1, cmap = 'gray')
plt.title('Filtered Image1'), plt.xticks([]), plt.yticks([])
plt.subplot(244),plt.imshow(img_back2, cmap = 'gray')
plt.title('Filtered Image2'), plt.xticks([]), plt.yticks([])
plt.subplot(245),plt.imshow(img_back3, cmap = 'gray')
plt.title('Filtered Image3'), plt.xticks([]), plt.yticks([])
plt.subplot(246),plt.imshow(mask1_magnitude, cmap = 'gray')
plt.title('Mask Image1'), plt.xticks([]), plt.yticks([])
plt.subplot(247),plt.imshow(mask2_magnitude, cmap = 'gray')
plt.title('Mask Image2'), plt.xticks([]), plt.yticks([])
plt.show()