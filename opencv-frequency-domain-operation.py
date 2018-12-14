import numpy as np
import cv2
from matplotlib import pyplot as plt

def generateGaussianMask(width,height,sigma):
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
    crow,ccol = rows//2 , cols//2
    mask = np.zeros((height,width,2),np.float32)
    c1=2*np.pi*sigma*sigma
    c2=2*sigma*sigma
    for x in range(cols):
        for y in range(rows):
            t_x=x-ccol
            t_y=y-crow
            v=1-10*np.exp((-t_x**2-t_y**2)/c2)/c1
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
            if (x-crow)**2+(y-ccol)**2<=radius*radius:
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

mask_r5=generateMask(cols,rows,5)
mask_r50=generateMask(cols,rows,50)
mask_r100=generateMask(cols,rows,100)
mask_blur=generateGaussianMask(cols,rows,10)
mask_sharpen=generateSharpenMask(cols,rows,50)

mask_r5_magnitude= (cv2.magnitude(mask_r5[:,:,0],mask_r5[:,:,1]))
mask_r50_magnitude= (cv2.magnitude(mask_r50[:,:,0],mask_r50[:,:,1]))
mask_r100_magnitude= (cv2.magnitude(mask_r100[:,:,0],mask_r100[:,:,1]))
mask_blur_magnitude= (cv2.magnitude(mask_blur[:,:,0],mask_blur[:,:,1]))
mask_sharpen_magnitude= (cv2.magnitude(mask_sharpen[:,:,0],mask_sharpen[:,:,1]))

img_r5=applyFilter(dft_shift,mask_r5)
img_r50=applyFilter(dft_shift,mask_r50)
img_r100=applyFilter(dft_shift,mask_r100)
img_blur=applyFilter(dft_shift,mask_blur)
img_sharpen=applyFilter(dft_shift,mask_sharpen)

plt.subplot(241),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(242),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(243),plt.imshow(img_r5, cmap = 'gray')
plt.title('ILPF R=5 Filtered Image'), plt.xticks([]), plt.yticks([])
plt.subplot(244),plt.imshow(img_r50, cmap = 'gray')
plt.title('ILPF R=50 Filtered Image'), plt.xticks([]), plt.yticks([])
plt.subplot(245),plt.imshow(img_r100, cmap = 'gray')
plt.title('ILPF R=100 Filtered Image'), plt.xticks([]), plt.yticks([])

plt.subplot(246),plt.imshow(mask_r5_magnitude, cmap = 'gray')
plt.title('ILPF R=5'), plt.xticks([]), plt.yticks([])
plt.subplot(247),plt.imshow(mask_r50_magnitude, cmap = 'gray')
plt.title('ILPF R=50'), plt.xticks([]), plt.yticks([])
plt.subplot(248),plt.imshow(mask_r100_magnitude, cmap = 'gray')
plt.title('ILPF R=100'), plt.xticks([]), plt.yticks([])

plt.figure()
plt.subplot(241),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(242),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(243),plt.imshow(mask_blur_magnitude, cmap = 'gray')
plt.title('GLPF Sigma=10'), plt.xticks([]), plt.yticks([])
plt.subplot(244),plt.imshow(img_blur, cmap = 'gray')
plt.title('GLPF Sigma=10 Filtered Image'), plt.xticks([]), plt.yticks([])
plt.subplot(245),plt.imshow(mask_sharpen_magnitude, cmap = 'gray')
plt.title('Sharpen mask'), plt.xticks([]), plt.yticks([])
plt.subplot(246),plt.imshow(img_sharpen, cmap = 'gray')
plt.title('Sharpened Image'), plt.xticks([]), plt.yticks([])
plt.show()