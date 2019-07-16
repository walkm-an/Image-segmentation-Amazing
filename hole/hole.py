import numpy as np 
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
# read image
# img = cv2.imread('myGray.jpg',0)

# 1. read image
img = cv2.imread('test3.jpg', 0)

# 2. de-noising

# 2.1 gaussian blur
# img_denoising_g = cv2.GaussianBlur(img,(9,9),2)

# 2.2 median blur
img_denoising_m = cv2.medianBlur(img, 3)

# 2.3 fast denoising method
# img_denoising_f = cv2.fastNlMeansDenoising(img)

# de-noising test
# plt.subplot(1,3,1),plt.imshow(img_denoising_g,'gray'),plt.title('gaussian')
# plt.subplot(1,3,2),plt.imshow(img_denoising_m,'gray'),plt.title('medain')
# plt.subplot(1,3,3),plt.imshow(img_denoising_f,'gray'),plt.title('fast')

# plt.show()

# 3. contrast enhancement 
# Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25,25))
cl1 = clahe.apply(img_denoising_m)

# test
# plt.imshow(cl1,'gray')
# plt.show()

# median filter / Gausain / 

# otsu_th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
             # cv2.THRESH_BINARY,121,3)

# ret3,adap_th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 4. Thresholding 
# global thresholding 
ret,th1 = cv2.threshold(cl1,110,255,cv2.THRESH_BINARY)

# test
# plt.imshow(th1,'gray')
# plt.show()

# 5. Hole filling
closing = ndimage.binary_fill_holes(th1).astype(int)

# 6. Smoothing 

cv2.imwrite('closing.jpg',closing)
closing = cv2.imread('closing.jpg', 0)
img_smooth = cv2.medianBlur(closing, 5)

# 7. Thinning
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(img_smooth,kernel,iterations = 1)

# plt.imshow(dilation,'gray')
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(dilation,kernel,iterations = 1)


# results Comparision
groundTruth = cv2.imread('train-labels03.jpg', 0)

plt.subplot(1,3,1),plt.imshow(img,'gray'),plt.title('original')
plt.subplot(1,3,2),plt.imshow(erosion,'gray'),plt.title('my_result')
plt.subplot(1,3,3),plt.imshow(groundTruth,'gray'),plt.title('groundTruth')

plt.show()

cv2.imwrite('out_img.jpg',erosion*255)
# ret,th1 = cv2.threshold(img_m,135,255,cv2.THRESH_BINARY)
# plt.subplot(2,5,1),plt.imshow(img,'gray'),plt.title('original')
# plt.subplot(2,5,2),plt.imshow(img_m,'gray'),plt.title('Median')
# plt.subplot(2,5,3),plt.imshow(cl1,'gray'),plt.title('clahe')
# plt.subplot(2,5,4),plt.imshow(otsu_th,'gray'),plt.title('otsu')
# plt.subplot(2,5,5),plt.imshow(th1,'gray'),plt.title('global')





