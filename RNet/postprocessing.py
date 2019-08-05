import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import slic, join_segmentations
from skimage.morphology import watershed
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import cv2
import os



def sobel_watershed (image):
    edges = sobel(image)
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < 90.0] = background
    markers[image > 180.0] = foreground
    ws = watershed(edges, markers)
    seg1 = label(ws == foreground)
    return seg1

def slic_super (image):
    seg2 = slic(image, n_segments=150, max_iter=180, sigma=5, compactness=0.1,multichannel=False)
    return seg2

def morphological (output1):
    kernal = np.ones((2,2), np.uint8)
    dilation = cv2.dilate(output1, kernal, iterations=1)
    imsave ('crf_erosion_output01.jpg',dilation*255)
    kernal3 = np.ones((3,3), np.uint8)
    erosion = cv2.erode(output1, kernal3, iterations=1)
    imsave ('crf_erosion_output01.jpg',erosion*255)
    kernal4 = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(output1, cv2.MORPH_OPEN, kernal4, iterations=1)
    kernal2 = np.ones((2,2), np.uint8)
    opdi = cv2.dilate(opening, kernal2, iterations=1)
    imsave ('crf_open_dilation_output01.jpg',opdi*255)

def crf(original_image, annotated_image,output_image):
    """

    Reference on https://github.com/Gurupradeep/FCN-for-Semantic-Segmentation/blob/master/CRF.ipynb
    """

    if(len(original_image.shape)<3):
        original_image = gray2rgb(original_image)
    if(len(annotated_image.shape)<3):
        annotated_image = gray2rgb(annotated_image)


        
    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,0] + (annotated_image[:,:,1]<<8) + (annotated_image[:,:,2]<<16)
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    
    #Setting up the CRF model

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,compat=5,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)


    #Run Inference for 5 steps 
    Q = d.inference(3)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    imsave(output_image,MAP.reshape(original_image.shape))
    return MAP.reshape(original_image.shape)



if __name__ == '__main__':

    image = imread('data/images/train-volume01.jpg')
    seg1 = sobel_watershed (image)
    seg2 = slic_super(image)
    # Combine the two.
    segj = join_segmentations(seg1, seg2)

    color1 = label2rgb(seg1, image=image, bg_label=0)
    imsave("water01.jpg",color1)
    color2 = label2rgb(seg2, image=image, image_alpha=0.5)
    imsave("SLIC superpixels01.jpg",color2)
    color3 = label2rgb(segj, image=image, image_alpha=0.5)
    imsave("slic_water01.jpg",color3)

    img_pre = imread('water01.jpg')
    label = imread('data/labels/train-labels01.jpg',0)
    pred = imread('poseprocessing/Yao/predict/output1.jpg',0)
    output1 = crf(img_pre,pred,'crf01.jpg')
    output1 = rgb2gray(output1)
    morphological(output1)

