import os
import numpy as np
import pandas as pd
import cv2
import imutils
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt        

ELASTIC_ON = False
MIRROR_PADDING_ON = False
ROTATION_ON = False
FLIP_ON = True
img_path = 'data/images/'
mask_path = 'data/labels/'
img_shotname = ''
mask_shotname = ''
def mirror_padding (im,im_mask):
    # !padding 
    reflect = cv2.copyMakeBorder(im,363,363,363,363,cv2.BORDER_REFLECT) 
    reflect2 = cv2.copyMakeBorder(im_mask,363,363,363,363,cv2.BORDER_REFLECT) 
    cv2.imwrite(os.path.join('img_pad/', img_shotname + '-' + str('pad') + str('.jpg')), reflect) 
    cv2.imwrite(os.path.join('lab_pad/', mask_shotname + '-' + str('pad') + str('.jpg')), reflect2)     

def rotation (im,im_mask):
    for angle in np.arange(30, 330, 30):
        rotated_im = imutils.rotate(im, angle)
        crop_im = rotated_im[363:875, 363:875]
        rotated_immask = imutils.rotate(im_mask, angle)
        crop_im2 = rotated_immask[363:875, 363:875]
        cv2.imwrite(os.path.join(img_path, img_shotname + '-' + str(angle) + str('.jpg')), crop_im) 
        cv2.imwrite(os.path.join(mask_path, mask_shotname + '-' + str(angle) + str('.jpg')), crop_im2)       

def flip (im,im_mask):
    horizontal_img = cv2.flip( im, 0 )
    vertical_img = cv2.flip( im, 1 )
    both_img = cv2.flip( im, -1 )
    cv2.imwrite(os.path.join(img_path, img_shotname + '-' + str('hor') + str('.jpg')), horizontal_img) 
    cv2.imwrite(os.path.join(img_path, img_shotname + '-' + str('ver') + str('.jpg')), vertical_img) 
    cv2.imwrite(os.path.join(img_path, img_shotname + '-' + str('both') + str('.jpg')), both_img)     
    horizontal_img2 = cv2.flip( im_mask, 0 )
    vertical_img2 = cv2.flip( im_mask, 1 )
    both_img2 = cv2.flip( im_mask, -1 )
    cv2.imwrite(os.path.join(mask_path, mask_shotname + '-' + str('hor') + str('.jpg')), horizontal_img2) 
    cv2.imwrite(os.path.join(mask_path, mask_shotname + '-' + str('ver') + str('.jpg')), vertical_img2) 
    cv2.imwrite(os.path.join(mask_path, mask_shotname + '-' + str('both') + str('.jpg')), both_img2)     

def elastic_transform(image, alp, sig, aa, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.                
     Reference on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]          
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    first_p = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    second_p = first_p + random_state.uniform(-aa, aa, size=first_p.shape).astype(np.float32)

    matrix = cv2.getAffineTransform(first_p, second_p)  
  
    image = cv2.warpAffine(image, matrix, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sig) * alp
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sig) * alp
    dz = np.zeros_like(dx)                

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))             
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)                

if __name__ == '__main__':

    img_list = sorted(os.listdir(img_path))
    mask_list = sorted(os.listdir(mask_path))

    count_total = 0
    for i in range(30):
        im = cv2.imread(os.path.join(img_path, img_list[i]), -1)
        im_mask = cv2.imread(os.path.join(mask_path, mask_list[i]), -1)                           
        # Merge images into separete channels (shape will be (cols, rols, 2))
        im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)                
        # get img and mask shortname
        (i_shotname, img_tale) = os.path.splitext(img_list[i])  #将文件名和扩展名分开
        (m_shotname, mask_tale) = os.path.splitext(mask_list[i])    
        img_shotname = i_shotname
        mask_shotname = m_shotname

        if MIRROR_PADDING_ON :
            mirror_padding(im,im_mask)
        if ROTATION_ON :
            rotation(im,im_mask)
        if FLIP_ON :
            flip(im,im_mask)
        if ELASTIC_ON :
            # !Elastic deformation 11 times
            count = 0                
            while count < 11:
        
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                            im_merge.shape[1] * 0.08)

                # Split image and mask
                im_t = im_merge_t[..., 0]
                im_mask_t = im_merge_t[..., 1]                
                # save the new imgs and masks
                cv2.imwrite(os.path.join(img_path, img_shotname + '-' + str(count) + str('.jpg')), im_t)
                cv2.imwrite(os.path.join(mask_path, mask_shotname + '-' + str(count) + str('.jpg')), im_mask_t)                
                count += 1
                count_total += 1
