import numpy as np
import cv2
import tifffile
import glob

image = "result/predict/*.png"
label = "result/label/*.png"
image = glob.glob(image)
image.sort()
image.sort(key = len)

label = glob.glob(label)
label.sort()
label.sort(key = len)

img = [cv2.imread(file,0) for file in image]
lb = [cv2.imread(file,0) for file in label]

img = np.float32(img)/255
lb = np.float32(lb)/255
output = img[0].reshape(1,512,512)

for im in img[1:]:

	output = np.concatenate((output,im.reshape(1,512,512)),axis = 0)


tifffile.imsave("predict.tif",output)


output = lb[0].reshape(1,512,512)

for im in lb[1:]:
	output = np.concatenate((output,im.reshape(1,512,512)),axis = 0)

tifffile.imsave("label.tif",output)
