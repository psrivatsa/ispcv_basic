# Computer Vision using CV2 in Python
# Read and display RGB color image
# https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html

# import libraries
import sys
import cv2 as cv
import numpy as np

# print Python version
print("Python Version : " + sys.version)
# Check CV2 version
print("CV2 version : " + cv.__version__)

# read image
img = cv.imread("../images/smarties.png",1)

# check if file read is successful
if img is None:
    sys.exit("Could not read the image.")

# Display image
cv.imshow("BGR Color image", img)

# Display B,G,R Channels seperately with fake colors
channel_initials = list('BGR')

for channel_index in range(3):
    channel = np.zeros(shape=img.shape, dtype=np.uint8)
    channel[:,:,channel_index] = img[:,:,channel_index]
    cv.imshow(f'{channel_initials[channel_index]}-RGB fake colors', channel)

# Display B,G,R Channels seperately as Grayscale
b_gray = img[:,:,0];
g_gray = img[:,:,1];
r_gray = img[:,:,2];
cv.imshow('B as Grayscale', b_gray)
cv.imshow('G as Grayscale', g_gray)
cv.imshow('R as Grayscale', r_gray)

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
