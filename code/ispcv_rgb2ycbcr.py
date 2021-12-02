# Computer Vision using CV2 in Python
# Convert RGB image to YCbCr color space
# https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

# import libraries
import sys
import cv2 as cv
import numpy as np

# print Python version
print("Python Version : " + sys.version)
# Check CV2 version
print("CV2 version : " + cv.__version__)

# read image
img_bgr = cv.imread("../images/smarties.png",1)

# check if file read is successful
if img_bgr is None:
    sys.exit("Could not read the image.")

# Display BGR image
cv.imshow("BGR Color space", img_bgr)

# Color space conversion to YCbCr
img_ycrcb = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)

# Display YCrCb
cv.imshow("YCrCb displayed in RGB color space", img_ycrcb)

# Display Y,Cr,Cb Channels seperately as Grayscale
Y_gray = img_ycrcb[:,:,0];
Cr_gray = img_ycrcb[:,:,1];
Cb_gray = img_ycrcb[:,:,2];
cv.imshow('Y as Grayscale', Y_gray)
cv.imshow('Cr as Grayscale', Cr_gray)
cv.imshow('Cb as Grayscale', Cb_gray)

# with fake colors
Cr_fake = np.zeros(shape=img_bgr.shape, dtype=np.uint8)
Cb_fake = np.zeros(shape=img_bgr.shape, dtype=np.uint8)
Cr_fake[:,:,2] = img_ycrcb[:,:,1]
Cb_fake[:,:,0] = img_ycrcb[:,:,2]
cv.imshow('Cr as red fake', Cr_fake)
cv.imshow('Cb as blue fake', Cb_fake)

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
