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
img_bgr = cv.imread("../images/23.tif",1)

# check if file read is successful
if img_bgr is None:
    sys.exit("Could not read the image.")

# Display BGR image
cv.imshow("BGR Color space", img_bgr)
(row, col, plane) = img_bgr.shape

# Color space conversion to YCbCr
img_data = cv.cvtColor(img_bgr, cv.COLOR_BGR2YUV_I420)

# Display YCbCr
cv.imshow("YCbCr420 displayed in Grayscale as large ndarray/single plane", img_data)

# Display Y,Cb,Cr Channels separately as Grayscale
# Y
Y_gray = img_data[1:row, 1:col]

# Cb
print(type(row))
Cb_gray0 = img_data[int(row+1):int(row+row/4),:]
Cb_gray1 = np.resize(Cb_gray0, (int(row/2),int(col/2)))
# Cr
Cr_gray0 = img_data[int(row+row/4+1):int(col),:]
Cr_gray1 = np.resize(Cr_gray0, (int(row/2),int(col/2)))

cv.imshow('Y as Grayscale', Y_gray)
cv.imshow('Cb as Grayscale', Cb_gray1)
cv.imshow('Cr as Grayscale', Cr_gray1)

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
