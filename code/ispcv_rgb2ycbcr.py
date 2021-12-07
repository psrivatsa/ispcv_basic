# Computer Vision using CV2 in Python
# Convert RGB image to YCbCr color space
# https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

# Function to convert BGR to YCrCb
def bgr_to_ycrcb(img):
    B_chan = img[:,:,0]
    G_chan = img[:,:,1]
    R_chan = img[:,:,2]
    res = np.zeros(shape=img.shape, dtype=np.uint8)
    res[:,:,0] = 0.299*R_chan + 0.587*G_chan + 0.114*B_chan
    # res[:,:,1] = 0.615*R_chan - 0.515*G_chan - 0.100*B_chan + 128
    # res[:,:,2] = -0.147*R_chan - 0.289*G_chan + 0.436*B_chan + 128
    res[:,:,1] = 0.5*R_chan - 0.418688*G_chan - 0.081312*B_chan + 128
    res[:,:,2] = -0.168763*R_chan - 0.331264*G_chan + 0.5*B_chan + 128
    return res

# Function to convert YCrCb to BGR
def ycrcb_to_bgr(img):
    res = np.zeros(shape=img.shape, dtype=np.uint8)
    Y = img[:,:,0]
    V = img[:,:,1]
    U = img[:,:,2]
    res[:,:,0] = Y + 0.99752 * (U - 128)
    res[:,:,1] = Y - 0.13086 * (U - 128) - (1 * (V -128))
    res[:,:,2] = Y + 1.00413 * (V - 128)
    return res

# import libraries
import sys
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim

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
img_ycrcb_inbuilt = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
img_ycrcb_manual = bgr_to_ycrcb(img_bgr)

# Color space conversion to BGR
img_bgr_inbuilt = cv.cvtColor(img_ycrcb_inbuilt, cv.COLOR_YCrCb2BGR)
img_bgr_manual = ycrcb_to_bgr(img_ycrcb_inbuilt)

# Display YCrCb
cv.imshow("YCrCb - Inbuilt", img_ycrcb_inbuilt)
cv.imshow("YCrCb - Manual", img_ycrcb_manual)

# Display BGR obtained from YCrCb
cv.imshow("YCrCb to BGR - Inbuilt", img_bgr_inbuilt)
cv.imshow("YCrCb to BGR - Manual", img_bgr_manual)

# Display Y,Cr,Cb Channels seperately as Grayscale
Y_gray = img_ycrcb_inbuilt[:,:,0]
Cr_gray = img_ycrcb_inbuilt[:,:,1]
Cb_gray = img_ycrcb_inbuilt[:,:,2]
# cv.imshow('Y as Grayscale', Y_gray)
# cv.imshow('Cr as Grayscale', Cr_gray)
# cv.imshow('Cb as Grayscale', Cb_gray)

# with fake colors
Cr_fake = np.zeros(shape=img_bgr.shape, dtype=np.uint8)
Cb_fake = np.zeros(shape=img_bgr.shape, dtype=np.uint8)
Cr_fake[:,:,2] = img_ycrcb_inbuilt[:,:,1]
Cb_fake[:,:,0] = img_ycrcb_inbuilt[:,:,2]
# cv.imshow('Cr as red fake', Cr_fake)
# cv.imshow('Cb as blue fake', Cb_fake)

# Checking peak-signal-to-noise-ratio (objective metric)
# Higher PSNR is better
psnr = cv.PSNR(img_ycrcb_inbuilt[:,:,0], img_ycrcb_manual[:,:,0])
print(psnr)

# Checking structural similarity index (metric closer to human perception but subjective)
# SSIR range is [0,1]; 1 implies a perfect match with original
(score, diff) = ssim(img_bgr_inbuilt[:,:,0], img_bgr_manual[:,:,0], full=True)
print(score)

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
