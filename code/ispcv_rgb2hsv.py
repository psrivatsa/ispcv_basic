# Computer Vision using CV2 in Python
# Read and display RGB color image.
# Convert to HSV colorspace

# import libraries
import sys
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to manually convert BGR image to HSV image
def bgr_to_hsv(img):
    res = np.zeros(shape=img.shape, dtype=np.uint8)
    r = img[:,:,2]/255
    g = img[:,:,1]/255
    b = img[:,:,0]/255
    maxc = np.maximum(r,b)
    maxc = np.maximum(maxc,g)
    minc = np.minimum(r,b)
    minc = np.minimum(minc,g)
    v = maxc
    diff = maxc - minc
    diff_eq = np.where(diff == 0, 1, diff)
    res[:,:,2] = v*255
    res[:,:,1] = np.divide(diff, maxc)*255
    rc = np.divide((maxc-r), diff_eq)
    gc = np.divide((maxc-g), diff_eq)
    bc = np.divide((maxc-b), diff_eq)
    h = np.where(maxc != minc, np.where(r == maxc, 0.0+bc-gc, res[:,:,0]), res[:,:,0])
    h = np.where(maxc != minc, np.where(g == maxc, 2.0+rc-bc, h), h)
    h = np.where(maxc != minc, np.where(b == maxc, 4.0+gc-rc, h), h)
    res[:,:,0] = np.around(((h/6.0) % 1.0)*179)
    return res

# Function for converting one pixel of RGB to HSV
# def rgb_to_hsv_pixel(r, g, b):
#   r /= 255
#   g /= 255
#   b /= 255
#   maxc = max(r, g, b)
#   minc = min(r, g, b)
#   v = maxc
#   if minc == maxc:
#       return 0.0, 0.0, v*255
#   s = (maxc-minc) / maxc
#   rc = (maxc-r) / (maxc-minc)
#   gc = (maxc-g) / (maxc-minc)
#   bc = (maxc-b) / (maxc-minc)
#   if r == maxc:
#       h = 0.0+bc-gc
#   elif g == maxc:
#       h = 2.0+rc-bc
#   else:
#       h = 4.0+gc-rc
#   h = (h/6.0) % 1.0
#   return h*179, s*255, v*255

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

# Convert BGR to HSV
hsv_inbuilt = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hsv_manual = bgr_to_hsv(img)

# define range of blue color in HSV
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# Threshold the HSV image to get only blue colors
mask1 = cv.inRange(hsv_inbuilt, lower_blue, upper_blue)
mask2 = cv.inRange(hsv_manual, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv.bitwise_and(img, img, mask=mask1)

# Checking peak-signal-to-noise-ratio (objective metric)
# Higher PSNR is better
psnr1 = cv.PSNR(hsv_inbuilt[:,:,0], hsv_manual[:,:,0])
print(psnr1)

# Checking structural similarity index (metric closer to human perception but subjective)
# SSIR range is [0,1]; 1 implies a perfect match with original
(score1, diff1) = ssim(hsv_inbuilt[:,:,0], hsv_manual[:,:,0], full=True)
print(score1)

# Display
cv.imshow('mask - inbuilt', mask1)
cv.imshow('mask - manual', mask2)
cv.imshow('res', res)
cv.imshow("HSV - inbuilt", hsv_inbuilt)
cv.imshow("HSV - manual", hsv_manual)

# Get HSV values to compare
green = np.uint8([[[42,179,142 ]]])
blue = np.uint8([[[255,0,0 ]]])
hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
hsv_blue = cv.cvtColor(blue,cv.COLOR_BGR2HSV)
# print( hsv_green )
# print( hsv_blue )

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
