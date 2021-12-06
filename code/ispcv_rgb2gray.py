# Computer Vision using CV2 in Python
# Convert RGB to grayscale image
# https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

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
img_bgr = cv.imread("../images/23.tif",1)

# check if file read is successful
if img_bgr is None:
    sys.exit("Could not read the image.")

# Display BGR image
cv.imshow("BGR Color space", img_bgr)

# Color space conversion to gray
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# Display Grayscale
cv.imshow("Grayscale image - Inbuilt conversion", img_gray)

# Manual color space conversion to gray
height, width, channel = img_bgr.shape
img_gray_man1 = np.zeros(shape=(height, width), dtype=np.uint8)
img_gray_man2 = np.zeros(shape=(height, width), dtype=np.uint8)
R_chan = img_bgr[:,:,2]
G_chan = img_bgr[:,:,1]
B_chan = img_bgr[:,:,0]

# RGB to gray equation 1
img_gray_man1 = 0.2125*R_chan + 0.7154*G_chan + 0.0721*B_chan         
img_gray_man1 = img_gray_man1.astype(np.uint8)                    # imshow needs unit8 datatype 

# RGB to gray equation 2
img_gray_man2 = 0.299*R_chan + 0.587*G_chan + 0.114*B_chan         
img_gray_man2 = img_gray_man2.astype(np.uint8)                    # imshow needs unit8 datatype 

# Checking peak-signal-to-noise-ratio (objective metric)
# Higher PSNR is better
psnr1 = cv.PSNR(img_gray, img_gray_man1)
psnr2 = cv.PSNR(img_gray, img_gray_man2)

# Checking structural similarity index (metric closer to human perception but subjective)
# SSIR range is [0,1]; 1 implies a perfect match with original
(score1, diff1) = ssim(img_gray, img_gray_man1, full=True)
(score2, diff2) = ssim(img_gray, img_gray_man2, full=True)

# Display manual grayscale and comparison values for equation 1
cv.putText(img_gray_man1, "PSNR={} SSIM={}".format(np.round(psnr1,2), np.round(score1,3)), (10, 30), 4, 0.8, (0, 0, 0), 3)
cv.imshow("Grayscale image - Manual conversion 1", img_gray_man1)

# Display manual grayscale and comparison values for equation 2
cv.putText(img_gray_man2, "PSNR={} SSIM={}".format(np.round(psnr2,2), np.round(score2,3)), (10, 30), 4, 0.8, (0, 0, 255), 3)
cv.imshow("Grayscale image - Manual conversion 2", img_gray_man2)

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
