# Computer Vision using CV2 in Python
# Convert RGB to grayscale image
# https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

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

# Color space conversion to YCbCr
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# Display Grayscale
cv.imshow("Grayscale image", img_gray)

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
