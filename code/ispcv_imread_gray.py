# Computer Vision using CV2 in Python
# Read and display grayscale image
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
img = cv.imread("../images/lena_gray_512.tif",0)

# check if file read is successful
if img is None:
    sys.exit("Could not read the image.")

# Display image
cv.imshow("Gray Scale Image", img)

# wait for keystroke
k = cv.waitKey(0)

# if key is 's', save as png file
if k == ord("s"):
    cv.imwrite("./data/lena_gray_512.png", img)

# close and exit
cv.destroyAllWindows()
