# Computer Vision using CV2 in Python
# Read and display RGB color image.
# Convert to HSV colorspace

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

# Convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv.bitwise_and(img, img, mask=mask)

# Display
cv.imshow('mask', mask)
cv.imshow('res', res)

# Get HSV values to compare
green = np.uint8([[[0,255,0 ]]])
blue = np.uint8([[[255,0,0 ]]])
hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
hsv_blue = cv.cvtColor(blue,cv.COLOR_BGR2HSV)
print( hsv_green )
print( hsv_blue )

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
