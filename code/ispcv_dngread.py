# Computer Vision using CV2 in Python
# Read RAW/DNG images and convert to RGB color space
# using 'rawpy' functions to read raw images

# import libraries
import sys
import cv2 as cv
import numpy as np
import rawpy

# print Python version
print("Python Version : " + sys.version)
# Check CV2 version
print("CV2 version : " + cv.__version__)

# read image
img_dng = rawpy.imread("../images/tintin1.dng")

# check if file read is successful
if img_dng is None:
    sys.exit("Could not read the image.")

# Convert to RGB with post process
img_rgb = img_dng.postprocess()

# Display BGR image
img_rgb_fhd = cv.resize(img_rgb, (1920, 1080))
cv.imshow("RGB Color space", img_rgb)
cv.imshow("RGB Color space, scaled to FHD", img_rgb_fhd)
img_bgr_fhd = cv.cvtColor(img_rgb_fhd, cv.COLOR_RGB2BGR)
cv.imshow("BGR Color space, scaled to FHD", img_bgr_fhd)

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
