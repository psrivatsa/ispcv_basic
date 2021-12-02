# Computer Vision using CV2 in Python
# Gamma Correction for Gray scale images

# import libraries
import sys
import cv2 as cv
import numpy as np

# Adjust gamma function
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

# print Python version
print("Python Version : " + sys.version)
# Check CV2 version
print("CV2 version : " + cv.__version__)

# read image
original = cv.imread("../images/woman.jpg", 1)

# check if file read is successful
if original is None:
    sys.exit("Could not read the image.")

# loop over various values of gamma
for gamma in np.arange(0.25, 3.0, 0.25):
    # apply gamma correction and show the images
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(original, gamma=gamma)
    cv.putText(adjusted, "g={}".format(gamma), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv.imshow("Images", np.hstack([original, adjusted]))
    cv.waitKey(0)

# close and exit
cv.destroyAllWindows()
