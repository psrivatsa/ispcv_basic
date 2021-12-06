# Computer Vision using CV2 in Python
# Read and display color image.
# Compare BGR and RGB colorspaces

# import libraries
import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

# read image (cv reads it as BGR)
img = cv.imread(r"C:\Users\HP.LAPTOP-QQJDP0U9\Desktop\Python\ISPCV\ispcv_basic\images\lena_color_512.tif",1)

# check if file read is successful
if img is None:
    sys.exit("Could not read the image.")

# display the type after reading
print(type(img))

# Convert to RGB
img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Display image
# cv.imshow("BGR - CV2", img)
# cv.imshow("RGB - CV2", img_RGB)

# Concatenate the images
img_conc = cv.hconcat([img, img_RGB])
cv.imshow("CV2 - BGR (left) vs RGB (right", img_conc)

# Displaying same image using Matplotlib
fig = plt.figure(num="Matplotlib", figsize=(8, 4))

fig.add_subplot(1,2,1)
plt.imshow(img)
plt.axis('off')
plt.title("BGR")
fig.add_subplot(1,2,2)
plt.imshow(img_RGB)
plt.axis('off')
plt.title("RGB")
plt.show()

# wait for keystroke
k = cv.waitKey(0)

# close and exit
cv.destroyAllWindows()