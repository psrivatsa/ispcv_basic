# Computer Vision using CV2 in Python
# Convert RGB image to YCbCr color space
# https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

# import libraries
import sys
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to convert BGR to YUV420
def bgr_to_ycrcb420(img, sampling_type=0):
    height, width, channel = img.shape
    res = np.zeros(shape=(int(1.5*height), width), dtype=np.uint8)
    yuv = bgr_to_ycrcb(img)
    res[0:height,:] = yuv[:,:,0]
    res[height:height+int(height/4), 0:int(width/2)] = ss(yuv[:,:,2],1,sampling_type)
    res[height:height+int(height/4), int(width/2):width] = ss(yuv[:,:,2],1,sampling_type)
    res[height+int(height/4):int(1.5*height), 0:int(width/2)] = ss(yuv[:,:,1],0,sampling_type)
    res[height+int(height/4):int(1.5*height), int(width/2):width] = ss(yuv[:,:,1],0,sampling_type)
    return res

# Function to convert BGR to YCrCb
def bgr_to_ycrcb(img):
    B_chan = img[:,:,0]
    G_chan = img[:,:,1]
    R_chan = img[:,:,2]
    res = np.zeros(shape=img.shape, dtype=np.uint8)
    res[:,:,0] = 0.299*R_chan + 0.587*G_chan + 0.114*B_chan
    res[:,:,1] = 0.5*R_chan - 0.418688*G_chan - 0.081312*B_chan + 128
    res[:,:,2] = -0.168763*R_chan - 0.331264*G_chan + 0.5*B_chan + 128
    return res

# Function used for choosing chroma subsampling
def ss(img,num,type):
    return naive_ss(img,num) if (type == 0) else average_ss(img, num)

# Function for naive sub-sampling
# Taking every other U or V values (eg. U0, U2, U4, ...)
def naive_ss(img,num):
    h, w = img.shape
    res = np.zeros(shape=(int(h/4),int(w/2)), dtype=np.uint8)
    res = img[num::4,::2]
    return res

# Function for average sub-sampling
# Taking average of 2x2 submatrices
def average_ss(img,num):
    h, w = img.shape
    res = np.zeros(shape=(int(h/4),int(w/2)), dtype=np.uint8)
    y = img.reshape(int(img.shape[0]/2), 2, int(img.shape[1]/2), 2)
    avg = y.mean(axis=(1, 3))
    res = avg[num::2,:]
    return res

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
img_data_man = bgr_to_ycrcb420(img_bgr,1)               # 0 = Naive subsampling, 1 = Average subsampling

# Checking peak-signal-to-noise-ratio (objective metric)
# Higher PSNR is better
psnr = cv.PSNR(img_data, img_data_man)

# Checking structural similarity index (metric closer to human perception but subjective)
# SSIR range is [0,1]; 1 implies a perfect match with original
(score, diff) = ssim(img_data, img_data_man, full=True)

# Display YCbCr
cv.imshow("YCbCr420 displayed in Grayscale as large ndarray/single plane", img_data)
cv.putText(img_data_man, "PSNR={} SSIM={}".format(np.round(psnr,3), np.round(score,3)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
cv.imshow("Manual", img_data_man)

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
