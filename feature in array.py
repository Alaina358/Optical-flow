import xlwt
import openpyxl
from xlwt import Workbook
import cv2
import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook
import csv
cap= cv2.VideoCapture(0)
i=0
while(i<2):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1
np_arr1=[]
first=cv2.imread('kang0.jpg')
resize_dim = 600
max_dim = max(first.shape)
scale = resize_dim/max_dim
first = cv2.resize(first, None, fx=scale, fy=scale)
prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)


mask = np.zeros_like(first)
mask[..., 1] = 255

frame=cv2.imread('kang1.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=scale, fy=scale)
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    # print(flow)
    # Compute the magnitude and angle of the 2D vectors
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
x=np.mean(magnitude)
np_arr1.append(x)
y=np.mean(angle)
np_arr1.append(y)

mask[..., 0] = angle * 180 / np.pi / 2
# Set image value according to the optical flow magnitude (normalized)
mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
# Convert HSV to RGB (BGR) color representation
rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # Resize frame size to match dimensions
frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # Open a new window and displays the output frame
dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
    #Crop 1
cropped1 = frame[0:150, 0:200]
    #cropped1 = np.sqrt(cropped1)
gx = cv2.Sobel(np.float32(cropped1), cv2.CV_32F, 1, 0)
gy = cv2.Sobel(np.float32(cropped1), cv2.CV_32F, 0, 1)
magnitude1, angle1 = cv2.cartToPolar(gx, gy)

x1=np.mean(magnitude1)
np_arr1.append(x1)
y1=np.mean(angle1)
np_arr1.append(y1)


cropped2 = frame[150:300, 0:200]
#cropped1 = np.sqrt(cropped1)
gx = cv2.Sobel(np.float32(cropped2), cv2.CV_32F, 1, 0)
gy = cv2.Sobel(np.float32(cropped2), cv2.CV_32F, 0, 1)
magnitude2, angle2 = cv2.cartToPolar(gx, gy)
# print(angle2.shape)
x2=np.mean(magnitude2)
np_arr1.append(x2)
y2=np.mean(angle2)
np_arr1.append(y2)


    #crop3
cropped3 = frame[300:450, 0:200]
#cropped1 = np.sqrt(cropped1)
gx = cv2.Sobel(np.float32(cropped3), cv2.CV_32F, 1, 0)
gy = cv2.Sobel(np.float32(cropped3), cv2.CV_32F, 0, 1)
magnitude3, angle3 = cv2.cartToPolar(gx, gy)
x3=np.mean(magnitude3)
np_arr1.append(x3)
y3=np.mean(angle3)
np_arr1.append(y3)

    #crop4
cropped4 = frame[0:150, 200:400]
#cropped1 = np.sqrt(cropped1)
gx = cv2.Sobel(np.float32(cropped4), cv2.CV_32F, 1, 0)
gy = cv2.Sobel(np.float32(cropped4), cv2.CV_32F, 0, 1)
magnitude4, angle4= cv2.cartToPolar(gx, gy)
x4=np.mean(magnitude4)
np_arr1.append(x4)
y4=np.mean(angle4)
np_arr1.append(y4)

    
    #crop5
cropped5 = frame[150:300, 200:400]
#cropped1 = np.sqrt(cropped1)
gx = cv2.Sobel(np.float32(cropped5), cv2.CV_32F, 1, 0)
gy = cv2.Sobel(np.float32(cropped5), cv2.CV_32F, 0, 1)
magnitude5, angle5= cv2.cartToPolar(gx, gy)
x5=np.mean(magnitude5)
np_arr1.append(x5)
y5=np.mean(angle5)
np_arr1.append(y5)

#crop6
cropped6 = frame[300:450, 200:400]
#cropped1 = np.sqrt(cropped1)
gx = cv2.Sobel(np.float32(cropped6), cv2.CV_32F, 1, 0)
gy = cv2.Sobel(np.float32(cropped6), cv2.CV_32F, 0, 1)
magnitude6, angle6= cv2.cartToPolar(gx, gy)
x6=np.mean(magnitude6)
np_arr1.append(x6)
y6=np.mean(angle6)
np_arr1.append(y6)

#crop7
cropped7 = frame[0:150, 400:600]
#cropped1 = np.sqrt(cropped1)
gx = cv2.Sobel(np.float32(cropped7), cv2.CV_32F, 1, 0)
gy = cv2.Sobel(np.float32(cropped7), cv2.CV_32F, 0, 1)
magnitude7, angle7= cv2.cartToPolar(gx, gy)
x7=np.mean(magnitude7)
np_arr1.append(x7)
y7=np.mean(angle7)
np_arr1.append(y7)


#crop8
cropped8 = frame[150:300, 400:600]
#cropped1 = np.sqrt(cropped1)
gx = cv2.Sobel(np.float32(cropped8), cv2.CV_32F, 1, 0)
gy = cv2.Sobel(np.float32(cropped8), cv2.CV_32F, 0, 1)
magnitude8, angle8= cv2.cartToPolar(gx, gy)
#print(angle8.shape)
x8=np.mean(magnitude8)
np_arr1.append(x8)
y8=np.mean(angle8)
np_arr1.append(y8)

#crop9
cropped9 = frame[300:450, 400:600]
#cropped1 = np.sqrt(cropped1)
gx = cv2.Sobel(np.float32(cropped9), cv2.CV_32F, 1, 0)
gy = cv2.Sobel(np.float32(cropped9), cv2.CV_32F, 0, 1)
magnitude9, angle9 = cv2.cartToPolar(gx, gy)
#print(angle9.shape)
x9=np.mean(magnitude9)
np_arr1.append(x9)
y9=np.mean(angle9)
np_arr1.append(y9)

print('array:', np_arr1)


cap.release()
cv2.destroyAllWindows()
