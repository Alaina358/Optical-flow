import xlwt
import openpyxl
from xlwt import Workbook
import cv2
from openpyxl import load_workbook
import numpy as np
import os
CATEGORIES = ["walking"]

# Workbook is created
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
sheet1.write(0, 0, 'Image Name')
sheet1.write(0, 1, 'T_magnitude')
sheet1.write(0, 2, 'T_angle')
sheet1.write(0, 3, '1_magnitude')
sheet1.write(0, 4, '1_angle')
sheet1.write(0, 5, '2_magnitude')
sheet1.write(0, 6, '2_angle')
sheet1.write(0, 7, '3_magnitude')
sheet1.write(0, 8, '3_angle')
sheet1.write(0, 9, '4_magnitude')
sheet1.write(0, 10, '4_angle')
sheet1.write(0, 11, '5_magnitude')
sheet1.write(0, 12, '5_angle')
sheet1.write(0, 13, '6_magnitude')
sheet1.write(0, 14, '6_angle')
sheet1.write(0, 15, '7_magnitude')
sheet1.write(0, 16, '7_angle')
sheet1.write(0, 17, '8_magnitude')
sheet1.write(0, 18, '8_angle')
sheet1.write(0, 19, '9_magnitude')
sheet1.write(0, 20, '9_angle')
sheet1.write(0, 21, 'Label')  
#wb.save('xlwt Sheet.xls')
row=1
#os.chdir("Frames\\walking\\person01_walking_d1")
for category in CATEGORIES:
#  os.makedirs("Optical_flows/%s" % category, exist_ok=True)
  dir="Frames\\walking"
# go through dataset
for folder in os.listdir(dir):
  print("Processing category %s" % folder)
  
  # Get all files in current category's folder.
  folder_path = os.path.join("Frames", "walking", folder)  #this gives folder path
  filenames = os.listdir(folder_path)  # gives a list of all files in particular folder
  #print(filenames)
  i=0
  for filename in filenames:  
    first=cv2.imread(os.path.join("Frames", "walking", folder, filename))
    print(first)
    resize_dim = 600
    max_dim = max(first.shape)
    scale = resize_dim/max_dim
    first = cv2.resize(first, None, fx=scale, fy=scale)
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    print(prev_gray.shape)

    mask = np.zeros_like(first)
    mask[..., 1] = 255
    out = cv2.VideoWriter('person01_walking_d3_uncomp.avi',-1,1,(600, 600))
    break

  for filename in filenames: 
      
    column=0
    i=i+1
    if not i%8==0:
        continue
    print(filename)  
    frame=cv2.imread(os.path.join("Frames","walking", folder, filename))  
    sheet1.write(row, column, filename)
    column=column+1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    # print(flow)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print(angle.shape)
    #average mean
    x=np.mean(magnitude)
    print (x)
    y=np.mean(angle)
    print (y) 
    sheet1.write(row, column, float(x))
    column=column + 1
    sheet1.write(row, column, float(y))
    column=column + 1
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
    print(angle1.shape)
    x1=np.mean(magnitude1)
    print (x1)
    y1=np.mean(angle1)
    print (y1)
      
    sheet1.write(row, column, float(x1))
    column=column+1  
    sheet1.write(row, column, float(y1))
    column=column+1

    #Crop2
    cropped2 = frame[150:300, 0:200]
    #cropped1 = np.sqrt(cropped1)
    gx = cv2.Sobel(np.float32(cropped2), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(cropped2), cv2.CV_32F, 0, 1)
    magnitude2, angle2 = cv2.cartToPolar(gx, gy)
    print(angle2.shape)
    x2=np.mean(magnitude2)
    print (x2)
    y2=np.mean(angle2)
    print (y2)
  
    sheet1.write(row, column, float(x2))
    column=column+1  
    sheet1.write(row, column, float(y2))
    column=column+1

    #crop3
    cropped3 = frame[300:450, 0:200]
    #cropped1 = np.sqrt(cropped1)
    gx = cv2.Sobel(np.float32(cropped3), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(cropped3), cv2.CV_32F, 0, 1)
    magnitude3, angle3 = cv2.cartToPolar(gx, gy)
    print(angle3.shape)
    x3=np.mean(magnitude3)
    print (x3)
    y3=np.mean(angle3)
    print (y3)

  
    sheet1.write(row, column, float(x3))
    column=column+1  
    sheet1.write(row, column, float(y3))
    column=column+1


    #crop4
    cropped4 = frame[0:150, 200:400]
    #cropped1 = np.sqrt(cropped1)
    gx = cv2.Sobel(np.float32(cropped4), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(cropped4), cv2.CV_32F, 0, 1)
    magnitude4, angle4= cv2.cartToPolar(gx, gy)
    print(angle4.shape)
    x4=np.mean(magnitude4)
    print (x4)
    y4=np.mean(angle4)
    print (y4)

    sheet1.write(row, column, float(x4))
    column=column+1  
    sheet1.write(row, column, float(y4))
    column=column+1
    
    #crop5
    cropped5 = frame[150:300, 200:400]
    #cropped1 = np.sqrt(cropped1)
    gx = cv2.Sobel(np.float32(cropped5), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(cropped5), cv2.CV_32F, 0, 1)
    magnitude5, angle5= cv2.cartToPolar(gx, gy)
    print(angle5.shape)
    x5=np.mean(magnitude5)
    print (x5)
    y5=np.mean(angle5)
    print (y5)

  
    sheet1.write(row, column, float(x5))
    column=column+1  
    sheet1.write(row, column, float(y5))
    column=column+1

    #crop6
    cropped6 = frame[300:450, 200:400]
    #cropped1 = np.sqrt(cropped1)
    gx = cv2.Sobel(np.float32(cropped6), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(cropped6), cv2.CV_32F, 0, 1)
    magnitude6, angle6= cv2.cartToPolar(gx, gy)
    print(angle6.shape)
    x6=np.mean(magnitude6)
    print (x6)
    y6=np.mean(angle6)
    print (y6)
  
    sheet1.write(row, column, float(x6))
    column=column+1  
    sheet1.write(row, column, float(y6))
    column=column+1

    #crop7
    cropped7 = frame[0:150, 400:600]
    #cropped1 = np.sqrt(cropped1)
    gx = cv2.Sobel(np.float32(cropped7), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(cropped7), cv2.CV_32F, 0, 1)
    magnitude7, angle7= cv2.cartToPolar(gx, gy)
    print(angle7.shape)
    x7=np.mean(magnitude7)
    print (x7)
    y7=np.mean(angle7)
    print (y7)
    
      
    sheet1.write(row, column, float(x7))
    column=column+1  
    sheet1.write(row, column, float(y7))
    column=column+1

    #crop8
    cropped8 = frame[150:300, 400:600]
    #cropped1 = np.sqrt(cropped1)
    gx = cv2.Sobel(np.float32(cropped8), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(cropped8), cv2.CV_32F, 0, 1)
    magnitude8, angle8= cv2.cartToPolar(gx, gy)
    #print(angle8.shape)
    x8=np.mean(magnitude8)
    print (x8)
    y8=np.mean(angle8)
    print (y8)
      
    sheet1.write(row, column, float(x8))
    column=column+1  
    sheet1.write(row, column, float(y8))
    column=column+1

     #crop9
    cropped9 = frame[300:450, 400:600]
    #cropped1 = np.sqrt(cropped1)
    gx = cv2.Sobel(np.float32(cropped9), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(cropped9), cv2.CV_32F, 0, 1)
    magnitude9, angle9 = cv2.cartToPolar(gx, gy)
    #print(angle9.shape)
    x9=np.mean(magnitude9)
    print (x9)
    y9=np.mean(angle9)
    print (y9)
    
  
    sheet1.write(row, column, float(x9))
    column=column+1  
    sheet1.write(row, column, float(y9))
    column=column+1

    #cv2.imshow("Dense optical flow", dense_flow)
    out.write(dense_flow)
    path = 'Optical flow\\walking\\person01_walking_d3'
    # Update previous frame
    prev_gray = gray
    # Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    row=row+1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
#vc.release()

wb.save('Sheet_walking.xls')
cv2.destroyAllWindows()


# In[ ]: