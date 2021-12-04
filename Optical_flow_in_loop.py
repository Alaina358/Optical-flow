import cv2
import numpy as np
import os
CATEGORIES = ["boxing"]
j=1
# make a folder named data
#os.makedirs("Optical_flows", exist_ok=True)

# traverse through categories to make the each caegory folder
for category in CATEGORIES:
#  os.makedirs("Optical_flows/%s" % category, exist_ok=True)
  dir="Frames\\boxing"
# go through dataset
for folder in os.listdir(dir):
  print("Processing category %s" % folder)
  
  # Get all files in current category's folder.
  folder_path = os.path.join("Frames", "boxing", folder)  #this gives folder path
  filenames = os.listdir(folder_path)  # gives a list of all files in particular folder
  #print(filenames)
  i=0
  for filename in filenames:  
    first=cv2.imread(os.path.join("Frames","boxing",folder, filename))
    print(first)
    resize_dim = 600
    max_dim = max(first.shape)
    scale = resize_dim/max_dim
    first = cv2.resize(first, None, fx=scale, fy=scale)
    # Convert to gray scale 
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    #print(pre)v_gray)
    print(prev_gray.shape)

    # Create mask
    mask = np.zeros_like(first)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    out = cv2.VideoWriter('person01_boxing_d3_uncomp.avi',-1,1,(600, 600))
    break

  for filename in filenames:
    i=i+1
    if not i%8==0:
        continue
    print(filename)  
    frame=cv2.imread(os.path.join("Frames","boxing", folder, filename))
    print(frame)
    # Convert new frame format`s to gray scale and resize gray frame obtained
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale)

    # Calculate dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print(magnitude.shape)
    # Set image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    # Resize frame size to match dimensions
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
    # Open a new window and displays the output frame
    dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
    cv2.imshow("Dense optical flow", dense_flow)
    out.write(dense_flow)
    image_path = os.path.join(folder, "frame%d.jpg" % i)
    path = 'Optical flow\\boxing\\person01_boxing_d3'
    cv2.imwrite(os.path.join("data","boxing",folder ,"framePF%d.jpg" % j), dense_flow)
    j=j+1 
    # Update previous frame
    prev_gray = gray
    # Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
#vc.release()

cv2.destroyAllWindows()

    # #Make separate folder for each file
    # persdon_folder = os.path.join("data", category, filename)
    # os.makedirs( persdon_folder , exist_ok=True)
    # filepath = os.path.join("input_videos", category, filename)
    # vidcap = cv2.VideoCapture(filepath)
    # success,image = vidcap.read()
    # count = 0
    # while success:
    #   image_path = os.path.join(persdon_folder, "frame%d.jpg" % count)
    #   cv2.imwrite(image_path, image)     # save frame as JPEG file      
    #   success, image = vidcap.read()
    #   print('Read a new frame: ', success)
    #   count += 1
