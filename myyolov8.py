#step 1: connect google drive to colab

from google.colab import drive
drive.mount('/content/drive')

import os
path = "/content/drive/My Drive/mytsr2023"
os.chdir(path)
print(os.getcwd())

# step 2 install libraries

!pip install wandb==0.15.0
!wandb --version
!pip install ultralytics

#step 3 prepare datasets




#step 4 train yolov8 model

import torch
from ultralytics import YOLO

model = YOLO("yolov8x.pt")  # load yolov8 model

# training
results = model.train(data="/content/drive/My Drive/mytsr2023/data.yaml", 
      name = 'mytsr', epochs=100, patience=20, batch = 16, 
      cache = True, imgsz=640, iou = 0.5, lr0=0.0001, optimizer='Adam')
	  
	  
	  
#step 5 test results
	  
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow

model = YOLO('./runs/detect/mytsr2/weights/best.pt')  # load the weights

results = model('/content/drive/My Drive/mytsr2023/datasets/test/0005.jpg')  # input testing image

for r in results:
    im_array = r.plot()  
    im = Image.fromarray(im_array[..., ::-1])  
    img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB) 
    cv2_imshow(img)  