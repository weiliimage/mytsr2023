1. Introduction
This project is developed for low-light traffic sign image recognition. A DRL-based framework is employed for low-light image enhancement, 
and a YOLOv8 model is applied for traffic sign classification. 

2. open Google drive in Colab
from google.colab import drive
drive.mount('/content/drive')
import os
path = "/content/drive/My Drive/mytsr2023"
os.chdir(path)
print(os.getcwd())

3. Install libraries
* Python 3.5+
* Chainer 7.8.0
* Chainerrl 0.8.0
* gym 0.17.3
* scikit-image 0.19.3
* Cupy 5.0+
* OpenCV 3.4+
* Torch 1.6
* wandv 0.15.0
* ultralytics


4. Prepare the datasets
The link to our dataset NZTS2024:
https://www.kaggle.com/datasets/willliwill2023/nzts2024low-light-traffic-sign-dataset-in-nz

NZTS2024 consists of LLIE folder and TSR folder.

or
you can prepare your dataset like this:
datasets/nzts01low     :low-light enhancement training dataset
datasets/nzts01high    :low-light enhancement labelled dataset
datasets/nzts02        :traffic sign classification dataset
datasets/lowlight      :low-light test dataset

5. Train the models
5.1 Training DRL model

```
%run traindrl.py
```

5.2 Training YOLOv8 model
5.2.1 prepareing the model
```
%run pretraintsr.py
```
5.2.2 train the model
'''
import torch
from ultralytics import YOLO
model = YOLO("yolov8x.pt")  
results = model.train(data="/content/drive/My Drive/mytsr2023/data.yaml",
      name = 'mytsr', epochs=100, patience=20, batch = 16,
      cache = True, imgsz=640, iou = 0.5, lr0=0.0001, optimizer='Adam')
'''


6. Test the model
6.1 low-light enhancement Test
```
%run testdrl.py
```

6.2 TSR test
``
from PIL import Image
import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

model = YOLO('./runs/detect/mytsr2/weights/best.pt')
results = model('/content/drive/My Drive/mytsr2023/datasets/lowoutput/')
for r in results:
    im_array = r.plot()  
    im = Image.fromarray(im_array[..., ::-1]) 
    img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB) 
    cv2_imshow(img)  


