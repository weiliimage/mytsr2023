import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')
from PIL import Image
import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import yaml



ano_paths=[]
for dirname, _, filenames in os.walk('/content/gdrive/MyDrive/mystr2023/datasets/nzts'):
    for filename in filenames:
        ano_paths+=[(os.path.join(dirname, filename))]
        
n=600 #len(ano_paths) 
print(n)
N=list(range(n))
random.shuffle(N)

train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1

train_size = int(train_ratio*n)
valid_size = int(valid_ratio*n)

train_i = N[:train_size]
valid_i = N[train_size:train_size+valid_size]
test_i = N[train_size+valid_size:]


for i in train_i:
    ano_path=ano_paths[i]
    img_path=os.path.join('/content/gdrive/MyDrive/mystr2023/datasets/nzts',
                          ano_path.split('/')[-1][0:-4]+'.jpg')
    try:
        !cp {ano_path} {train_path}
        !cp {img_path} {train_path}
    except:
        continue

for i in valid_i:
    ano_path=ano_paths[i]
    img_path=os.path.join('/content/gdrive/MyDrive/mystr2023/datasets/nzts',
                          ano_path.split('/')[-1][0:-4]+'.jpg')
    try:
        !cp {ano_path} {valid_path}
        !cp {img_path} {valid_path}
    except:
        continue

for i in test_i:
    ano_path=ano_paths[i]
    img_path=os.path.join('/content/gdrive/MyDrive/mystr2023/datasets/nzts',
                          ano_path.split('/')[-1][0:-4]+'.jpg')
    try:
        !cp {ano_path} {test_path}
        !cp {img_path} {test_path}
    except:
        continue
print(len(os.listdir(test_path)))  


data_yaml = dict(
    train ='/content/gdrive/MyDrive/mystr2023/datasets/train',
    val ='/content/gdrive/MyDrive/mystr2023/datasets/valid',
    test='/content/gdrive/MyDrive/mystr2023/datasets/test',
    nc =9,
    names =['giveway','keep_left','no_right_turn','giveway_roundabout', 
            'road_diverges', 'attention','speed','narrow_ahead','other']
)

with open('data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)
    
%cat data.yaml


names =['giveway','keep_left','no_right_turn','giveway_roundabout', 
            'road_diverges', 'attention','speed','narrow_ahead','other']
M=list(range(len(names)))
class_map=dict(zip(M,names))


###Train
model = YOLO("yolov8x.pt") 

!yolo task=detect mode=train model=yolov8x.pt data=data.yaml epochs=12 imgsz=480

paths2=[]
for dirname, _, filenames in os.walk('/content/gdrive/MyDrive/mystr2023/runs/detect/train'):
    for filename in filenames:
        if filename[-4:]=='.jpg':
            paths2+=[(os.path.join(dirname, filename))]
paths2=sorted(paths2)


for path in paths2:
    image = Image.open(path)
    image=np.array(image)
    plt.figure(figsize=(20,10))
    plt.imshow(image)
    plt.show()
    
#################################


###Predict
best_path0='/content/gdrive/MyDrive/yolo8tsr/runs/detect/train3/weights/best.pt'
source0='/content/gdrive/MyDrive/yolo8tsr/datasets/test'


ppaths=[]
for dirname, _, filenames in os.walk(source0):
    for filename in filenames:
        if filename[-4:]=='.jpg':
            ppaths+=[(os.path.join(dirname, filename))]
ppaths=sorted(ppaths)
print(ppaths[0])
print(len(ppaths))

####################################
model2 = YOLO(best_path0)

!yolo task=detect mode=predict model={best_path0} conf=0.5 source={source0}


###result of prediction
results = model2.predict(source0,conf=0.5)
print(len(results))

####################################
print((results[0].boxes.data))


PBOX=pd.DataFrame(columns=range(6))
for i in range(len(results)):
    arri=pd.DataFrame(results[i].boxes.data.cpu().numpy()).astype(float)
    path=ppaths[i]
    file=path.split('/')[-1]
    arri=arri.assign(file=file)
    arri=arri.assign(i=i)
    PBOX=pd.concat([PBOX,arri],axis=0)
PBOX.columns=['x','y','x2','y2','confidence','class','file','i']
display(PBOX)


PBOX['class']=PBOX['class'].apply(lambda x: class_map[int(x)])
PBOX=PBOX.reset_index(drop=True)
display(PBOX)
display(PBOX['class'].value_counts())



def draw_box2(n0):
    
    ipath=ppaths[n0]
    image=cv2.imread(ipath)
    H,W=image.shape[0],image.shape[1]
    file=ipath.split('/')[-1]
    
    if PBOX[PBOX['file']==file] is not None:
        box=PBOX[PBOX['file']==file]
        box=box.reset_index(drop=True)
        #display(box)

        for i in range(len(box)):
            label=box.loc[i,'class']
            x=int(box.loc[i,'x'])
            y=int(box.loc[i,'y'])
            x2=int(box.loc[i,'x2']) 
            y2=int(box.loc[i,'y2'])
            #print(label,x,y,x2,y2)
            cv2.putText(image, f'{label}', (x, int(y-4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)
            cv2.rectangle(image,(x,y),(x2,y2),(0,255,0),2) #green
    
    #plt.imshow(image)
    #plt.show()   
    
    return image
    

def create_animation(ims):
    fig=plt.figure(figsize=(12,8))
    im=plt.imshow(cv2.cvtColor(ims[0],cv2.COLOR_BGR2RGB))
    text = plt.text(0.05, 0.05, f'Slide {0}', transform=fig.transFigure, fontsize=14, color='blue')
    plt.axis('off')
    plt.close()

    def animate_func(i):
        im.set_array(cv2.cvtColor(ims[i],cv2.COLOR_BGR2RGB))
        text.set_text(f'Slide {i}')        
        return [im]    
    
    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000)
    

images2=[]
for i in tqdm(range(len(ppaths))):
    images2+=[draw_box2(i)]
    

create_animation(images2)


