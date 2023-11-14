import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import animation, rc
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

rc('animation', html='jshtml')

# Constants for paths, these should be configured as per your directory structure
BEST_MODEL_PATH = '/content/drive/MyDrive/mytsr2023/runs/detect/train/weights/best.pt'
TEST_IMAGES_PATH = '/content/drive/MyDrive/mytsr2023/datasets/test'

# Creating a mapping from class indices to human-readable class names
"""
names = ['giveway', 'keep_left', 'no_right_turn', 'giveway_roundabout',
         'road_diverges', 'attention', 'speed', 'narrow_ahead', 'other']
"""
names = ['prohibitor','danger','mandatory','other']
class_map = {i: name for i, name in enumerate(names)}

# Loading the model
model2 = YOLO(BEST_MODEL_PATH)

# Running the model on the test dataset
results = model2.predict(TEST_IMAGES_PATH, conf=0.5)
print(f"Number of results: {len(results)}")

# Print the bounding box details of the first result
if len(results) > 0:
    print(f"Details of the first result's bounding boxes: {results[0].boxes.data}")

# Processing the results for all images and creating a DataFrame
PBOX = pd.DataFrame(columns=range(6))
ppaths = []
for dirname, _, filenames in os.walk(TEST_IMAGES_PATH):
    for filename in filenames:
        if filename.endswith('.jpg'):
            path = os.path.join(dirname, filename)
            ppaths.append(path)
            image_results = next((res for res in results if res.image_path == path), None)
            if image_results is not None:
                boxes_data = image_results.boxes.data.cpu().numpy()
                for box in boxes_data:
                    box_df = pd.DataFrame([box], columns=['x', 'y', 'x2', 'y2', 'confidence', 'class'])
                    box_df['file'] = filename
                    box_df['i'] = len(ppaths) - 1
                    PBOX = pd.concat([PBOX, box_df])

# Adding class names to the DataFrame
PBOX['class'] = PBOX['class'].apply(lambda x: class_map[int(x)])
PBOX = PBOX.reset_index(drop=True)
display(PBOX)
display(PBOX['class'].value_counts())

# Function to draw bounding boxes on an image
def draw_box2(n0):
    ipath = ppaths[n0]
    image = cv2.imread(ipath)
    file = ipath.split('/')[-1]
    
    if PBOX[PBOX['file'] == file] is not None:
        box = PBOX[PBOX['file'] == file]
        box = box.reset_index(drop=True)

        for i in range(len(box)):
            label = box.loc[i, 'class']
            x = int(box.loc[i, 'x'])
            y = int(box.loc[i, 'y'])
            x2 = int(box.loc[i, 'x2']) 
            y2 = int(box.loc[i, 'y2'])
            cv2.putText(image, f'{label}', (x, int(y - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    
    return image

# Create an animation showing the bounding boxes
def create_animation(ims):
    fig = plt.figure(figsize=(12, 8))
    im = plt.imshow(cv2.cvtColor(ims[0], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.close()

    def animate_func(i):
        im.set_array(cv2.cvtColor(ims[i], cv2.COLOR_BGR2RGB))
        return [im]

    anim = animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000)
    return anim

# Collecting the images with drawn bounding boxes
images2 = [draw_box2(i) for i in tqdm(range(len(ppaths)))]

# Create and display the animation
anim = create_animation(images2)
display(anim)
