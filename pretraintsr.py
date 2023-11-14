import os
import cv2
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import ultralytics
from ultralytics import YOLO
import yaml
import subprocess

# Constants for paths, these should be configured as per your directory structure
DATA_DIR = '/content/drive/MyDrive/mytsr2023/datasets/nzts02'
TRAIN_PATH = '/content/drive/MyDrive/mytsr2023/datasets/train'
VALID_PATH = '/content/drive/MyDrive/mytsr2023/datasets/valid'
TEST_PATH = '/content/drive/MyDrive/mytsr2023/datasets/test'

# Collect all annotation paths
ano_paths = []
for dirname, _, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        if filename.endswith('.txt'):  # Assuming annotations are in .txt format
            ano_paths.append(os.path.join(dirname, filename))

# Shuffle and split the dataset
random.shuffle(ano_paths)
n = len(ano_paths)
train_size = int(n * 0.7)
valid_size = int(n * 0.2)
test_size = n - train_size - valid_size

train_paths = ano_paths[:train_size]
valid_paths = ano_paths[train_size:train_size + valid_size]
test_paths = ano_paths[train_size + valid_size:]

# Function to copy annotation and image files to the specified directory
def copy_files(file_paths, src_dir, dst_dir):
    for file_path in tqdm(file_paths):
        img_path = file_path.replace('.txt', '.jpg')  # Assuming images are in .jpg format
        # Copy annotation file
        shutil.copy(file_path, dst_dir)
        # Copy corresponding image file
        if os.path.exists(img_path):
            shutil.copy(img_path, dst_dir)

# Copying files to the respective train/valid/test directories
copy_files(train_paths, DATA_DIR, TRAIN_PATH)
copy_files(valid_paths, DATA_DIR, VALID_PATH)
copy_files(test_paths, DATA_DIR, TEST_PATH)

# Check the number of files in each directory (optional)
print(f"Train set size: {len(os.listdir(TRAIN_PATH))}")
print(f"Valid set size: {len(os.listdir(VALID_PATH))}")
print(f"Test set size: {len(os.listdir(TEST_PATH))}")

# Creating and saving the data.yaml configuration file
data_yaml = {
    'train': TRAIN_PATH,
    'val': VALID_PATH,
    'test': TEST_PATH,
    'nc': 9,  # Number of classes
    #'names': ['prohibitor','danger','mandatory','other']
    'names': ['Attention', 'Giveway', 'Giveway_Roundabout', 'Keepleft',
         'Narrowahead', 'Norightturn', 'Other', 'Roaddiverges', 'Speedlimits']
}

with open('data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)




