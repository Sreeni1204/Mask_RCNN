'''
Detection of objects (bottle caps) in input image.
'''

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import cv2

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "withroi_28012021/mask_rcnn_coco_0159.h5")

# loading config
import bottlecaps
config = bottlecaps.CocoConfig()
COCO_DIR = os.path.join(ROOT_DIR, "dataset/coco")
# inference config
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

#load and prepare dataset
dataset = bottlecaps.CocoDataset()
dataset.load_coco(COCO_DIR, "val")

# Must call before using the dataset
dataset.prepare()

#preferences
DEVICE = "/cpu:0"
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
weights_path = COCO_MODEL_PATH

model.load_weights(weights_path, by_name=True)

#loading test image
parser = argparse.ArgumentParser()

parser.add_argument('input_video', help = 'provide input video for detection', type=str)
parser.add_argument('result_folder', help = 'provide path for output files', type=str)

args = parser.parse_args()

input_video = args.input_video
result_folder = args.result_folder

# Extract static image from input video
read_video = cv2.VideoCapture(input_video)
length = int(read_video.get(cv2.CAP_PROP_FRAME_COUNT))
currentframe = 0

while(True):
    ret, frame = read_video.read()
    
    if ret:
        name = result_folder + '/frame_' + str(currentframe) + '.png'
        
        if currentframe == (int(length/2) + 10):
            cv2.imwrite(name, frame)
            break
    
        currentframe += 1
    else:
        break

img = load_img(name)
img = img_to_array(img)

output_file_name = os.path.basename(input_video)
output_file_name = output_file_name.split('.')
output_file_name = result_folder + '/' + output_file_name[0] +'.csv'

# Run object detection
results = model.detect([img], verbose=0)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions", output_filename=output_file_name, output_folder=result_folder, frame_number=currentframe)