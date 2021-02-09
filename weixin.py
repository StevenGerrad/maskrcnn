import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import PIL
import PIL.Image as Image
import yaml


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

print(-4)
class skinConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "skin"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 10

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1


config = skinConfig()
config.display()


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class SkinDataset(utils.Dataset):
    def load_info(self):
        self.add_class("skin", 1, "closedcomedo")
        self.add_class("skin", 2, "opencomedo")
        self.add_class("skin", 3, "papule")
        self.add_class("skin", 4, "nudule")
        self.add_class("skin", 5, "scar")
        self.add_class("skin", 6, "undefined")

        data_dir_list = []
        for name in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, name)
            if os.path.isdir(path):
                data_dir_list.append(path)

        for i, data_dir in enumerate(data_dir_list):
            self.add_image("skin", image_id=i, path=data_dir)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        img_path = os.path.join(info['path'], img_file)
        image = np.asarray(Image.open(img_path))
        return image

    def draw_mask(self, mask, mask_flat):
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                pixel = mask_flat[i, j]
                if pixel > 0:
                    mask[i, j, pixel - 1] = 1
        return mask

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        mask_path = os.path.join(info['path'], mask_file)
        mask_flat = np.asarray(Image.open(mask_path))
        instance_num = np.max(mask_flat)
        mask = np.zeros([mask_flat.shape[0], mask_flat.shape[1], instance_num], dtype=np.uint8)
        mask = self.draw_mask(mask, mask_flat)
        '''
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(instance_num - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        '''
        instance_name_path = os.path.join(info['path'], instance_name_file)
        with open(instance_name_path, encoding='utf-8') as file:
            instance_name_list = file.readlines()
        instance_name_list = [instance_name.strip('\n') for instance_name in instance_name_list]
        del instance_name_list[0]
        class_ids = np.array([self.class_names.index(s) for s in instance_name_list])

        return mask.astype(np.bool), class_ids.astype(np.int32)

img_file = 'img.png'
mask_file = 'mask.png'
instance_name_file = 'instance_name.txt'

dataset_dir = './facedata'

dataset_train = SkinDataset()
dataset_train.load_info()
dataset_train.prepare()

dataset_val = SkinDataset()
dataset_val.load_info()
dataset_val.prepare()

print(-3)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

print(-2)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads')
#
# class InferenceConfig(skinConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#
# inference_config = InferenceConfig()
#
# # Recreate the model in inference mode
# model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
#
# # Get path to saved weights
# # Either set a specific path or find last trained weights
# # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#
#
# # model_path = model.find_last()
# model_path = './mask_rcnn_skin_0002.h5'
#
# # Load trained weights
# print("Loading weights from ", model_path)
# model.load_weights(model_path, by_name=True)
#
#
# # Test on a random image
# image_id = random.choice(dataset_val.image_ids)
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
#
# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)
#
# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))
