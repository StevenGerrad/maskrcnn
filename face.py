import os
import sys
import random
import math
import re
import time
import numpy as np
import labelme
import base64
import io
import cv2
import matplotlib
import matplotlib.pyplot as plt
import PIL
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import json
import keras
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
print(MODEL_DIR, COCO_MODEL_PATH)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"]= '0'  # 使用0号gpu（想使用其他编号GPU，对应修改引号中的内容即可）

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# train config
class FacesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "faces"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = FacesConfig()
# config.display()

# dataset
class FacesDataset(utils.Dataset):
    def __init__(self, dataset_type='train'):
        super(FacesDataset, self).__init__()

        self.DATA_ROOT_DIR = './facedata'

        # Add classes
        self.add_class("faces", 1, "opencomedo")
        self.add_class("faces", 2, "closedcomedo")
        self.add_class("faces", 3, "papule")
        self.add_class("faces", 4, "nudule")
        self.add_class("faces", 5, "scar")
        self.add_class("faces", 6, "undefined")

        all_label_path_list = os.listdir(self.DATA_ROOT_DIR)
        data_len = len(all_label_path_list)

        if dataset_type=='train':
            label_path_list = all_label_path_list[:int(0.8*data_len)]
        elif  dataset_type=='val':
            label_path_list = all_label_path_list[int(0.8*data_len):]
        else:
            raise NotImplementedError

        # 将所有信息放到Image info 中
        for i, label_path in enumerate(label_path_list):
            self.add_image("faces", image_id=i, path=label_path)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        label_path = info['path']

        # 读取json文件
        with open(os.path.join(self.DATA_ROOT_DIR, label_path), encoding='utf-8') as json_file:
            labelmeJson = json.load(json_file)
            # height = labelmeJson['imageHeight']
            # width = labelmeJson['imageWidth']
            # shape_list = labelmeJson['shapes']
            image = self.img_b64_to_arr(labelmeJson['imageData'])
            # bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
            # image = np.ones([labelmeJson['height'], labelmeJson['width'], 3], dtype=np.uint8)
            # image = image * bg_color.astype(np.uint8)
            #
            # for shape, color, dims in info['shapes']:
            #     image = self.draw_shape(image, shape, dims, color)

            return image

    def img_b64_to_arr(self, img_b64):
        img_data = base64.b64decode(img_b64)
        f = io.BytesIO()
        f.write(img_data)
        img_pil = PIL.Image.open(f)
        img_arr = np.asarray(img_pil)
        return img_arr

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "faces":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def shape_to_mask(self, img_shape, points, shape_type=None, line_width=10, point_size=5):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        draw = PIL.ImageDraw.Draw(mask)
        xy = [tuple(point) for point in points]
        if shape_type == "circle":
            assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
            (cx, cy), (px, py) = xy
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
        elif shape_type == "rectangle":
            assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
            draw.rectangle(xy, outline=1, fill=1)
        elif shape_type == "line":
            assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == "linestrip":
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == "point":
            assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
            cx, cy = xy[0]
            r = point_size
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
        else:
            assert len(xy) > 2, "Polygon must have points more than 2"
            draw.polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        label_path = info['path']

        # 读取json文件
        with open(os.path.join(self.DATA_ROOT_DIR, label_path), encoding='utf-8') as json_file:
            labelmeJson = json.load(json_file)
            height = labelmeJson['imageHeight']
            width = labelmeJson['imageWidth']
            shapes = labelmeJson['shapes']

            count = len(shapes)
            mask = np.zeros([height, width, count], dtype=np.uint8)

            for i, shape in enumerate(shapes):
                mask[:, :, i] = self.shape_to_mask(mask.shape, shape['points'], shape['shape_type'])

            # Map class names to class IDs.
            class_ids = np.array([self.class_names.index(shape['label']) if shape['label'] in self.class_names else self.class_names.index('undefined') for shape in shapes])
            #print('class_ids:', class_ids)
            #input()
            return mask.astype(np.bool), class_ids.astype(np.int32)


print(-4)

# Training dataset
dataset_train = FacesDataset('train')
dataset_train.prepare()

# Validation dataset
dataset_val = FacesDataset('val')
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
print(-3)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

print(-2)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')

print(-1)
# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
# model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=1, layers="all")