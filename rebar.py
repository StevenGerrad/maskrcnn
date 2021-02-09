import os
import sys
import datetime
import numpy as np
import skimage.draw
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import pickle
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DATA_DIR = './data/'


############################################################
#  Configurations
############################################################


class RebarConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "rebar"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + rebar

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    GPU_COUNT = 1

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 512

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 5


############################################################
#  Dataset
############################################################

class RebarDataset(utils.Dataset):

    def load_rebar(self, dataset_dir, subset):
        """Load a subset of the Rebar dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("rebar", 1, "rebar")

        file_list = os.listdir(dataset_dir + subset)
        for file in file_list:
            if file.split('.')[-1] == 'jpg':
                image_id = file
                image_path = dataset_dir + subset + '/' + file
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                # 读取坐标数据
                polygons = pickle.load(open(dataset_dir + subset + '/' + file.split('.')[0] + '.val', 'rb'))

                self.add_image(
                    "rebar",
                    image_id=file,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a rebar dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "rebar":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "rebar":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = RebarDataset()
    dataset_train.load_rebar(DATA_DIR, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RebarDataset()
    dataset_val.load_rebar(DATA_DIR, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')


def fine_tune(model):
    dataset_train = RebarDataset()
    dataset_train.load_rebar(DATA_DIR, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RebarDataset()
    dataset_val.load_rebar(DATA_DIR, "val")
    dataset_val.prepare()

    print("fine_tune network all")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers="all")


config = RebarConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)
weights_path = COCO_WEIGHTS_PATH
model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)


fine_tune(model)