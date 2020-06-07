"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------"""
import os
import sys
import json
import numpy as np
import skimage.draw
import argparse
import matplotlib.pyplot as plt

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
# from mrcnn.model import MaskRCNN, log
from mrcnn import utils

# Weights path for unfollow_weights.h5 "r" may be removed depending on system
UNFOLLOW_WEIGHTS_PATH = r"mask\logs\unfollow\unfollow_weights.h5"
dataset_dir = r'button'
# Image needs to be updated...
IMAGE = r'button\val\24.png'
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class ModelConfiguration(Config):
    """Base configuration class. CHANGE NUMBER OF CLASSES (NUM_CLASSES)
    You can leave most of the values as defualt but if you wanted to try
    and improve accuracy or increase time it is available by tweaking values.
    """
    # give the configuration a recognizable name
    NAME = "unfollow_model"

    # number of classes ( add +1 for the background (BG))
    NUM_CLASSES =   2             #NUMBER OF CLASSES!!!!

    # gpu count
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
   
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

   
    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_INFERENCE = 1000

    # Input image resizing
    # Generally, use the "square" resizing mode for predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none, square, pad64, crop
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

config = ModelConfiguration()
config.display()  


############################################################
#  Dataset
############################################################

class Model_Dataset(utils.Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, subset):
        # Add classes. .add_class(model name, class id number, name of class)
        # MUST FILL IN ###### AS CLASS NAME
        self.add_class("unfollow_model", 1, 'unfollow')
     
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # dictionary of x and y coordinates of each region and region class name
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values()) 
        
        # decrease dimensions within annotations
        annotations = [a for a in annotations if a['regions']]
      
        # Add images
        for a in annotations:
            # dependant on VIA version
            if type(a['regions']) is dict:
                # polygons are bboxes and objects are the class name 
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                objects = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
                objects = [r['region_attributes'] for r in a['regions']]

            # check to see if report and more line up with appropriate id
            num_ids = [list(n.values()) for n in objects]
            num_ids = [1]
            # NUMBER IDS MUST BE CHANGED IF USING <2 OR >2
            
            # load_mask() needs the image size to convert polygons to masks.
            # Not provided in annotation json
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            # loading the dataset with image information to be used in load_mask()
            self.add_image(
                "unfollow_model",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                num_ids=num_ids,
                width=width, height=height,
                polygons=polygons)
            
        
    def load_mask(self, image_id):
        # obtains info for each image in dataset
        info = self.image_info[image_id]
    
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"],
                         info["width"],
                         len(info["polygons"])],
                         dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # one makes the transparent mask
            mask[rr, cc, i] = 1
        # Map class names to class IDs.
        num_ids = info['num_ids']
        num_ids = np.array(num_ids, dtype=np.int32)
        
        return mask.astype(np.bool), num_ids
                 
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "unfollow_model":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Command line
############################################################

if __name__ == '__main__':

#    # Parse command line arguments
#    parser = argparse.ArgumentParser(
#        description='Train Mask R-CNN to detect unfollow button on twitter.')
#    parser.add_argument('--dataset', required=False,
#                        metavar= dataset_dir,
#                        help='Only val dataset available')
#    parser.add_argument('--weights', required = False,
#                        metavar = UNFOLLOW_WEIGHTS_PATH ,
#                        help="Path to weights .h5 file, only weights_unfollow.h5 available")
#    parser.add_argument('--logs', required=False,
#                        default = DEFAULT_LOGS_DIR,
#                        metavar="/path/to/logs/",
#                        help='Logs and checkpoints directory (default=logs/)')
#   # IMAGE may be required change to True
#    parser.add_argument('--image', required=False,
#                        metavar="path or URL to image",
#                        help='Image to apply the color splash effect on')
#    args = parser.parse_args()
#
#    print("Weights: ", args.weights)
#    print("Dataset: ", args.dataset)
#    print("Logs: ", args.logs)

    # Configurations
    class InferenceConfig(ModelConfiguration):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        
    config = InferenceConfig()
    # can be removed to not show configuration 
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", 
                              config=config,
                              model_dir=DEFAULT_LOGS_DIR)
     
    
    # Load weights
    print("Loading weights ", UNFOLLOW_WEIGHTS_PATH)
    
    model.load_weights(UNFOLLOW_WEIGHTS_PATH, by_name=True)
    
    def get_ax(rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.
        
        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
    
    # Load dataset
    dataset = Model_Dataset()
    dataset.load_dataset(dataset_dir, subset = 'val')
    # Must call before using the dataset
    dataset.prepare()
         
    # run detection
    image = skimage.io.imread(IMAGE)
    # Remove alpha channel, if it has one
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    # Run object detection
    results = model.detect([image], verbose=0)
    
    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    
    #Extract the first bbox
    print (r['rois'][0])
    # has the format of [y1, x1, y2, x2]
#############################################################################
        # Evaluation/ Inference
#############################################################################
# Load dataset
dataset = Model_Dataset()
dataset.load_dataset(dataset_dir, subset = 'val')
# Must call before using the dataset
dataset.prepare()
     
class InferenceConfig(ModelConfiguration):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()



# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=UNFOLLOW_WEIGHTS_PATH)

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

print("Loading weights from ", UNFOLLOW_WEIGHTS_PATH)
model.load_weights(UNFOLLOW_WEIGHTS_PATH, by_name=True)

# run detection
image = skimage.io.imread(IMAGE)
# Remove alpha channel, if it has one
if image.shape[-1] == 4:
    image = image[..., :3]

# Run object detection
results = model.detect([image], verbose=0)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")

#Extract the first bbox
print (r['rois'][0])
# has the format of [y1, x1, y2, x2]