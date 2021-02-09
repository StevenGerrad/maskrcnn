
"""
数据集读取(不含数据强化)

"""


import json
import numpy as np
import os
import base64
import io
import math
import PIL

import labelme.utils as lu

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
            # label_path_list = all_label_path_list[:int(0.8*data_len)]
            label_path_list = all_label_path_list[:5]
        elif  dataset_type=='val':
            # label_path_list = all_label_path_list[int(0.8*data_len):]
            label_path_list = all_label_path_list[6:8]
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


if __name__ == '__main__':
    
    # Training dataset
    dataset_train = FacesDataset('train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FacesDataset('val')
    dataset_val.prepare()