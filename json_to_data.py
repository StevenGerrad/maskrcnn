import base64
import json
import os
import PIL
import glob
import io
import numpy as np
import uuid
import math
import imgviz

def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    img_arr = np.asarray(img_pil)
    return img_arr

def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
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

def shapes_to_label(img_shape, shapes, class_name_list, instance_list):
    ins = np.zeros(img_shape[:2], dtype=np.int32)
    for shape in shapes:
        label = shape["label"]
        points = shape["points"]
        group_id = shape.get("group_id")
        shape_type = shape.get("shape_type")
        if group_id is None:
            group_id = uuid.uuid1()

        if label in class_name_list:
            instance = (label, group_id)
            if instance not in instance_list:
                instance_list.append(instance)

            ins_id = instance_list.index(instance)
            mask = shape_to_mask(img_shape, points, shape_type)
            ins[mask] = ins_id
    return ins, instance_list

def lblsave(filename, lbl):
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)

def main():
    class_name_list = ['closedcomedo',
                       'opencomedo',
                       'papule',
                       'nudule',
                       'scar',
                       'undefined']

    root_dir = os.path.abspath('.')
    dataset_dir = os.path.join(root_dir, 'facedata')

    json_path_list = glob.glob(os.path.join(dataset_dir, '*.json'))
    for json_path in json_path_list:
        out_folder = os.path.basename(json_path).split(".")[0]
        out_dir = os.path.join(dataset_dir, out_folder)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        with open(json_path, encoding='utf-8') as json_file:
            data = json.load(json_file)

        imageData = data["imageData"]
        img = img_b64_to_arr(imageData)
        instance_list = [('_background_', 0)]
        lbl, instance_list = shapes_to_label(img.shape, data["shapes"], class_name_list, instance_list)

        PIL.Image.fromarray(img).save(os.path.join(out_dir, "img.png"))
        lblsave(os.path.join(out_dir, "mask.png"), lbl)
        with open(os.path.join(out_dir, "instance_name.txt"), "w") as f:
            for instance in instance_list:
                f.write(instance[0] + "\n")

if __name__ == "__main__":
    main()
