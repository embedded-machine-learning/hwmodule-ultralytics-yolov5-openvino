#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Futures
from __future__ import print_function

# Built-in/Generic Imports
import sys, os, json, argparse
import logging
import time

# Libs
import numpy as np
import pandas as pd
from openvino.inference_engine import IECore
import cv2

__author__ = "Matvey Ivanov"
__copyright__ = (
    "Copyright 2021, Christian Doppler Laboratory for " "Embedded Machine Learning"
)
__credits__ = ["Matvey Ivanov"]
__license__ = "ISC"
__version__ = "0.1.0"
__maintainer__ = "Matvey Ivanov"
__email__ = "matvey.ivanov@tuwien.ac.at"
__status__ = "Experimental"

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self,  side):
        self.num = 3 #if 'num' not in param else int(param['num'])
        self.coords = 4 #if 'coords' not in param else int(param['coords'])
        self.classes = 80 #if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] #if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        #self.isYoloV3 = False

        #if param.get('mask'):
        #    mask = [int(idx) for idx in param['mask'].split(',')]
        #    self.num = len(mask)

        #    maskedAnchors = []
        #    for idx in mask:
        #        maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
        #    self.anchors = maskedAnchors

        #    self.isYoloV3 = True # Weak way to determine but the only one.

    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]

def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    w, h = size

    # Scale ratio (new / old)
    r = min(h / shape[0], w / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / shape[1], h / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0
    if img.shape[0] != h:
        top2 = (h - img.shape[0])//2
        bottom2 = top2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1])//2
        right2 = left2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, resized_im_h=640, resized_im_w=640):
    gain = min(resized_im_w / im_w, resized_im_h / im_h)  # gain  = old / new
    pad = (resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2  # wh padding
    x = int((x - pad[0]) / gain)
    y = int((y - pad[1]) / gain)

    w = int(width / gain)
    h = int(height / gain)

    xmin = max(0, int(x - w / 2))
    ymin = max(0, int(y - h / 2))
    xmax = min(im_w, int(xmin + w))
    ymax = min(im_h, int(ymin + h))
    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())

def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    out_blob_n, out_blob_c, out_blob_h, out_blob_w = blob.shape
    predictions = 1.0 / (1.0 + np.exp(-blob))

    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()

    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    bbox_size = int(out_blob_c / params.num)  # 4+1+num_classes

    for row, col, n in np.ndindex(params.side, params.side, params.num):
        bbox = predictions[0, n * bbox_size:(n + 1) * bbox_size, row, col]

        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]
        if object_probability < threshold:
            continue
        x = (2 * x - 0.5 + col) * (resized_image_w / out_blob_w)
        y = (2 * y - 0.5 + row) * (resized_image_h / out_blob_h)
        if int(resized_image_w / out_blob_w) == 8 & int(resized_image_h / out_blob_h) == 8:  # 80x80,
            idx = 0
        elif int(resized_image_w / out_blob_w) == 16 & int(resized_image_h / out_blob_h) == 16:  # 40x40
            idx = 1
        elif int(resized_image_w / out_blob_w) == 32 & int(resized_image_h / out_blob_h) == 32:  # 20x20
            idx = 2

        width = (2 * width) ** 2 * params.anchors[idx * 6 + 2 * n]
        height = (2 * height) ** 2 * params.anchors[idx * 6 + 2 * n + 1]
        class_id = np.argmax(class_probabilities)
        confidence = object_probability
        objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                  im_h=orig_im_h, im_w=orig_im_w, resized_im_h=resized_image_h,
                                  resized_im_w=resized_image_w))
    return objects

def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCS2 settings test")
    parser.add_argument("-m", "--model_path", default="./model.xml", help="model to test with", type=str,required=False,)
    parser.add_argument( "-i", "--image_dir", default="./images", help="images for the inference", type=str,required=False,)
    parser.add_argument("-d",  "--device",  default="CPU",  help="target device to run the inference [CPU, MYRIAD]",
                        type=str,  required=False,)
    parser.add_argument("-out", '--detections_out', default='detections.csv', help='Output file detections', type=str, required=False)
    parser.add_argument("-is", '--input_source', default='tf2oda', help='Model Source Framework [onnx, tf2oda]', type=str, required=False)
    parser.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    parser.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    parser.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--no-show', dest='show', action='store_false')
    parser.set_defaults(yolo=False)

    args = parser.parse_args()
    print(args)


    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    if not os.path.isfile(args.model_path):
        sys.exit("Invalid model xml given!")

    model_xml = args.model_path
    model_bin = args.model_path.split(".xml")[0] + ".bin"

    if not os.path.isfile(model_xml) or not os.path.isfile(model_bin):
        sys.exit("Could not find IR model for: " + model_xml)

    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    print("Loaded model: {}, weights: {}".format(model_xml, model_bin))

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    in_blob = net.input_info[input_blob].input_data.shape
    net.input_info[input_blob].precision = "U8"
    net.batch_size = 1

    print("Loading network and perform on:", args.device)
    exec_net = ie.load_network(network=net, device_name=args.device, num_requests=1)

    combined_data = []
    _, _, net_h, net_w = net.input_info[input_blob].input_data.shape

    for filename in os.listdir(args.image_dir):
        total_latency_start_time = time.time()
        original_image = cv2.imread(os.path.join(args.image_dir, filename))
        image = letterbox(original_image.copy(), (net_w, net_h))

        if image.shape[:-1] != (net_h, net_w):
            log.info(f"Image {args.image_dir} is resized from {image.shape[:-1]} to {(net_h, net_w)}")
            image = cv2.resize(image, (net_w, net_h))

        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        print("\n\nStarting inference for picture:" + filename)
        start_time = time.time()
        output = exec_net.infer(inputs={input_blob: image})
        print("Inference time: {:.3f} ms".format((time.time() - start_time) * 1e3))

        h, w, _ = original_image.shape
        objects = list()

        for layer_name, out_blob in exec_net.requests[0].output_blobs.items():
            layer_params = YoloParams(side=out_blob.buffer.shape[2])
            print("Layer {} parameters: ".format(layer_name))
            layer_params.log_params()
            objects += parse_yolo_region(out_blob.buffer, image.shape[2:],
                                         # in_image.shape[2:], layer_params,
                                         original_image.shape[:-1], layer_params,
                                         args.prob_threshold)

        # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
        objects = sorted(objects, key=lambda obj: obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if intersection_over_union(objects[i], objects[j]) > args.iou_threshold:
                    objects[j]['confidence'] = 0

        # Drawing objects with respect to the --prob_threshold CLI parameter
        objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold]

        print("\n********************THESE ARE THE RECOGNIZED OBJECTS********************\n")
        for o in objects:
            print(o)

        # the objects can now be shown in a frame
        if args.show:
            # show frame with bounding boxes
            origin_im_size = original_image.shape[:-1]

            for obj in objects:
                # Validation bbox of detected object
                if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj[
                    'ymin'] < 0:
                    continue
                color = (int(min(obj['class_id'] * 12.5, 255)),
                         min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
                det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                    str(obj['class_id'])

                cv2.rectangle(original_image, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
                cv2.putText(original_image,
                            "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                            (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

            # uncomment the next line for better overview on a 1080p display
            #cv2.imshow("DetectionResults", cv2.resize(original_image, dsize=(1500,1000)))
            cv2.imshow("DetectionResults", original_image)

            key = cv2.waitKey(0)
            # ESC key
            if key == 27:
                break
