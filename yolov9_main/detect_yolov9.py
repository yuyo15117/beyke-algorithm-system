import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from yolov9_main.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov9_main.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov9_main.utils.plots import Annotator, colors, save_one_box
from yolov9_main.utils.torch_utils import select_device, smart_inference_mode
from yolov9_main.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)

class YOLOV9_detect:

    def __init__(self,image):
        self.image = image

    def detect_image(self):

        weights = "v9_trained_weights/yolov9_decay_weight.pt"
        # Load model
        device = select_device('0')
        imgsz = (1500, 1500)
        model = DetectMultiBackend(weights, device=device, dnn=False, data="v9_trained_weights/data.yaml", fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size


        # Read image
        im0 = self.image  # BGR

        im = letterbox(im0, imgsz, stride=stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        # with dt[1]:
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True)
        pred = model(im, augment=False, visualize=False)

        conf_thres =0.21
        iou_thres = 0.45
        # NMS
        # with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, max_det=1000)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # s += '%gx%g /' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            location = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()


                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)
                    location.append(line)

            return location









