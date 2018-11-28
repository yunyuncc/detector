from __future__ import division
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest = "images", 
                        help="Image / Directory has images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest= 'det',
                        help="Image / Derectory to store detections to",
                        default = "det", type=str)
    parser.add_argument("--bs", dest="bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object confidence to filter predictions",
                        default=0.5)
    parser.add_argument("--nms_thresh", dest= "nms threshold",
                        help="NMS Threshold",
                        default=0.4)
    parser.add_argument("--cfg", dest = "cfgfile",help="config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default='yolov3.weights', type=str)
    parser.add_argument("--reso", dest='reso', 
                        help="Input resolution of the network, Increase to increase accuracy",
                        default="416", type=str)
    return parser.parse_args()

args = arg_parse()
print(args)
