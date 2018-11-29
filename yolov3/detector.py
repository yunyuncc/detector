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
    parser.add_argument("--nms_thesh", dest= "nms_thesh",
                        help="NMS Threshold",
                        default=0.4)
    parser.add_argument("--cfg", dest = "cfgfile",help="config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default='weights/yolov3.weights', type=str)
    parser.add_argument("--reso", dest='reso', 
                        help="Input resolution of the network, Increase to increase accuracy",
                        default="416", type=str)
    return parser.parse_args()

def load_classes(namesfile):
    fp = open(namesfile,"r")
    names = fp.read().split("\n")[:-1]
    return names

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thesh)
start = 0
CUDA = torch.cuda.is_available()
num_classes = 80
classes = load_classes("data/coco.names")
print("images:", images)
print("batch_size:", batch_size)
print("confidence:", confidence)
print("nms_thesh:", nms_thesh)
print("CUDA:", CUDA)
print("classes:", classes)

print("Loading network......")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")
model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])

assert( inp_dim % 32 == 0)
assert( inp_dim > 32)

if CUDA:
    #Moves all model parameters and buffers to the GPU.
    print("setting network to CUDA model......")
    model.cuda()
    print("use CUDA model success")
#Sets the module in evaluation mode.
model.eval()

