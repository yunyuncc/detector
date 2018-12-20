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
from darknet import get_test_input
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
                        default=0.6)
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

def get_img_names(images_path):
    try:
        imlist = [osp.join(osp.realpath('.'), images_path, img) for img in os.listdir(images_path)]
    except NotADirectoryError:
        print("{} is not a dir".format(images_path))
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images_path))
    except FileNotFoundError:
        print("No file or dir with the name {}".format(images_path))
        exit()
    return imlist


def get_num_img_names(images_path, num):
    img_list = []
    for i in range(num):
        name = "{}.jpg".format(i)
        try:
            img_list.append(osp.join(osp.realpath('.'), images_path, name))
        except FileNotFoundError:
            print("{} file not exist break".format(name))
            break
    return img_list

def create_model(args, CUDA):

    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")
    model.net_info["height"] = args.reso

    if CUDA:
        #Moves all model parameters and buffers to the GPU.
        print("setting network to CUDA model......")
        model.cuda()
        print("use CUDA model success")
    #Sets the module in evaluation mode.
    model.eval()
    return model

args = arg_parse()
images_dir = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thesh)

start = 0
CUDA = torch.cuda.is_available()
num_classes = 80
classes = load_classes("data/coco.names")
model = create_model(args, CUDA)
inp_dim = int(model.net_info["height"])
assert( inp_dim % 32 == 0)
assert( inp_dim > 32)


read_dir_time = time.time()
img_name_list = get_num_img_names(images_dir, 100)
#print("img_name_list:",img_name_list)


if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch_time = time.time()
loaded_imgs = [cv2.imread(x) for x in img_name_list]

# change all cv::Mat to Tensor([1, 3, 416, 416]) 
img_batches = list(map(prepare_image, loaded_imgs, [inp_dim for x in range(len(loaded_imgs))]))
img_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs]

img_dim_list = torch.FloatTensor(img_dim_list).repeat(1,2)

if CUDA:
    img_dim_list = img_dim_list.cuda()

#create batches
img_batches = create_batch(img_batches,3)


write = False
start_det_loop_time = time.time()

for i, batch in enumerate(img_batches):
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    prediction = model(batch, CUDA)
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
    end = time.time()

    if type(prediction) == int:
        for img_num, image in enumerate(img_name_list[i*batch_size: min((i +  1)*batch_size, len(img_name_list))]):
            img_id = i*batch_size + img_num
            print("img_id {} has not object detected", img_id)
        continue
    
    prediction[:,0] += i*batch_size # transform the first attribute from index in mini batch to 
                                    # index in img_name_list
    if not write:
        output = prediction
        write = True
    else:
        output = torch.cat((output, prediction), dim = 0)

    for img_num, image in enumerate(img_name_list[i*batch_size: min((i +  1)*batch_size, len(img_name_list))]):
        img_id = i*batch_size + img_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == img_id]
        image_name = image.split("/")[-1]
        print("------------------img_id {}----------img_name {}------------------------------".format(img_id, image_name))
        print("{0:20s} predicted in {1:6.3f} seconds".format(image_name, (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")
    if CUDA:
        torch.cuda.synchronize()

try:
    output
except NameError:
    print("No detection result")
    exit()
print("------")
