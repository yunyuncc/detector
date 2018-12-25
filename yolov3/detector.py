from __future__ import division
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from util import *
import argparse
import os

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




def create_model(args, CUDA):

    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)

    if CUDA:
        #Moves all model parameters and buffers to the GPU.
        model.cuda()
    #Sets the module in evaluation mode.
    model.eval()
    return model

args = arg_parse()
images_dir = "/home/wyy/pytorch/my/detector/yolov3/imgs"
batch_size = 3
confidence = 0.7
nms_thesh = 0.6

img_width = 640
img_height = 480

start = 0
CUDA = torch.cuda.is_available()
num_classes = 80
classes = load_classes("data/coco.names")
model = create_model(args, CUDA)
assert(int(model.net_info["height"]) == int(model.net_info["width"]))
inp_dim = int(model.net_info["height"])
assert( inp_dim % 32 == 0)
assert( inp_dim > 32)

read_dir_time = time.time()
img_name_list = get_num_img_names(images_dir, 500)

if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch_time = time.time()
loaded_imgs = [cv2.imread(x) for x in img_name_list]
# change all cv::Mat to Tensor([1, 3, 416, 416]) 
img_batches = list(map(prepare_image, loaded_imgs, [img_name for img_name in img_name_list],[inp_dim for x in range(len(loaded_imgs))]))

img_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs]

img_dim_list = torch.FloatTensor(img_dim_list).repeat(1,2)

if CUDA:
    img_dim_list = img_dim_list.cuda()

#create batches
img_batches = create_batch(img_batches,batch_size)

write = False
start_det_loop_time = time.time()

#分批迭代所有的打包好的img
for i, batch in enumerate(img_batches):
    print("detecting......", i)
    start = time.time()
    #1.将图片数据复制到显存中
    if CUDA:
        batch = batch.cuda()
    #2.走前向传播
    with torch.no_grad():
        prediction = model(batch, CUDA)
    #3.解析前向传播的结果，处理后的prediction的格式为:
    # index_in_mini_batch,
    # top-left-x,
    # top-left-y,
    # bottom-right-x,
    # bottom-right-y,
    # object_score,
    # max-class-score,
    # max-class-score-idx
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
    end = time.time()
    #4.prediction为空
    if type(prediction) == int:
        print("has no detection")
        continue
    #5.将bouding box按转换成比例
    prediction[:,[1,3]] /= int(model.net_info["width"])
    prediction[:,[2,4]] /= int(model.net_info["height"])


    #6.同步等待当前mini_batch中的每个图片都处理完成,因为cuda kernel的调用是异步调用
    if CUDA:
        torch.cuda.synchronize()

    rects_tensor = prediction[:,[1,2,3,4]]
    labels_tensor = prediction[:,[0,5,6,7]]

    #处理非法值
    neg_mask = rects_tensor < 0
    rects_tensor[neg_mask] = 0
    big_mask = rects_tensor > 1
    rects_tensor[big_mask] = 1

    #根据bounding box还原出矩形框
    rects_tensor[:,[0,2]] *= img_width
    rects_tensor[:,[1,3]] *= img_height

    rects = rects_tensor.cpu().int().numpy().tolist()
    labels = labels_tensor.cpu().numpy().tolist()
    for k, rect in enumerate(rects):
        
        #提取出detection result
        top_left = (rect[0],rect[1])
        bottum_right = (rect[2], rect[3])
        color = (0,255,0)
        obj_score = labels[k][1]
        class_score = labels[k][2]
        class_label = classes[int(labels[k][3])]
        img_id = i*batch_size + int(labels[k][0])

        cv2.rectangle(loaded_imgs[img_id],
                        top_left, 
                        bottum_right,
                        color,
                        1)
        cv2.putText(loaded_imgs[img_id],class_label, top_left,cv2.FONT_HERSHEY_PLAIN,1,color, 1)
    
    cv2.imwrite("{}-detected.jpg".format(i), loaded_imgs[i])
try:
    output
except NameError:
    print("No detection result")
    exit()

#将img_dim_list的len 变成和output的len一样长
img_index = output[:,0].long()
img_dim_list = torch.index_select(img_dim_list, 0, img_index)

scaling_factor = torch.min(inp_dim/img_dim_list, dim=1, keepdim=True)[0].view(-1, 1)
print("output.size=", output.size())
print(output)
#[index_in_mini_batch,top_left_x, top_left_y, right_bottom_x, right_bottom_y...]
