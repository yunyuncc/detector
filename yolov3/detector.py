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
batch_size = 2
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
    start = time.time()
    #1.将图片数据复制到显存中
    if CUDA:
        batch = batch.cuda()
    #2.走前向传播
    with torch.no_grad():
        prediction = model(batch, CUDA)
    #3.解析前向传播的结果，处理后的prediction的格式为:
    # TODO [index_in_mini_batch,top_left_x, top_left_y, right_bottom_x, right_bottom_y...])
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
    end = time.time()
    print("get prediction size:", prediction.size())
    #4.显示prediction结果
    #4.1.没有检测出对象的帧就不跳过，不保存到output里面去
    if type(prediction) == int:
        for img_num, image_name in enumerate(img_name_list[i*batch_size: min((i +  1)*batch_size, len(img_name_list))]):
            img_id = i*batch_size + img_num
        continue
    
    prediction[:,0] += i*batch_size # transform the first attribute from index in mini batch to 
                                    # index in img_name_list
    #将bouding box按比例输出
    prediction[:,[1,3]] /= int(model.net_info["width"])
    prediction[:,[2,4]] /= int(model.net_info["height"])
    print("i={}, prediction[:,0]={}".format(i, prediction[:,0]))

    #5.将每个mini_batch的结果都串联到output里面去
    if not write:
        output = prediction
        write = True
    else:
        output = torch.cat((output, prediction))

    #6.print出每一帧检测出的物体
    for img_num, full_image_path in enumerate(img_name_list[i*batch_size: min((i +  1)*batch_size, len(img_name_list))]):
        img_id = i*batch_size + img_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == img_id]
        image_name = full_image_path.split("/")[-1]
        print("------------------img_id {}----------img_name {}------------------------------".format(img_id, image_name))
        print("{0:20s} predicted in {1:6.3f} seconds".format(image_name, (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")
    #7.同步等待当前mini_batch中的每个图片都处理完成,因为cuda kernel的调用是异步调用
    if CUDA:
        torch.cuda.synchronize()

    rects_tensor = prediction[:,[1,2,3,4]]

    #处理非法值
    neg_mask = rects_tensor < 0
    rects_tensor[neg_mask] = 0
    big_mask = rects_tensor > 1
    rects_tensor[big_mask] = 1

    #根据bounding box还原出矩形框
    rects_tensor[:,[0,2]] *= img_width
    rects_tensor[:,[1,3]] *= img_height

    rects = rects_tensor.cpu().int().numpy().tolist()
    for _, rect in enumerate(rects):
        cv2.rectangle(loaded_imgs[i],
                        (rect[0],rect[1]), 
                        (rect[2], rect[3]),
                        (0,255,0),2)
    
    cv2.imwrite("{}-detected.jpg".format(img_id), loaded_imgs[i])
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
print(output[:,[0,1,2,3,4]] )
#[index_in_mini_batch,top_left_x, top_left_y, right_bottom_x, right_bottom_y...]
