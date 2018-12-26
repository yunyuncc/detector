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

class detector:
    def __init__(self, cfg_file, weight_file, classes_file, CUDA=torch.cuda.is_available(), batch_size = 1, obj_thresh=0.85, nms_thresh=0.6):
        self.model = Darknet(cfg_file)
        self.model.load_weights(weight_file)
        if CUDA:
            self.model.cuda()
        self.model.eval()
        assert(int(self.model.net_info["height"]) == int(self.model.net_info["width"]))
        inp_dim = int(self.model.net_info["height"])
        #保证图片可以分割为一个个grad
        assert( inp_dim % 32 == 0)
        assert( inp_dim > 32)

        self.__classes_file = classes_file
        self.__cuda = CUDA
        self.__batch_size = batch_size
        self.__obj_thresh = obj_thresh
        self.__nms_thresh = nms_thresh
        self.__classes = self.__load_classes()
        self.__num_classes = len(self.__classes)


    def __load_classes(self):
        #load all class label
        fp = open(self.__classes_file,"r")
        names = fp.read().split("\n")[:-1]
        return names
    

    def __prepare_image(self, img, inp_dim):
        #resize image to square and change it to tensor
        if img is None:
            raise ValueError("img data is empty")
        img = cv2.resize(img, (inp_dim, inp_dim))
        img = img[:,:,::-1].transpose((2,0,1)).copy() #BGR->RGB | H W C -> C H W 
        img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
        return img
    
    def __create_batch(self, tensor_list, batch_size):
        #make image list to some images_batch
        leftover = 0
        if (len(tensor_list) % batch_size):
            leftover = 1

        num_batches = len(tensor_list)//batch_size + leftover
        batched_img_list = [torch.cat((tensor_list[i*batch_size : min((i+1)*batch_size, len(tensor_list))])) for i in range(num_batches)]
        return batched_img_list

    def detect(self, images):
        #images is a list, hold the result of imread 
        if len(images) != self.__batch_size:
            raise ValueError("len of images should equal batch_size")
        # change all cv::Mat to Tensor([1, 3, inp_dim, inp_dim]) 
        img_batches = list(map(self.__prepare_image, images, [inp_dim for x in range(len(images))]))
        img_batches = self.__create_batch(img_batches,batch_size)
        assert(len(img_batches) == 1)
        batch = img_batches[0]
        #1.将图片数据复制到显存中
        if self.__cuda:
            batch = batch.cuda()
        #2.走前向传播
        with torch.no_grad():
            prediction = self.model(batch, self.__cuda)
        prediction = write_results(prediction, self.__obj_thresh, self.__num_classes, nms_conf = self.__nms_thresh)
        if type(prediction) == int:
            return[]
        #3.将bouding box按转换成比例
        prediction[:,[1,3]] /= int(self.model.net_info["width"])
        prediction[:,[2,4]] /= int(self.model.net_info["height"])


        #4.同步等待当前mini_batch中的每个图片都处理完成,因为cuda kernel的调用是异步调用
        if self.__cuda:
            torch.cuda.synchronize()

        rects_tensor = prediction[:,[1,2,3,4]]
        labels_tensor = prediction[:,[0,5,6,7]]

        #处理非法值
        neg_mask = rects_tensor < 0
        rects_tensor[neg_mask] = 0
        big_mask = rects_tensor > 1
        rects_tensor[big_mask] = 1


        rects = rects_tensor.cpu().float().numpy().tolist()
        labels = labels_tensor.cpu().numpy().tolist()
        detect_ress = []
        for k, rect in enumerate(rects):    
            #提取出detection result
            top_left = (rect[0],rect[1])
            bottom_right = (rect[2], rect[3])
            obj_score = labels[k][1]
            class_score = labels[k][2]
            batch_idx = int(labels[k][0])
            class_label = self.__classes[int(labels[k][3])]
            detect_res = {  
                            "top_left":top_left,
                            "bottom_right":bottom_right,
                            "obj_score":obj_score,
                            "class_score":class_score,
                            "class_label":class_label,
                            "batch_idx":batch_idx
                         }
            detect_ress.append(detect_res)
        return detect_ress


images_dir = "/home/wyy/pytorch/my/detector/yolov3/imgs"
det_dir = "det_dir/"
img_width = 640
img_height = 480

batch_size = 2
confidence = 0.85
nms_thesh = 0.6
cfgfile = "cfg/yolov3.cfg"
weightsfile = "weights/yolov3.weights"
classes_file = "data/coco.names"
CUDA = torch.cuda.is_available()

detector_ = detector(cfgfile, weightsfile, classes_file, CUDA, batch_size, confidence, nms_thesh)
inp_dim = int(detector_.model.net_info["height"])

img_name_list = get_num_img_names(images_dir, 500)

if not os.path.exists(det_dir):
    os.makedirs(det_dir)

loaded_imgs = [cv2.imread(x) for x in img_name_list]
for i in range(0, len(loaded_imgs), batch_size):
    print("detect batch i=", i)
    detect_ress = detector_.detect(loaded_imgs[i:i+batch_size])
    for _, detect_res in enumerate(detect_ress):
        top_left_one = detect_res["top_left"]
        bottom_right_one = detect_res["bottom_right"]
        top_left = (int(top_left_one[0]*img_width), int(top_left_one[1]*img_height))
        bottom_right = (int(bottom_right_one[0]*img_width), int(bottom_right_one[1]*img_height))

        obj_score = detect_res["obj_score"]
        class_score = detect_res["class_score"]
        class_label = detect_res["class_label"]
        batch_idx = detect_res["batch_idx"]

        img_id = i + batch_idx
        color = (0,255,0)
        cv2.rectangle(loaded_imgs[img_id],
                        top_left, 
                        bottom_right,
                        color,
                        1)
        info = "[{}({:.3f})]obj:{:.3f}".format(class_label, class_score, obj_score)
        cv2.putText(loaded_imgs[img_id],info, top_left,cv2.FONT_HERSHEY_PLAIN,1,color, 1)
    for j in range(0, batch_size):
        cv2.imwrite("{}{}-detected.jpg".format(det_dir,i+j), loaded_imgs[i+j])

