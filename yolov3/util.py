from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

def predict_transform(prediction, input_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    print("stride:", stride)
    grid_size = input_dim // stride #13 26 52
    print("prediction.size(2)=", prediction.size(2), "  grid_size=", grid_size)
    bbox_attrs = 5 + num_classes # 85
    num_anchors = len(anchors)# 3
    print("input dim:", input_dim) #416

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    print("[batch_size, bbox_attrs*num_anchors, grid_size**2]",batch_size, ",", bbox_attrs*num_anchors, ",", grid_size*grid_size)
    print(prediction.size())

    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #sigmoid the center_x, center_y and object confidence
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    if CUDA:
        x_y_offset = x_y_offset.cpu()
    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    print("------------anchors:", anchors)
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    print("------------anchors2:", anchors)
    if CUDA:
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors.cpu()
    else:
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    #apply sigmoid to the class scores
    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))

    prediction[:,:,:4] *= stride

    return prediction