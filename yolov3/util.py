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

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    print("pre size:", prediction.size())
    conf_mask = (prediction[:,:,4] > confidence)
    conf_mask = conf_mask.float()
    conf_mask = conf_mask.unsqueeze(2)
    prediction = prediction*conf_mask
    print(prediction.size())

    #center_x, center_y, height, width ==> top_left_x, top_left_y, right_bottom_x, right_bottom_y
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False
    for i in range(batch_size):
        image_pred = prediction[i]
        print(image_pred.size())
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes], 1)
        print("max_conf.size=", max_conf.size())
        print("max_conf_score.size=", max_conf_score.size())
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        #top-left-x,top-left-y,bottom-right-x,bottom-right-y,max-class-score-index,max-class-score
        image_pred = torch.cat(seq,1)
        print("image_pred.size=", image_pred.size())

        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        print("non-zero-ind.size=", non_zero_ind.size())
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        #For PyTorch 0.4 compatibility
        #Since the above code with not raise exception for no detection 
        #as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue 
        print("image_pred_.size=", image_pred_.size())
        print("image_pred_=", image_pred_)
        img_classes = unique(image_pred_[:,-1])#-1 is the last col,which is class label

        print("img_classes.size=", img_classes.size())
        print("img_classes=", img_classes)