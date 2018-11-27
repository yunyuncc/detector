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

def bbox_iou(box1, box2):
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    print("pre size:", prediction.size())
    #将 object_score 小于confidence的全部置为0
    conf_mask = (prediction[:,:,4] > confidence)
    conf_mask = conf_mask.float()
    conf_mask = conf_mask.unsqueeze(2)
    prediction = prediction*conf_mask
    #为了非极大值抑制的计算方便，将bounding box的表示方法作以下转换
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
        #将prediction矩阵的每一行的信息变成如下形式
        #top-left-x,top-left-y,bottom-right-x,bottom-right-y,max-class-score-index,max-class-score
        max_class_score_idx, max_class_score = torch.max(image_pred[:,5:5+num_classes], 1)
        max_class_score_idx = max_class_score_idx.float().unsqueeze(1)
        max_class_score = max_class_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_class_score_idx, max_class_score)
        image_pred = torch.cat(seq,1)

        #去掉所有object_score值为0的行
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


        img_classes = unique(image_pred_[:,-1])#-1 is the last col,which is class label

        #对每个不同的class进行非极大值抑制
        for cls in img_classes:
            #perform NMS
            #单次获取每个class的所有的行
            cls_mask = (image_pred_[:,-1] == cls).float().unsqueeze(1)
            cls_mask = image_pred_*cls_mask
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            #print("[",cls,"]image_pred_class=\n", image_pred_class) 

            #按object_score 降序排列
            object_score_sort_idx = torch.sort(image_pred_class[:,4], descending = True)[1]
            image_pred_class = image_pred_class[object_score_sort_idx]
            #print("[",cls,"] sorted image_pred_class=\n", image_pred_class) 

            idx = image_pred_class.size(0)

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                #zero all the detections that have iou > treshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                #remove the zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
            
            print("[", cls,"] after nms image_pred_class:", image_pred_class)

