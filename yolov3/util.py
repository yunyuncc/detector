from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os.path as osp

def predict_transform(prediction, input_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride #13 26 52
    bbox_attrs = 5 + num_classes # 85
    num_anchors = len(anchors)# 3

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    #print("[batch_size, bbox_attrs*num_anchors, grid_size**2]",batch_size, ",", bbox_attrs*num_anchors, ",", grid_size*grid_size)

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
    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

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
    # 将模型输出的tensor(batch_size * n_before * 85) 转换成可以表示最终detection结果的tensor(n_after*8)
    # n_before: 在进行非极大值抑制前的一帧当中的detection结果的数量
    # n_after:  进行非极大值抑制后的batch_size帧当中的detection结果的数量

    #将 object_score 小于confidence的全部置为0
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
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
    for ind in range(batch_size):
        image_pred = prediction[ind]
        #将prediction矩阵的每一行的信息变成如下形式
        #top-left-x,top-left-y,bottom-right-x,bottom-right-y,object_score,max-class-score,max-class-score-idx
        max_class_score, max_class_score_idx = torch.max(image_pred[:,5:5+num_classes], 1)
        max_class_score_idx = max_class_score_idx.float().unsqueeze(1)
        max_class_score = max_class_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_class_score, max_class_score_idx)
        image_pred = torch.cat(seq,1)

        #去掉所有object_score值为0的行
        non_zero_ind = (torch.nonzero(image_pred[:,4]))
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
            ###cls_mask = (image_pred_[:,-1] == cls).float().unsqueeze(1)
            ###cls_mask = image_pred_*cls_mask
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            #print("[",cls,"]image_pred_class=\n", image_pred_class) 

            #按object_score 降序排列
            object_score_sort_idx = torch.sort(image_pred_class[:,4], descending = True)[1]
            image_pred_class = image_pred_class[object_score_sort_idx]
            #print("[",cls,"] sorted image_pred_class=\n", image_pred_class) 

            #进行非极大值抑制
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
            
            #print("[", cls,"] after nms image_pred_class:", image_pred_class, image_pred_class.size())
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    ###fix this bug, 因为没有对齐tab，导致提前退出for循环
    try:
        #print("write_result output:", output[:,0])
        return output
    except:
        print("write_write no result")
        return 0

def letterbox_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    scaler = min(w/img_w, h/img_h)
    new_w = int(img_w * scaler)
    new_h = int(img_h * scaler)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    return canvas

def load_classes(namesfile):
    #load all class label
    fp = open(namesfile,"r")
    names = fp.read().split("\n")[:-1]
    return names


def prepare_image(img, img_name, inp_dim):
    #resize image to square and change it to tensor

    if img is None:
        raise ValueError("{} data is empty".format(img_name))
    #in blog, just resize
    img = cv2.resize(img, (inp_dim, inp_dim))

    #in code, add letterbox
    #img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy() #BGR->RGB | H W C -> C H W 
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    #print("prepare_image return size:", img.size())
    return img

def get_num_img_names(images_path, num):
    #get image from images_path, file name is 1.jpg 2.jpg 3.jpg ......
    img_list = []
    for i in range(num):
        name = "{}.jpg".format(i)
        full_file_path = osp.join(osp.realpath('.'), images_path, name)
        if osp.exists(full_file_path):
            img_list.append(full_file_path)
    return img_list


def create_batch(img_list_to_batch, batch_size):
    #make image list to some images_batch
    leftover = 0
    if (len(img_list_to_batch) % batch_size):
        leftover = 1

    num_batches = len(img_list_to_batch)//batch_size + leftover
    batched_img_list = [torch.cat((img_list_to_batch[i*batch_size : min((i+1)*batch_size, len(img_list_to_batch))])) for i in range(num_batches)]
    return batched_img_list


