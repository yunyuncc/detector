import torch
import torch.nn as nn
import numpy as np
import cv2
from util import *
import os

from darknet import Darknet

class detector:
    def __init__(self, cfg_file, weight_file, classes_file, CUDA=torch.cuda.is_available(), batch_size = 1, obj_thresh=0.85, nms_thresh=0.6):
        self.model = Darknet(cfg_file)
        self.model.load_weights(weight_file)
        if CUDA:
            self.model.cuda()
        self.model.eval()
        assert(int(self.model.net_info["height"]) == int(self.model.net_info["width"]))
        self.__inp_dim = int(self.model.net_info["height"])
        #保证图片可以分割为一个个grad
        assert( self.__inp_dim % 32 == 0)
        assert( self.__inp_dim > 32)

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

    def __post_process(self,prediction, confidence, num_classes, nms_conf = 0.4):
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



    def detect(self, images):
        #images is a list, hold the result of imread 
        if len(images) != self.__batch_size:
            raise ValueError("len of images should equal batch_size")
        # change all cv::Mat to Tensor([1, 3, inp_dim, inp_dim]) 
        img_batches = list(map(self.__prepare_image, images, [self.__inp_dim for x in range(len(images))]))
        img_batches = self.__create_batch(img_batches,self.__batch_size)
        assert(len(img_batches) == 1)
        batch = img_batches[0]
        #1.将图片数据复制到显存中
        if self.__cuda:
            batch = batch.cuda()
        #2.走前向传播
        with torch.no_grad():
            prediction = self.model(batch, self.__cuda)
        prediction = self.__post_process(prediction, self.__obj_thresh, self.__num_classes, nms_conf = self.__nms_thresh)
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

if __name__ == '__main__':
    images_dir = "/home/wyy/pytorch/my/detector/yolov3/imgs"
    det_dir = "det_dir/"
    img_width = 640
    img_height = 480

    batch_size = 1
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

