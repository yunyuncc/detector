from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
#from torch.autograd import Variable (deprecated)
import numpy as np

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img,(416,416))

    img_ = img[:,:,::-1].transpose((2,0,1)) #BGR->RGB | HXWC->CXHXW
    img_ = img_[np.newaxis,:,:,:]/255.0 # add a channel at 0 for batch and normalise
    img_ = torch.from_numpy(img_).float()
    return img_

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(cfg_file):
    """
    Deal with cfg file
    Returns a list of blocks,each block describes a network to be build.
    block is a dictionary
    """
    file = open(cfg_file, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks
############handles############
def handle_upsample(
                block, 
                index, 
                net_info, 
                output_filters,
                module_list,
                prev_filters
                ):
    module = nn.Sequential()
    stride = int(block["stride"])
    upsample = nn.Upsample(scale_factor=stride, mode="nearest")#this is 2
    module.add_module("upsample_{}".format(index), upsample)
    module_list.append(module)


def handle_convolutional(block, 
                index, 
                net_info, 
                output_filters,
                module_list, 
                prev_filters):
    module = nn.Sequential()#each block may has muti layers
    activation = block["activation"]
    try:
        batch_normalize = int(block["batch_normalize"])
        bias = False
    except:
        batch_normalize = 0
        bias = True
    filters = int(block["filters"])
    padding = int(block["pad"])
    kernel_size = int(block["size"])
    stride = int(block["stride"])

    if padding:
        pad = (kernel_size-1)//2
    else:
        pad = 0
    #add conv layer
    prev_filter = prev_filters[0]
    conv = nn.Conv2d(prev_filter, filters, kernel_size, stride, pad, bias = bias)
    module.add_module("conv_{0}".format(index), conv)

    #add the batch norm layer
    if batch_normalize:
        bn = nn.BatchNorm2d(filters)# TODO what is batch norm????
        module.add_module("batch_norm_{0}".format(index), bn)
    #add the activation    
    if activation == "leaky":
        activn = nn.LeakyReLU(0.1, inplace=True)
        module.add_module("leaky_{0}".format(index), activn)
    prev_filters[0] = filters
    module_list.append(module)
    
def handle_route(block, 
                index, 
                net_info, 
                output_filters,
                module_list, 
                prev_filters):
    module = nn.Sequential()
    block["layers"] = block["layers"].split(',')
    start = int(block["layers"][0])
    try:
        end = int(block["layers"][1])
    except:
        end = 0
    if start > 0:
        start = start - index
    if end > 0:
        end = end - index
    route = EmptyLayer()
    module.add_module("route_{0}".format(index), route)
    if end < 0:
        prev_filters[0] = output_filters[index + start] + output_filters[index + end]
    else:
        prev_filters[0] = output_filters[index + start]
    module_list.append(module)
    
def handle_shortcut(block, 
                index, 
                net_info, 
                output_filters,
                module_list, 
                prev_filters):
    module = nn.Sequential()
    shortcut = EmptyLayer()
    module.add_module("shortcut_{}".format(index), shortcut)
    module_list.append(module)
def handle_yolo(block, 
                index, 
                net_info, 
                output_filters,
                module_list, 
                prev_filters):
    module = nn.Sequential()
    mask = block["mask"].split(",")
    mask = [int(x) for x in mask]
    anchors = block["anchors"].split(",")
    anchors = [int(a) for a in anchors]
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask]

    detection = DetectionLayer(anchors)
    module.add_module("Detection_{}".format(index), detection)
    module_list.append(module)

block_handlers = {
'convolutional':handle_convolutional,
'upsample':handle_upsample,
'route':handle_route,
'shortcut':handle_shortcut,
'yolo':handle_yolo
}

def create_modules_from_blocks(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = [3] #前一层的filter的数量
    output_filters = [] #所有层的filter的数量
    for index, block in enumerate(blocks[1:]):
        block_type = block["type"]
        #try:
        if block_type in block_handlers:
            try:
                block_handlers[block_type](
                    block, 
                    index, 
                    net_info, 
                    output_filters,
                    module_list, 
                    prev_filters)
                
            except Exception as err:
                print("call handler_", block_type, " has except:", err)
        else:
            print("has not handler_", block_type)
        output_filters.append(prev_filters[0])
    print("len of output_filters = ", len(output_filters))
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules_from_blocks(self.blocks)
    
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {} #cache the outputs for the router layer

        write = 0 #TODO
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type =="convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                #Get the input dim
                input_dim = int(self.net_info["height"])

                num_classes = int(module["classes"])
                #transform
                x = x.data
                print("---,", x.size())#last lay x.size is torch.Size([1, 255, 52, 52])
                x = predict_transform(x, input_dim,anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections,x), 1)

            outputs[i] = x
        return detections

###############test################

#blocks = parse_cfg("./cfg/yolov3.cfg")
#net_info , module_list = create_modules_from_blocks(blocks)
model = Darknet("./cfg/yolov3.cfg")
inp = get_test_input()
pred = model(inp, torch.cuda.is_available()) # TODO check why cuda can not use
#pred = model(inp, False)
print(pred)