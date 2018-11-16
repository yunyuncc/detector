from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable (deprecated)
import numpy as np

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
def handle_upsample(block, 
                index, 
                net_info, 
                module_list, 
                prev_filters, 
                output_filters):
    print('handle_upsample')
def handle_convolutional(block, 
                index, 
                net_info, 
                module_list, 
                prev_filters, 
                output_filters):
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
        bn = nn.BatchNorm2d(filters)
        module.add_module("batch_norm_{0}".format(index), bn)
    #add the activation    
    if activation == "leaky":
        activn = nn.LeakyReLU(0.1, inplace=True)
        module.add_module("leaky_{0}".format(index), activn)
    module_list.append(module)
    prev_filters[0] = filters
    output_filters.append(filters)
    
def handle_route(block, 
                index, 
                net_info, 
                module_list, 
                prev_filters, 
                output_filters):
    print('handle_route')
def handle_shortcut(block, 
                index, 
                net_info, 
                module_list, 
                prev_filters, 
                output_filters):
    print('handle_shortcut')
def handle_yolo(block, 
                index, 
                net_info, 
                module_list, 
                prev_filters, 
                output_filters):
    print('handle_yolo')
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
                    module_list, 
                    prev_filters, 
                    output_filters)
            except Exception as err:
                print("call handler_", block_type, " has except:", err)
        else:
            print("has not handler_", block_type)
        print("after handle_{0}".format(block_type), " pre_filters=",prev_filters, " output_filters=", output_filters)

        #except:
        #    print("no handle deal with:", block_type)
###############test################
blocks = parse_cfg("./cfg/yolov3.cfg")
net_info = blocks[0]
print(net_info)
print(block_handlers)
print('-------------')
create_modules_from_blocks(blocks)