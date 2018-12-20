from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *
#from torch.autograd import Variable (deprecated)
import numpy as np

def get_test_input():
    img = cv2.imread("imgs/dog-cycle-car.png")
    img = cv2.resize(img,(416,416))
    #  ::-1 means start:end:step
    #  width and height not change, change the color channel dim
    img_ = img[:,:,::-1].transpose((2,0,1)) #BGR->RGB | H W C -> C H W 

    img_ = img_[np.newaxis,:,:,:]/255.0 # add a channel at 0 for batch and normalise
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)  
    #BCHW
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
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules_from_blocks(self.blocks)
        assert(len(self.blocks) == len(self.module_list)+1)
        self.output_sizes = {}
    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
    def load_weights_old(self, weightfile):
        fp = open(weightfile, "rb")

        # header information
        # 1.major version number
        # 2.minor version number
        # 3.subversion number
        # 4,5 images seen by the network

        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype = np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]

                if (batch_normalize):
                    bn = model[1]
                    print("bn:",bn)
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    #cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    #bn_running_var.copy_(bn_running_var)    
                    # fixed bud !!!!!!!上面这里手滑写错,导致权重load出错，bn.running_var没有被设置
                    # 查看下面的这些内容的一致性是通过输出这些信息到终端，重定向到文件，然后对比md5sum
                    # 1.查看网络输出是否一致
                    # 2.查看输入是否一致
                    # 3.查看网络结构是否一致
                    # 4.查看网络的参数是否一致，发现加载的参数不一样
                    # 5.发现load_weights函数的bug
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)




    def forward(self, x, CUDA):
        #x is BCHW format
        batch, channels, height, width = x.size()
        assert(channels == int(self.net_info["channels"]))
        assert(height == int(self.net_info["height"]))
        assert(width == int(self.net_info["width"]))
        #print("input batch is ", batch)
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
                    assert(map1.size()[2] == map2.size()[2])
                    assert(map1.size()[3] == map2.size()[3])
                    x = torch.cat((map1, map2), 1)
                    assert(map1.size()[1] + map2.size(1) == x.size()[1])
            elif module_type == "shortcut":
                from_ = int(module["from"])
                assert(outputs[i-1].size() == outputs[i + from_].size())
                x = outputs[i - 1] + outputs[i + from_]
            elif module_type == "yolo":
                #[1, 255, 13, 13]==>[1, 507, 85]
                #[1, 255, 26, 26]==>[1, 2028, 85]
                #[1, 255, 52, 52]==>[1, 8112, 85]
                anchors = self.module_list[i][0].anchors
                #Get the input dim
                input_dim = int(self.net_info["height"])

                num_classes = int(module["classes"])
                #transform
                x = x.data
                #print("row yolo output:,", x.size())#last lay x.size is torch.Size([1, 255, 52, 52])
                #print("predict_transform: input_dim=", input_dim, " anchors=", anchors, " num_classes=", num_classes)
                x = predict_transform(x, input_dim,anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections,x), 1)

            outputs[i] = x
            self.output_sizes[i] = x.size()
        return detections

###############test################

#blocks = parse_cfg("./cfg/yolov3.cfg")
#net_info , module_list = create_modules_from_blocks(blocks)
if __name__ == '__main__':
    model = Darknet("/home/wyy/pytorch/my/detector/yolov3/cfg/yolov3.cfg")
    
    model.load_weights_old("/home/wyy/pytorch/my/detector/yolov3/weights/yolov3.weights")
    print("------begin")
    #print(model.state_dict())
    print("------end")
    inp = get_test_input()
    CUDA = torch.cuda.is_available()
    if CUDA:
        model.cuda()
        print("use CUDA model")
    model.eval()

    
    print("inp size:", inp.size())
    pred = model(inp.cuda(), CUDA) 
    output = write_results(pred, 0.6, 80)
    print("output size:",output.size())
    