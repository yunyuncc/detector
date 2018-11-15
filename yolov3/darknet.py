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
###############test################

blocks = parse_cfg("./cfg/yolov3.cfg")
net_info = blocks[0]
print(net_info)
