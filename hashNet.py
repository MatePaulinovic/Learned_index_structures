# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:57:09 2018

@author: matep
"""

import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F

class HashNet(nn.Module):
    