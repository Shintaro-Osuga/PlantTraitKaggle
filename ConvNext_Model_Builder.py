import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict
from functools import partial
import math
from math import lcm, gcd
import warnings

from PreBuilt import Encoder
from NonLinWeight import NonLinWeight

import timm

class ConvNext(nn.Module):
    def __init__(self, n_classes:int, pretrained:bool=True):

        super(ConvNext, self).__init__()

        self.model = timm.create_model("convnext_small_384_in22ft1k", pretrained=False)
        if pretrained:
            self.model.load_state_dict(torch.load("../input/timm-convnext-xcit/convnext_small_384_in22ft1k.pth"))
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, n_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class InceptionV4(nn.Module):
    def __init__(self, n_classes:int):
        super(InceptionV4, self).__init__()

        self.model = timm.create_model("inception_v4", pretrained=False, num_classes=n_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class InceptionResNetV2(nn.Module):
    def __init__(self, n_classes:int):
        super(InceptionResNetV2, self).__init__()

        self.model = timm.create_model("inception_resnet_v2", pretrained=False, num_classes=n_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class Double_InceptionV4(nn.Module):
    def __init__(self, n_classes:int):
        super(Double_InceptionV4, self).__init__()
        self.n_classes = n_classes
        
        self.model_1 = timm.create_model("inception_v4", pretrained=False, num_classes=n_classes//2)
        self.model_2 = timm.create_model("inception_v4", pretrained=False, num_classes=n_classes//2)
        
        # self.feature_classifier = nn.Sequential(
        #                                         nn.Linear(163, 64),
        #                                         nn.Dropout(p=0.1),
        #                                         nn.Linear(64, 32),
        #                                         nn.Linear(32, 32),
        #                                         nn.Dropout(p=0.1),
        #                                         nn.Linear(32, 3),
        #                                         )

        
        # self.classifier = nn.Sequential(
        #                                  nn.Linear(6, 64),   
        #                                  nn.Linear(64, 128),
        #                                  nn.Dropout(p=0.1),   
        #                                  nn.Linear(128, 32),   
        #                                  nn.Dropout(p=0.1),   
        #                                  nn.Linear(32, 32),   
        #                                  nn.Linear(32, 3),   
        #                                 )
        
    def forward(self, x, feats):
        # feat_out = self.feature_classifier(feats)
        # first_half = self.classifier(torch.concat([self.model_1(x), feat_out], dim=1))
        # second_half = self.classifier(torch.concat([self.model_2(x), feat_out], dim=1))
        first_half = self.model_1(x)
        second_half = self.model_2(x)
        
        # out = torch.zeros([x.shape[0], self.n_classes], device='cuda')
        
        # out[:,0::2] = first_half
        # out[:,1::2] = second_half
        
        return first_half, second_half