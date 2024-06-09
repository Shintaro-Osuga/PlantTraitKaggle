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


class InceptionV3Block3x3(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 num_channels_filter_1x1:int, 
                 num_channels_filter_3x3_in:int, 
                 num_channels_filter_3x3_out:int, 
                 pooling_out:int, 
                 factorize:bool = True):
        super(InceptionV3Block3x3, self).__init__()
        
        self.factorize = factorize
        
        self.x3block_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_1x1, kernel_size=(1,1), stride=1, padding=0))
        if factorize:
            torch._assert(condition=num_channels_filter_3x3_out % 2 == 0,
                        message="num_channels_filter_3x3_out must be even")
            self.x3block_2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_3x3_in, kernel_size=(1,1), stride=1, padding=0))
            out_size = num_channels_filter_3x3_out//2
            # print(type(out_size))
            self.x3block_2_1 = nn.Sequential(nn.Conv2d(in_channels=num_channels_filter_3x3_in, out_channels=out_size, kernel_size=(1,3), stride=1, padding="same"))
            self.x3block_2_2 = nn.Sequential(nn.Conv2d(in_channels=num_channels_filter_3x3_in, out_channels=out_size, kernel_size=(3,1), stride=1, padding="same"))
        else:
            self.x3block_2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_3x3_in, kernel_size=(1,1), stride=1, padding=0),
                                         nn.Conv2d(in_channels=num_channels_filter_3x3_in, out_channels=num_channels_filter_3x3_out, kernel_size=(3,3), stride=1, padding="same"))
        
        self.x3block_3 = nn.Sequential(nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=(1,1)),
                                       nn.Conv2d(in_channels=in_channels, out_channels=pooling_out, kernel_size=(1,1), stride=1, padding=0))
        
    def forward(self, img):
        first_block = self.x3block_1(img)
        second_block = self.x3block_2(img)
        if self.factorize:
            second_block_fac_1 = self.x3block_2_1(second_block)
            second_block_fac_2 = self.x3block_2_2(second_block)
            second_block = torch.concat([second_block_fac_1, second_block_fac_2], dim=1)
            
        third_block = self.x3block_3(img)
        
        concat = torch.concat([first_block, second_block, third_block], dim = 1)
        return concat

class InceptionV3Block5x5(nn.Module):
    def __init__(self,
                 in_channels:int, 
                 num_channels_filter_1x1:int, 
                 num_channels_filter_3x3_in:int, 
                 num_channels_filter_3x3_out:int, 
                 num_channels_filter_5x5_in:int,
                 num_channels_filter_5x5_out:int,
                 pooling_out:int, 
                 factorize:bool = True):
        super(InceptionV3Block5x5, self).__init__()
        
        self.factorize = factorize
        
        self.InceptionBlock3x3 = InceptionV3Block3x3(in_channels=in_channels, 
                                                     num_channels_filter_1x1=num_channels_filter_1x1, 
                                                     num_channels_filter_3x3_in=num_channels_filter_3x3_in,
                                                     num_channels_filter_3x3_out=num_channels_filter_3x3_out,
                                                     pooling_out=pooling_out,
                                                     factorize=factorize)
        if factorize:
            torch._assert(condition=num_channels_filter_5x5_out % 2 == 0,
                          message="num_channels_filter_5x5_out must be even")
            self.x5block_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_3x3_in, kernel_size=(1,1), stride=1, padding=0),
                                           nn.Conv2d(in_channels=num_channels_filter_3x3_in, out_channels=num_channels_filter_5x5_in, kernel_size=(3,3), stride=1, padding="same"))
            self.x5block_1_1 = nn.Sequential(nn.Conv2d(in_channels=num_channels_filter_5x5_in, out_channels=num_channels_filter_5x5_out//2, kernel_size=(1,5), stride=1, padding="same"))
            self.x5block_1_2 = nn.Sequential(nn.Conv2d(in_channels=num_channels_filter_5x5_in, out_channels=num_channels_filter_5x5_out//2, kernel_size=(5,1), stride=1, padding="same"))
        else:
            self.x5block_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_3x3_in, kernel_size=(1,1), stride=1, padding=0),
                                           nn.Conv2d(in_channels=num_channels_filter_3x3_in, out_channels=num_channels_filter_5x5_in, kernel_size=(3,3), stride=1, padding="same"),
                                           nn.Conv2d(in_channels=num_channels_filter_5x5_in, out_channels=num_channels_filter_5x5_out, kernel_size=(5,5), stride=1, padding="same"))
        
    def forward(self, img):
        incept_3x3_out = self.InceptionBlock3x3(img)
        x5block_out = self.x5block_1(img)
        if self.factorize:
            x5block_fac_1 = self.x5block_1_1(x5block_out)
            x5block_fac_2 = self.x5block_1_2(x5block_out)
            x5block_out = torch.concat([x5block_fac_1, x5block_fac_2], dim=1)
            
        concat = torch.concat([incept_3x3_out, x5block_out], dim=1)
        
        return concat
    
class InceptionV3Block7x7(nn.Module):
    def __init__(self,
                 in_channels:int, 
                 num_channels_filter_1x1:int, 
                 num_channels_filter_3x3_in:int, 
                 num_channels_filter_3x3_out:int, 
                 num_channels_filter_5x5_in:int,
                 num_channels_filter_5x5_out:int,
                 num_channels_filter_7x7_in:int,
                 num_channels_filter_7x7_out:int,
                 pooling_out:int, 
                 factorize:bool = True):
        super(InceptionV3Block7x7, self).__init__()
        
        self.factorize = factorize
        
        self.InceptionBlock5x5 = InceptionV3Block5x5(in_channels=in_channels, 
                                                     num_channels_filter_1x1=num_channels_filter_1x1, 
                                                     num_channels_filter_3x3_in=num_channels_filter_3x3_in,
                                                     num_channels_filter_3x3_out=num_channels_filter_3x3_out,
                                                     num_channels_filter_5x5_in=num_channels_filter_5x5_in,
                                                     num_channels_filter_5x5_out=num_channels_filter_5x5_out,
                                                     pooling_out=pooling_out,
                                                     factorize=factorize)
        
        if factorize:
            torch._assert(condition=num_channels_filter_7x7_out % 2 == 0,
                          message="num_channels_filter_7x7_out must be even")
            self.x7block_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_3x3_in, kernel_size=(1,1), stride=1, padding=0),
                                           nn.Conv2d(in_channels=num_channels_filter_3x3_in, out_channels=num_channels_filter_5x5_in, kernel_size=(3,3), stride=1, padding="same"),
                                           nn.Conv2d(in_channels=num_channels_filter_5x5_in, out_channels=num_channels_filter_7x7_in, kernel_size=(5,5), stride=1, padding="same"))
            self.x7block_1_1 = nn.Sequential(nn.Conv2d(in_channels=num_channels_filter_7x7_in, out_channels=num_channels_filter_7x7_out//2, kernel_size=(1,7), stride=1, padding="same"))
            self.x7block_1_2 = nn.Sequential(nn.Conv2d(in_channels=num_channels_filter_7x7_in, out_channels=num_channels_filter_7x7_out//2, kernel_size=(7,1), stride=1, padding="same"))
        else:
            self.x7block_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_3x3_in, kernel_size=(1,1), stride=1, padding=0),
                                           nn.Conv2d(in_channels=num_channels_filter_3x3_in, out_channels=num_channels_filter_5x5_in, kernel_size=(3,3), stride=1, padding="same"),
                                           nn.Conv2d(in_channels=num_channels_filter_5x5_in, out_channels=num_channels_filter_7x7_in, kernel_size=(5,5), stride=1, padding="same"),
                                           nn.Conv2d(in_channels=num_channels_filter_7x7_out, out_channels=num_channels_filter_7x7_out, kernel_size=(7,7), stride=1, padding="same"))
        
    def forward(self, img):
        incept_5x5_out = self.InceptionBlock5x5(img)
        x7block_out = self.x7block_1(img)
        if self.factorize:
            x7block_fac_1 = self.x7block_1_1(x7block_out)
            x7block_fac_2 = self.x7block_1_2(x7block_out)
            x7block_out = torch.concat([x7block_fac_1, x7block_fac_2], dim=1)
            
        concat = torch.concat([incept_5x5_out, x7block_out], dim=1)
        
        return concat
        
class ReductionBlock5x5(nn.Module):
    def __init__(self,
                 in_channels:int, 
                 num_channels_filter_1x1:int, 
                 num_channels_filter_3x3_in:int, 
                 num_channels_filter_3x3_out:int,):
        super(ReductionBlock5x5, self).__init__()
        
        self.Block1 = nn.Sequential(nn.MaxPool2d(kernel_size=(3,3), stride=2))
        self.Block2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_3x3_out, kernel_size=(3,3), stride=2, padding=0),)
        self.Block3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_1x1, kernel_size=(1,1), stride=1, padding=0),
                                    nn.Conv2d(in_channels=num_channels_filter_1x1, out_channels=num_channels_filter_3x3_in, kernel_size=(3,3), stride=1, padding=1),
                                    nn.Conv2d(in_channels=num_channels_filter_3x3_in, out_channels=num_channels_filter_3x3_out, kernel_size=(3,3), stride=2, padding=0))
        
    def forward(self, x):
        x1 = self.Block1(x)
        x2 = self.Block2(x)
        x3 = self.Block3(x)
        
        return torch.concat([x1,x2,x3], dim=1)
        
class RecuctionBlock7x7(nn.Module):
    def __init__(self,
                 in_channels:int, 
                 num_channels_filter_1x1:int, 
                 num_channels_filter_3x3_in:int, 
                 num_channels_filter_3x3_out:int, 
                 num_channels_filter_7x7_in:int,
                 num_channels_filter_7x7_out:int,):
        super(RecuctionBlock7x7, self).__init__()
        
        self.Block1 = nn.Sequential(nn.MaxPool2d(kernel_size=(3,3), stride=2))
        self.Block2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_3x3_in, kernel_size=(1,1), stride=1, padding=0),
                                    nn.Conv2d(in_channels=num_channels_filter_3x3_in, out_channels=num_channels_filter_3x3_out, kernel_size=(3,3), stride=2))
        self.Block3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=num_channels_filter_1x1, kernel_size=(1,1), stride=1, padding=0),
                                    nn.Conv2d(in_channels=num_channels_filter_1x1, out_channels=num_channels_filter_7x7_in, kernel_size=(1,7), stride=1, padding=1),
                                    nn.Conv2d(in_channels=num_channels_filter_7x7_in, out_channels=num_channels_filter_7x7_out, kernel_size=(7,1), stride=1, padding=1),
                                    nn.Conv2d(in_channels=num_channels_filter_7x7_out, out_channels=num_channels_filter_3x3_out, kernel_size=(3,3), stride=2, padding=1))
        
    def forward(self, x):
        x1 = self.Block1(x)
        x2 = self.Block2(x)
        x3 = self.Block3(x)
        
        return torch.concat([x1, x2, x3], dim=1)
    

class Inception_classifier(nn.Module):
    def __init__(self, inception_cfg, cnn_classifier_cfg, aux_classifer_cfg, classifier_cfg):
        super(Inception_classifier, self).__init__()
        
        self.conv_model = create_inception_layer(inception_cfg)
        
        self.cnn_classifier = _make_classifier(classifier_list=cnn_classifier_cfg['classifier_list'],
                                               norm_type=cnn_classifier_cfg['norm_type'],
                                               act=cnn_classifier_cfg['act'],
                                               dropout=cnn_classifier_cfg['dropout'],
                                               out_act=cnn_classifier_cfg['out_act'],
                                               insize=484)
        
        # self.aux_classifier = _make_classifier(classifier_list=aux_classifer_cfg['classifier_list'],
        #                                        norm_type=aux_classifer_cfg['norm_type'],
        #                                        act=aux_classifer_cfg['act'],
        #                                        dropout=aux_classifer_cfg['dropout'],
        #                                        out_act=aux_classifer_cfg['out_act'],
        #                                        insize=163)
        
        # self.cnn_aux_con = nn.Sequential(nn.Linear(64+32, 128),
        #                                  nn.Linear(128, 256),
        #                                  nn.Linear(256, 64),
        #                                  nn.Linear(64, 6))
        
        # self.aux_cnn_con = nn.Sequential(nn.Linear(64+32, 128),
        #                                  nn.Linear(128, 256),
        #                                  nn.Linear(256, 64),
        #                                  nn.Linear(64, 6))
        
        self.classifier = _make_classifier(classifier_list=classifier_cfg['classifier_list'],
                                           norm_type=classifier_cfg['norm_type'],
                                           act=classifier_cfg['act'],
                                           dropout=classifier_cfg['dropout'],
                                           out_act=classifier_cfg['out_act'],
                                           insize=64)
    def forward(self, img, aux):
        x = self.conv_model(img)
        x = self.cnn_classifier(torch.flatten(x, start_dim=1))
        
        # x2 = self.aux_classifier(aux)
        x = self.classifier(x)
        # cnn_aux_out = self.cnn_aux_con(torch.concat([x.clone(), x2.clone()], dim=1))
        # aux_cnn_out = self.aux_cnn_con(torch.concat([x.clone(), x2.clone()], dim=1))
        # out = self.classifier(con)
        return x
    
def _make_classifier(classifier_list:List[int], norm_type:str, act:nn.Module, dropout:float, out_act:nn.Module, insize:int) -> nn.Sequential:
    from torch.nn.utils.parametrizations import weight_norm

    layers:List[nn.Module] = []
    prev_value = insize #10816 #7688 # 729# this is the outside of the flattened CNN network output
    
    for val in classifier_list:
        if norm_type == "batchnorm":
            layers += [nn.Linear(prev_value, val), nn.BatchNorm1d(val), act, nn.Dropout1d(p=dropout)]
        elif norm_type == "weightSTD":
            layers += [weight_norm(nn.Linear(prev_value, val)), act, nn.Dropout1d(p=dropout)]
        elif norm_type == "batch+weight":
            layers += [weight_norm(nn.Linear(prev_value, val)), nn.BatchNorm1d(val), act, nn.Dropout1d(p=dropout)]
        elif norm_type == "none":
            layers += [nn.Linear(prev_value, val), act, nn.Dropout1d(p=dropout)]
        elif norm_type == "no_act":
            layers += [nn.Linear(prev_value, val), nn.Dropout1d(p=dropout)]
        elif norm_type == "no_drop":
            layers += [nn.Linear(prev_value, val), act]
        elif norm_type == "no_act_drop":
            layers += [nn.Linear(prev_value, val)]

        prev_value = val
    
    if out_act != None:
        layers += [out_act]
    
    return nn.Sequential(*layers)
# layers = [128, 16, 32, I7, 64, M, 16, I, 128, M]
# incep configs: 7: 7x7_A_CFG
#                5: 5x5_A_CFG
def create_inception_layer(cfg:Dict) -> nn.Module:
    from torch.nn.utils.parametrizations import weight_norm

    layers:list[nn.Module] = []
    prev = cfg["conv_cfg"]["in_channels"]
    for idx, v in enumerate(cfg['architecture']):
        # print(f'hw:{hw} v: {v}')
        if isinstance(v, list):
            kernel_size = v[0]
            # print(v)
            # print(kernel_size)
        elif v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)]
            # hw = _calculated_maxpool_hw(kernel_size=2, shape=self.hw, padding=0, stride=2)
            # self._update_shape(new_hw=self._calculated_maxpool_hw(kernel_size=2, stride=1, padding=1), new_channel=in_channel)
        elif v == "A":
            layers += [nn.AvgPool2d(kernel_size=2, stride=1, padding=1)]
            # self.hw = self._calculated_avgpool_hw(kernel_size=2, stride=1, padding=1, shape=self.hw)
        elif v == "AD":
            layers += [nn.AdaptiveAvgPool2d((cfg["conv_cfg"]["adapt_size"], cfg["conv_cfg"]["adapt_size"]))]
            # self.hw = self.adapt_size
            # self._update_shape(new_hw=self._calculated_avgpool_hw(kernel_size=2, stride=1, padding=1), new_channel=in_channel)
        # elif v == 'N':
        #     func = torch.sin if func == torch.cos else torch.cos
        #     negative = True if negative == False else False
        #     # print(hw)
        #     layers += [NonLinWeight(function=func, shape=self.hw, num_channels=self.last, batch_first=self.batch_first, negative=negative)]
        elif isinstance(v, int):
            v = cast(int, v)
            # print(f'kernel_size: {kernel_size}')
            #used to have padding = "same"
            conv2d = nn.Conv2d(in_channels=prev, 
                               out_channels=v, 
                               kernel_size=kernel_size, 
                               stride=cfg["conv_cfg"]["stride"], 
                               padding=cfg["conv_cfg"]["padding"], 
                               dilation=cfg["conv_cfg"]["dilation"], 
                               groups=cfg["conv_cfg"]["groups"])
            # print(f'conv hw: {hw}')
            # self._update_shape(new_hw=self._calculated_conv2d_hw(kernel_size=kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation), new_channel=v)
            # print(self.shape)
            act = nn.ReLU()
            # act = nn.LeakyReLU(inplace=True)
            if cfg["norm_type"] == 'batch':
                layers += [conv2d, nn.BatchNorm2d(v), act, nn.Dropout2d(p=cfg["conv_cfg"]["dropout"])]
            elif cfg["norm_type"] == "group+weightSTD":
                groupnum = v // 2 if v > 1 else 1
                layers += [weight_norm(conv2d, dim=1), nn.GroupNorm(groupnum, v), act, nn.Dropout2d(p=cfg["conv_cfg"]["dropout"])]
            elif cfg["norm_type"] == "group":
                groupnum = v // 2 if v > 2 else 1
                layers += [conv2d, nn.GroupNorm(groupnum, v), act, nn.Dropout2d(p=cfg["conv_cfg"]["dropout"])]
            elif cfg["norm_type"] == "weightSTD":
                layers += [weight_norm(conv2d, dim=1), act, nn.Dropout2d(p=cfg["conv_cfg"]["dropout"])]
            else:
                layers += [conv2d, act]
            prev = v
            kernel_size = 3 if kernel_size !=  3 else kernel_size
        elif v[0] == "I":
            if v[1] == "7":
                layers += [InceptionV3Block7x7(in_channels = prev,
                                               num_channels_filter_1x1 = cfg["incept_cfg"][v[1]]["num_channels_filter_1x1"],
                                               num_channels_filter_3x3_in = cfg["incept_cfg"][v[1]]["num_channels_filter_3x3_in"],
                                               num_channels_filter_3x3_out = cfg["incept_cfg"][v[1]]["num_channels_filter_3x3_out"],
                                               num_channels_filter_5x5_in = cfg["incept_cfg"][v[1]]["num_channels_filter_5x5_in"],
                                               num_channels_filter_5x5_out = cfg["incept_cfg"][v[1]]["num_channels_filter_5x5_out"],
                                               num_channels_filter_7x7_in = cfg["incept_cfg"][v[1]]["num_channels_filter_7x7_in"],
                                               num_channels_filter_7x7_out = cfg["incept_cfg"][v[1]]["num_channels_filter_7x7_out"],
                                               pooling_out = cfg["incept_cfg"][v[1]]["pooling_out"],
                                               factorize = cfg["incept_cfg"][v[1]]["factorize"],
                                               )]
                prev = cfg["incept_cfg"][v[1]]["num_channels_filter_1x1"] + cfg["incept_cfg"][v[1]]["num_channels_filter_3x3_out"] + cfg["incept_cfg"][v[1]]["pooling_out"] + cfg["incept_cfg"][v[1]]["num_channels_filter_5x5_out"] + cfg["incept_cfg"][v[1]]["num_channels_filter_7x7_out"]
            elif v[1] == "5":
                layers += [InceptionV3Block5x5(in_channels = prev,
                                               num_channels_filter_1x1 = cfg["incept_cfg"][v[1]]["num_channels_filter_1x1"],
                                               num_channels_filter_3x3_in = cfg["incept_cfg"][v[1]]["num_channels_filter_3x3_in"],
                                               num_channels_filter_3x3_out = cfg["incept_cfg"][v[1]]["num_channels_filter_3x3_out"],
                                               num_channels_filter_5x5_in = cfg["incept_cfg"][v[1]]["num_channels_filter_5x5_in"],
                                               num_channels_filter_5x5_out = cfg["incept_cfg"][v[1]]["num_channels_filter_5x5_out"],
                                               pooling_out = cfg["incept_cfg"][v[1]]["pooling_out"],
                                               factorize = cfg["incept_cfg"][v[1]]["factorize"],
                                               )]
                prev = cfg["incept_cfg"][v[1]]["num_channels_filter_1x1"] + cfg["incept_cfg"][v[1]]["num_channels_filter_3x3_out"] + cfg["incept_cfg"][v[1]]["pooling_out"] + cfg["incept_cfg"][v[1]]["num_channels_filter_5x5_out"]
            else:
                layers += [InceptionV3Block3x3(in_channels = prev,
                                               num_channels_filter_1x1 = cfg["incept_cfg"][v[1]]["num_channels_filter_1x1"],
                                               num_channels_filter_3x3_in = cfg["incept_cfg"][v[1]]["num_channels_filter_3x3_in"],
                                               num_channels_filter_3x3_out = cfg["incept_cfg"][v[1]]["num_channels_filter_3x3_out"],
                                               pooling_out = cfg["incept_cfg"][v[1]]["pooling_out"],
                                               factorize = cfg["incept_cfg"][v[1]]["factorize"],
                                                )]
                prev = cfg["incept_cfg"][v[1]]["num_channels_filter_1x1"] + cfg["incept_cfg"][v[1]]["num_channels_filter_3x3_out"] + cfg["incept_cfg"][v[1]]["pooling_out"]
        elif v[0] == "R":
            if v[1] == "5":
                layers += [ReductionBlock5x5(in_channels=prev,
                                             num_channels_filter_1x1=cfg["reduc_cfg"][v[1]]["num_channels_filter_1x1"],
                                             num_channels_filter_3x3_in=cfg["reduc_cfg"][v[1]]["num_channels_filter_3x3_in"],
                                             num_channels_filter_3x3_out=cfg["reduc_cfg"][v[1]]["num_channels_filter_3x3_out"])]
                prev = prev + cfg["reduc_cfg"][v[1]]["num_channels_filter_3x3_out"] + cfg["reduc_cfg"][v[1]]["num_channels_filter_3x3_out"]
            elif v[1] == "7":
                layers += [RecuctionBlock7x7(in_channels=prev,
                                             num_channels_filter_1x1=cfg["reduc_cfg"][v[1]]["num_channels_filter_1x1"],
                                             num_channels_filter_3x3_in=cfg["reduc_cfg"][v[1]]["num_channels_filter_3x3_in"],
                                             num_channels_filter_3x3_out=cfg["reduc_cfg"][v[1]]["num_channels_filter_3x3_out"],
                                             num_channels_filter_7x7_in=cfg["reduc_cfg"][v[1]]["num_channels_filter_7x7_in"],
                                             num_channels_filter_7x7_out=cfg["reduc_cfg"][v[1]]["num_channels_filter_7x7_out"])]
                
                prev = prev + cfg["reduc_cfg"][v[1]]["num_channels_filter_3x3_out"] + cfg["reduc_cfg"][v[1]]["num_channels_filter_3x3_out"]
    return nn.Sequential(*layers)
    
def _calculated_conv2d_hw(kernel_size, shape, stride = 1, padding = 0, dilation = 1) -> int:
    return math.floor(((shape + (2 * padding) - dilation*(kernel_size-1) - 1) / stride) + 1)
    
def _calculated_maxpool_hw(kernel_size, shape, padding = 0, dilation = 1, stride = 1) -> int:
    return math.floor(((shape + (2 * padding) - dilation*(kernel_size -1) - 1) / stride ) + 1)
    
def _calculated_avgpool_hw(kernel_size, shape, padding = 0, stride = 1) -> int:
    return math.floor(((shape + (2 * padding) - kernel_size) / stride) + 1)
    