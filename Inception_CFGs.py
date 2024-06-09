import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict
from functools import partial
import math
from math import lcm, gcd
import warnings

class Incept_CFGS:
    '''
    in_channels:int                 |  The channel number of the input image
    num_channels_filter_1x1:int     |  Out size of the 1x1 kernel size layer, the in size is implicitly in_channels.
    num_channels_filter_3x3_in:int  |  In size of the 3x3 kernel size layer, int
    num_channels_filter_3x3_out:int |  Out size of the 3x3 kernel size layer, int.
    num_channels_filter_5x5_in:int  |  In size of the 5x5 kernel size layer, int.
    num_channels_filter_5x5_out:int |  Out size of the 5x5 kernel size layer, int.
    num_channels_filter_7x7_in:int  |  In size of the 7x7 kernel size layer, int.
    num_channels_filter_7x7_out:int |  Out size of the 7x7 kernel size layer, int. 
    pooling_out:int                 |  Out size of the pooling layer, int.
    factorize:bool = True           |  Bool, Deafult True. Changes CNN architecture to factorize to save 30% comptutation.
    '''
    Incept_CFG: Dict[str,Union[int, bool]] = {
            '7x7_A_CFG' :
                {
                    "in_channels" : 3,
                    "num_channels_filter_1x1" : 64,
                    "num_channels_filter_3x3_in" : 32,
                    "num_channels_filter_3x3_out" : 48,
                    "num_channels_filter_5x5_in" : 32,
                    "num_channels_filter_5x5_out" : 48,
                    "num_channels_filter_7x7_in" : 48,
                    "num_channels_filter_7x7_out" : 64,
                    "pooling_out" : 64,
                    "factorize" : True,
                },
            '5x5_A_CFG' :
                {
                    "in_channels" : 3,
                    "num_channels_filter_1x1" : 64,
                    "num_channels_filter_3x3_in" : 32,
                    "num_channels_filter_3x3_out" : 64,
                    "num_channels_filter_5x5_in" : 64,
                    "num_channels_filter_5x5_out" : 64,
                    "pooling_out" : 64,
                    "factorize" : True,
                },
            '3x3_A_CFG' :
                {
                    "in_channels" : 3,
                    "num_channels_filter_1x1" : 128,
                    "num_channels_filter_3x3_in" : 32,
                    "num_channels_filter_3x3_out" : 128,
                    "pooling_out" : 128,
                    "factorize" : True,
                }
        }
    
    Reduct_CFG: Dict[str, int] = {
        "7x7_A_CFG":
            {
                "num_channels_filter_1x1" : 32,
                "num_channels_filter_3x3_in" : 32,
                "num_channels_filter_3x3_out" : 64,
                "num_channels_filter_7x7_in" : 32,
                "num_channels_filter_7x7_out" : 64,
            },
        "5x5_A_CFG":
            {
                "num_channels_filter_1x1" : 32,
                "num_channels_filter_3x3_in" : 32,
                "num_channels_filter_3x3_out" : 64,
            }
    }
    
    CNN_CFG: Dict[str,Union[int, bool, float]] = {
        "Base" : {
                "dropout": 0.2,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups" : 1,
                "in_channels": 4,
                "adapt_size": 10,
        }
    }
    
    cnn_classifier_CFG: Dict[str, Dict[str, Union[List[int], float, str, nn.Module]]] = {
        "A":
            {
                "classifier_list": [7000, 2048, 4096, 1024, 512, 2048, 256, 4096, 128, 512, 1024, 10],
                "norm_type": "batch+weight",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": nn.Softmax(dim=1)
            },
        "B":
            {
                "classifier_list": [4096, 1024, 512, 2048, 256, 4096, 128, 512, 1024, 10],
                "norm_type": "batch+weight",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": nn.Softmax(dim=1)
            },
        "C":
            {
                "classifier_list": [4096, 128, 512, 1024, 10],
                "norm_type": "batchnorm",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": nn.Softmax(dim=1)
            },
        "D":
            {
                "classifier_list": [16 * 5 * 5, 128, 84, 10],
                "norm_type": "no_act",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": None
            },
        "E":
            {
                "classifier_list": [128, 32, 64],
                "norm_type": "no_act",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": None
            },
            
    }
    
    classifier_CFG: Dict[str, Dict[str, Union[List[int], float, str, nn.Module]]] = {
        "A":
            {
                "classifier_list": [7000, 2048, 4096, 1024, 512, 2048, 256, 4096, 128, 512, 1024, 10],
                "norm_type": "batch+weight",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": nn.Softmax(dim=1)
            },
        "B":
            {
                "classifier_list": [4096, 1024, 512, 2048, 256, 4096, 128, 512, 1024, 10],
                "norm_type": "batch+weight",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": nn.Softmax(dim=1)
            },
        "C":
            {
                "classifier_list": [4096, 128, 512, 1024, 10],
                "norm_type": "batchnorm",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": nn.Softmax(dim=1)
            },
        "D":
            {
                "classifier_list": [16 * 5 * 5, 128, 84, 10],
                "norm_type": "no_act",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": None
            },
        "E":
            {
                "classifier_list": [64, 128, 6],
                "norm_type": "no_act",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": None
            },
            
    }
    
    aux_CFG: Dict[str, Dict[str, Union[List[int], float, str, nn.Module]]] = {
        "A":
            {
                "classifier_list": [7000, 2048, 4096, 1024, 512, 2048, 256, 4096, 128, 512, 1024, 10],
                "norm_type": "batch+weight",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": nn.Softmax(dim=1)
            },
        "B":
            {
                "classifier_list": [4096, 1024, 512, 2048, 256, 4096, 128, 512, 1024, 10],
                "norm_type": "batch+weight",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": nn.Softmax(dim=1)
            },
        "C":
            {
                "classifier_list": [4096, 128, 512, 1024, 64],
                "norm_type": "batchnorm",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": nn.Softmax(dim=1)
            },
        "D":
            {
                "classifier_list": [16 * 5 * 5, 128, 84, 32],
                "norm_type": "no_act",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": None
            },
        "E":
            {
                "classifier_list": [128, 64, 32],
                "norm_type": "no_act",
                "act": nn.LeakyReLU(),
                "dropout": 0.1,
                "out_act": None
            },
            
    }
    
    Incept_CNN_CFG: Dict[str,Union[int, bool, str]] = {
        "Incept_v1" : {
            "incept_cfg":
            {
                "architecture": [[3], 64, 32, 128,"I3", "R5", "I7", "R7", "I5",
                                 "M", 10, 1],
                "conv_cfg": CNN_CFG["Base"],
                "incept_cfg": {
                    "7": Incept_CFG["7x7_A_CFG"],
                    "5": Incept_CFG["5x5_A_CFG"],
                    "3": Incept_CFG["3x3_A_CFG"],
                    },
                "reduc_cfg": {
                    "7" : Reduct_CFG["7x7_A_CFG"],
                    "5" : Reduct_CFG["5x5_A_CFG"]
                },
                "norm_type": "batch",
            },
            "cnn_classifier_cfg": cnn_classifier_CFG["E"],
            "aux_classifier_cfg" : aux_CFG["E"],
            "classifier_cfg": classifier_CFG["E"]
        },
        "Incept_test" : {
            "incept_cfg":
            {
                "architecture": ["I5"],
                "conv_cfg": CNN_CFG["Base"],
                "incept_cfg": {
                    "7": Incept_CFG["7x7_A_CFG"],
                    "5": Incept_CFG["5x5_A_CFG"],
                    "3": Incept_CFG["3x3_A_CFG"],
                    },
                "norm_type": "batch",
            },
            "cnn_classifier_cfg": cnn_classifier_CFG["D"],
            "aux_classifier_cfg" : aux_CFG["D"],
            "classifier_cfg": classifier_CFG["D"]
        }
    }
    