import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict
from functools import partial
import math
from math import lcm, gcd
import warnings

#Bigger patches, less memory
#bigger patches might lead to less information

class Part_CFGS:
    CEH_cfg: Dict[str,Dict[str,Union[List[int], List[List[int]], int]]] = {
    "A" :
        {
            "block_list": [[[32, 64],
                            [64, 16]],
                            [[32, 64],
                            [64, 16]]],
            "kernel_sizes": [[9,7],
                            [5,3]],
            "patch_size": 2,
            "image_size": 224,
            "in_channel": 3
        },
    "B" :
        {
            "block_list": [[32,"M",16],
                           [16,"A",16]],
            "kernel_sizes": [5,7],
            "patch_size": [8,2],
            "image_size": 224,
            "in_channel": 3
        },
    "C" :
        {
            "block_list": [[16, 32, "M", 64, 32, "A", 64, 16],
                           [16, 64, "M", 128, 256, "A", 64, 16],
                           [16, 64, "M", 32, 64, "A", 32, 16],
                           [16, 64, "M", 128, 64, "A", 32, 16],
                           [8, "M", 4, "A", 2, "M", 1]],
            "kernel_sizes": [9,3,7,5,3],
            "patch_size": [8,2],
            "image_size": 224,
            "in_channel": 3
        },
    "D" :
        {
            "block_list": [[16, 32, "M", 64, 32, "A", 64, 16],
                           [16, 64, "M", 128, 256, "A", 64, 16],
                           [16, 64, "M", 32, 64, "A", 32, 16],
                           [16, 64, "M", 128, 64, "A", 32, 16],],
            "kernel_sizes": [9,3,7,5],
            "image_size": 224,
            "in_channel": 3
        },
    "E" :
        {
            "block_list": [[32,16,"M",64,32,"M",32,1]],
            "kernel_sizes": [5],
            "patch_size": [8,2],
            "image_size": 224,
            "in_channel": 3
        },
    "F" :
        {
            "block_list": [[32,16,"M",64,32,"M",32,1]],
            "kernel_sizes": [11],
            "patch_size": [8,2],
            "image_size": 224,
            "in_channel": 3
        },
    "G" :
        {
            "block_list": [[32,16,"M",64,32,"M",32,16],
                           [32,64,"M",16,64,"M",32,1,"AD"]],
            "kernel_sizes": [11,5],
            "patch_size": [8,2],
            "image_size": 224,
            "in_channel": 3
        },
    }
    cnn_cfg: Dict[str,Dict[str,Union[float, bool]]] = {
        "A" :
            {
                "dropout": 0.1,
                "batchnorm": True,
                "reshape": True,
                "padding": 0,
                "dilation": 1,
                "groups": 1,
                "stride": 1
            },
        "B" :
            {
                "dropout": 0.1,
                "batchnorm": True,
                "reshape": True,
                "padding": 0,
                "dilation": 2,
                "groups": 1,
                "stride": 1
            },
        "C" :
            {
                "dropout": 0.1,
                "batchnorm": True,
                "reshape": False,
                "padding": 0,
                "dilation": 2,
                "groups": 1,
                "stride": 1
            },
    }
    encoder_cfg: Dict[str,Dict[str,Union[int, float, Callable[..., nn.Module]]]] = {
        "A" :
            {
                "num_layers": 2,
                "num_heads": 2,
                "mlp_dim": 64,
                "dropout": 0.1,
                "dropout_atten": 0.1,
                "norm_layer": partial(nn.LayerNorm, eps=1e-6)
            }
    }
    head_cfg: Dict[str,Dict[str,Union[int, float, Optional[nn.Module]]]] = {
        "A" :
            {
                "representation_size": 128,
                "head_out": 6,
                "dropout": 0.1,
                "act": nn.ReLU()
            }
    }
    feat_cfg: Dict[str,Dict[str,Union[bool, int, float]]] = {
        "A" :
            {
                "include_features": True,
                "representation_size": 256,
                "dropout": 0.1,
                "out_size": 128,
                "feat_size": 163
            },
        "B" :
            {
                "representation_size": 128,
                "output_size": 32,
                "input_size": 163
            }
    }
    encoder_cnn_cfg: Dict[str,Dict[str,Union[float,int,bool]]] = {
        "A" : {
            "block_list": [[2]],
            "kernel_size": [1],
            "in_channels": 16,
            "out_channels": 2,
            "dropout": 0.1,
            "batchnorm": False,
            "reshape": True,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1
        }
    }
    

class kernelwise_CFG:
    CEH_cfg = Part_CFGS.CEH_cfg
    cnn_cfg = Part_CFGS.cnn_cfg
    encoder_cfg = Part_CFGS.encoder_cfg
    head_cfg = Part_CFGS.head_cfg
    feat_cfg = Part_CFGS.feat_cfg
    
    kernel_wise_parallel_vit_cfg: Dict[str,Dict[str,Union[Dict[str,Union[List[int], List[List[int]], int]], 
                                           Dict[str,Union[float, bool]], 
                                           Dict[str,Union[int, float, Callable[..., nn.Module]]],
                                           Dict[str,Union[int, float, Optional[nn.Module]]],
                                           Dict[str,Union[bool, int, float]],
                                           int, 
                                           float]]] = {
        "standard": {
            "CEH_cfg": CEH_cfg["B"],
            "cnn_cfg": cnn_cfg["A"],
            "encoder_cfg": encoder_cfg["A"],
            "head_cfg": head_cfg["A"],
            "feat_cfg":feat_cfg["A"],
            "classifier_size": 512,
            "classifier_dropout": 0.1,
            "num_classes": 12
        }                                       
    }


class Emsemble_CFG:
    CEH_cfg = Part_CFGS.CEH_cfg
    cnn_cfg = Part_CFGS.cnn_cfg
    encoder_cfg = Part_CFGS.encoder_cfg
    head_cfg = Part_CFGS.head_cfg
    feat_cfg = Part_CFGS.feat_cfg
    encoder_cnn_cfg = Part_CFGS.encoder_cnn_cfg
    
    cnn_enc_cfg: Dict[str,Dict[str,Union[Dict[str,Union[List[int], List[List[int]], int]], 
                            Dict[str,Union[float, bool]], 
                            Dict[str,Union[int, float, Callable[..., nn.Module]]],
                            Dict[str,Union[int, float, Optional[nn.Module]]],
                            Dict[str,Union[bool, int, float]],
                            int, 
                            float]]] = {
        "A": 
            {
            "kernel_sizes": CEH_cfg["E"]["kernel_sizes"],
            "block_list": CEH_cfg["E"]["block_list"],
            "in_channels": 4,
            "dropout": 0.2,
            "batchnorm": True,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "num_classes": 64
            },
        "B": 
            {
            "kernel_sizes": CEH_cfg["F"]["kernel_sizes"],
            "block_list": CEH_cfg["F"]["block_list"],
            "in_channels": 4,
            "dropout": 0.2,
            "batchnorm": True,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "num_classes": 64
            },
        "C": 
            {
            "kernel_sizes": CEH_cfg["G"]["kernel_sizes"],
            "block_list": CEH_cfg["G"]["block_list"],
            "in_channels": 4,
            "dropout": 0.2,
            "batchnorm": True,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "num_classes": 64
            },
                
                                        }
                            
    emsemble_cfgs = { 
                     "model_cfg": [cnn_enc_cfg["C"]],
                     "num_classes": 6,
                     'adapt_size': 16,
                     "encoder_cfg": encoder_cfg['A'],
                     "head_cfg": head_cfg['A'],
                    }
    
class CNNLIN_cfg:
    cnn_cfg = {
        "block_list" : [[16, 32, "M", 64, 32, "M", 64, 16],
                        [16, 32, "A", 64, 32, "A", 64, 16],],
        "kernel_sizes" : [7, 3,],
        "batchnorm" : True,
        "dropout" : 0.2,
        "in_channels" : 4,
        "outsize" : 4,
        "adapt_size" : 20
    }

class Seq_vit_CFG:
    CEH_cfg = Part_CFGS.CEH_cfg
    cnn_cfg = Part_CFGS.cnn_cfg
    encoder_cfg = Part_CFGS.encoder_cfg
    head_cfg = Part_CFGS.head_cfg
    feat_cfg = Part_CFGS.feat_cfg       
    encoder_cnn_cfg = Part_CFGS.encoder_cnn_cfg
                                        
    blockwise_seqeuntial_CnnEnc_cfg: Dict[str,Dict[str,Union[Dict[str,Union[List[int], List[List[int]], int]], 
                                    Dict[str,Union[float, bool]], 
                                    Dict[str,Union[int, float, Callable[..., nn.Module]]],
                                    Dict[str,Union[int, float, Optional[nn.Module]]],
                                    Dict[str,Union[bool, int, float]],
                                    int, 
                                    float]]] = {
        "standard": {
            "kernel_sizes": CEH_cfg["B"]["kernel_sizes"],
            "block_list": CEH_cfg["B"]["block_list"],
            "cnn_cfg": cnn_cfg["B"],
            "encoder_cfg": encoder_cfg["A"],
            "head_cfg": head_cfg["A"],
            "encoder_cnn_cfg":encoder_cnn_cfg["A"],
            "patch_size": 44,
            "in_channel": 3,
            "adapt_size": [44,44],
            "representation_size": 256,
            "num_classes": 128
        }                                       
    }

class Original_cfgs:
                       
    one_step_up = [16, "A", 16, 32, "M", 32, 64, "A", 64, 32, "M", 32, 16, "A", 16, 32, "M", 32, 64, "A", 64, 128, "M", 128, 256, "A", 256, 512, "M"] #19*4 + 10
    one_step_down = [512, 256, "A", 256, "M", 128, 64, "A", 64, 128, "M", 128,  64, "A",  64, 32, "M", 32, 16, "A"] #13*4 + 7
    one_step = one_step_up + one_step_down
    kernelUp = ["k+"]
    one_step_dec = kernelUp*4 + one_step_up + kernelUp*2 + one_step_down + kernelUp*3 + one_step_up + kernelUp + one_step_down #((19*4 + 10)*2 + (13*4 +7)) *2
    one_step_inc = kernelUp + one_step_up + kernelUp*3 + one_step_down + kernelUp*2 + one_step_up + kernelUp*4 + one_step_down #(19*4 + 10)*2 + (13*4 +7)

    conv_cfgs: Dict[str, List[Union[str, int]]] = {
        "A": [64, 128, "A", 512, "M", 512, 1024, "A", 1024, 512, "M", 512, 256, "A"], #shallow 
        "B": [64, 256, "M", 512, 512, "A", 256, 256, "M"], #shallow and WIDE
        "C": [64, 128, "M", 128, 64, "A"], #super shallow and thin <- current best
        "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        "E": [64, 64, "A", 128, 128, "M", 256, 256, "A", 1024, 1024, "M", 512, 512, "A", 256, 256, "M", 128, 128, "A", 256, 256, "M"],
        "F": [64, 128, "M", 128, 32, "M", 64, 128, "M"],
        "G": [64, "A"],
        "H": [64, 128, "A", 128, 32, "M", 64, 128, "A"],
        "I": one_step_dec + one_step_inc + one_step_dec + kernelUp +one_step + ["A", 16, 8, "M", 8], #super deep super thin
        "J": kernelUp*4+[32, "M", 32, 64, "A", 64, 128, "M", 128,]+kernelUp*2+[128, "A", 64, 32, "M", 32, 16, "A"]+kernelUp*3+[32, "M", 32, 64, "A", 64, 128, "M", 128,]+kernelUp+[128, "A", 64, 32, "M", 32, 16, "A"]#for Vit Test
    }


    model_cfgs: Dict[str, Dict[str, Union[int, float, List[int], bool]]] = {
        "Standard" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 256,
            "include features": False,
            "batch norm": True,
            "max kernel size": 3
        },
        "Standard-with-Features" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 256,
            "include features": True,
            "batch norm": True,
            "max kernel size": 3
        },
        "Small-with-Features" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 32,
            "include features": True,
            "batch norm": True,
            "max kernel size": 3
        },
        "Small-with-Features-KernelAlt" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 32,
            "include features": True,
            "batch norm": True,
            "max kernel size": 9
        },
        "Small-with-Features-KernelAlt-Reversed" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 32,
            "include features": True,
            "batch norm": True,
            "max kernel size": 9
        },
        "Small-no-Features-KernelAlt-Reversed" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 32,
            "include features": False,
            "batch norm": True,
            "max kernel size": 9
        },
        "Small-with-Features-KernelAlt-Reversed-nobatch" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 32,
            "include features": True,
            "batch norm": False,
            "max kernel size": 9
        },
        "Small-with-Features-KernelAlt-Reversed-smallout" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 8,
            "include features": True,
            "batch norm": True,
            "max kernel size": 9
        },
        "Small-with-Features-KernelAlt-Reversed-smallerout" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 1,
            "include features": True,
            "batch norm": True,
            "max kernel size": 9
        },
        "Small-with-Features-KernelAlt-Reversed-smallerout" : {
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 1,
            "include features": True,
            "batch norm": True,
            "max kernel size": 3,
            "calc padding": True
        },
        "Vit-Test-Standard": {
            "image size": 224,
            "patch size": 2,
            "num layers": 1,
            "num heads" : 8,
            "hidden dim": 256,
            "mlp dim": 256,
            "dropout enc": 0.1,
            "dropout atten": 0.1,
            "representation size": 1024,
            
            "num classes": 6,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            "adapt size": [9,9],
            "conv outsize": 64,
            "include features": True,
            "batch norm": True,
            "max kernel size": 3,
            "calc padding": True,
        }
    }

    vit_cfg:Dict[str, Dict[str, Union[List[List[int]],List[int]]]] = {
        "J": {'blocks':[[16, 32, 64],[64, 128, 64],[64, 32, 64]], 'kernels':[3,7,3]}
        }

    # image_size:int, patch_size:int, num_layers:int, num_heads:int, hidden_dim:int, mlp_dim:int, features: nn.Module, 
    # norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), num_classes: int = 6, dropout_fc: float = 0.1, 
    # dropout_enc:float = 0.1, dropout_atten:float = 0.1, adapt_size: List[int] = [9,9], conv_outsize: int = 512, 
    # include_features:bool = False, representation_size:Optional[int] = None


    vit_cfg: Dict[str, Dict[str, Union[int, float, List[int], bool, List[List[int]], List[int]]]] = {
        "Vit-Standard": {
            "image size": 224,
            "patch size": 2,
            
            "num layers": 1,
            "num heads" : 8,
            
            "hidden dim": 256,
            
            "mlp dim": 256,
            "representation size": 1024,
            
            "dropout enc": 0.1,
            "dropout atten": 0.1,
            "dropout fc": 0.1,
            "dropout conv": 0.1,
            
            "num classes": 6,
            "adapt size": [9,9],
            "conv outsize": 64,
            "include features": True,
            "batchnorm": True,
            
            'blocks':[[16, 32, 64],[64, 128, 64],[64, 32, 64]],
            'kernels':[3,7,3]
        }
    }