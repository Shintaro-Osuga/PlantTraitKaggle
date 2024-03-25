import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict
from functools import partial
import math

from PreBuilt import Encoder
#cube_padding = (33*33) - cube_size*cube_size
#input dims: (6(num_sides), cub_size+cube_padding(6,33*2,33*3), max_move_num(99))
#output dims: (1, num_moves+move_padding)
class CNNVit(nn.Module):
    def __init__(self,  image_size:int, 
                 patch_size:float, 
                 num_layers:int, 
                 num_heads:int, 
                 hidden_dim:int, 
                 mlp_dim:int, 
                 features: nn.Module, 
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), 
                 num_classes: int = 6, 
                 dropout_fc: float = 0.1, 
                 dropout_enc:float = 0.1, 
                 dropout_atten:float = 0.1, 
                 adapt_size: List[int] = [9,9], 
                 conv_outsize: int = 512, 
                 include_features:bool = False,
                 representation_size:Optional[int] = None):
        super(CNNVit, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layer = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        
        
        self.adpavgp = nn.AdaptiveAvgPool2d((adapt_size[0],adapt_size[1]))
        self.features = features
        self.include_features = include_features
        self.feature_outsize = 2 if include_features else 1
        
        seq_length = int((image_size // patch_size)**2)
        # print(seq_length)
        self.class_token = nn.Parameter(torch.zeros(1,1,hidden_dim))
        seq_length += 1
        
        self.encoder = Encoder(seq_length=seq_length, 
                               num_layers=num_layers, 
                               num_heads=num_heads,
                               hidden_dim=hidden_dim, 
                               mlp_dim=mlp_dim, 
                               dropout=dropout_enc, 
                               attention_dropout=dropout_atten,
                               norm_layer=norm_layer)
        
        self.seq_length = seq_length
        
        
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        
        heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(representation_size, 512)

        self.heads = nn.Sequential(heads_layers)
        
        heads_con_layers: OrderedDict[str, nn.Module] = OrderedDict()
        
        heads_con_layers["pre_logits"] = nn.Linear(hidden_dim+512, representation_size)
        heads_con_layers["act"] = nn.Tanh()
        heads_con_layers["head"] = nn.Linear(representation_size, 512)

        self.heads_con = nn.Sequential(heads_con_layers)
        
        heads_layers_out: OrderedDict[str, nn.Module] = OrderedDict()
        
        heads_layers_out["pre_logits"] = nn.Linear(512+512, representation_size)
        heads_layers_out["act"] = nn.Tanh()
        heads_layers_out["head"] = nn.Linear(representation_size, num_classes)
        
        self.heads_out = nn.Sequential(heads_layers_out)
        
        if include_features:
            self.feature_nn = nn.Sequential(
                nn.Linear(163, 1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(1024, 256),
                nn.ReLU(True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(256, 512),
            )
        
        if isinstance(self.features, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.features.in_channels * self.features.kernel_size[0] * self.features.kernel_size[1]
            nn.init.trunc_normal_(self.features.weight, std=math.sqrt(1 / fan_in))
            if self.features.bias is not None:
                nn.init.zeros_(self.features.bias)
        elif self.features[-1] is not None and isinstance(self.features[-1], nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.features[-1].weight, mean=0.0, std=math.sqrt(2.0 / self.features[-1].out_channels)
            )
            if self.features[-1].bias is not None:
                nn.init.zeros_(self.features[-1].bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)
        
    
    #from https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L143
    def _process_input(self, x:torch.Tensor, block:nn.Module) -> torch.Tensor:
        n,c,h,w = x.shape
        p = self.patch_size
        torch._assert(h==self.image_size and w == self.image_size, "Wrong image Size")
        # torch._assert(p in self._find_patch_sizes(), "WRONG PATCH SIZE")
        p_h = int(h//p)
        p_w = int(w//p)
        
        #output of cnn
        x = block(x)
        # print(x.shape)
        #we need to 
        enc_x = x.reshape(n, self.hidden_dim, p_h*p_w).detach().clone()
        
        enc_x = enc_x.permute(0, 2, 1)
        
        return x, enc_x
    
    def _find_patch_sizes(self):
        possible_patch_sizes = []
        for x in range(1,self.image_size):
            if (self.image_size**2)% (x**2) == 0:
                possible_patch_sizes.append(x)
        return possible_patch_sizes
    
    def forward(self, img:torch.Tensor, feature:torch.Tensor) -> torch.Tensor:
        x = img
        n = x.shape[0]
        
        enc_in = torch.zeros((len(self.features), n, (self.image_size//self.patch_size)**2, self.hidden_dim), device='cuda')
        for idx, block in enumerate(self.features):
            # x = block(x)
            # print(f'before:{x.shape}')
            x, enc_x = self._process_input(x, block)
            # print(f'after:{x.shape}')
            # print(block)
            enc_in[idx] = enc_x.detach().clone()
            
        batch_class_token = self.class_token.expand(n, -1, -1)
        for idx, enc in enumerate(enc_in):
            # print(x.shape)
            enc = torch.cat([batch_class_token, enc], dim=1)
            enc = self.encoder(enc)
            enc = enc[:,0]
            
            if idx == 0:
                x = self.heads(enc)
            else:
                x = torch.concat([x, enc], dim=1)
                x = self.heads_con(x)
        
        feat_out = self.feature_nn(feature)
        x = torch.cat([x, feat_out], dim=1)
        head_out = self.heads_out(x)
        # if self.include_features:
        #     head_aux_out = self.heads_aux(x)
        # else:
        head_out_aux = None
        return head_out, head_out_aux


def make_block_layers(cfg: List[List[Union[str,int]]], kernel_size_per_block:List[int], batch_norm:bool = False, dropout: float = 0.1, in_channel:int = 3, outsize: int = 512, max_kernel_size: int = 3, calc_padding: bool = False) -> nn.Sequential:
    seq:nn.Module = nn.Sequential()
    in_channel = 3
    for idx, (block, kernel_size) in enumerate(zip(cfg, kernel_size_per_block)):
        # print(block)
        # print(kernel_size)
        layers: List[nn.Module] = []
        for v in block:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=1, padding=1)]
            elif v == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=1, padding=1)]
            else:
                # print(in_channel)
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=v, kernel_size=kernel_size, padding="same")
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True), nn.Dropout2d(p=dropout)]
                    
                in_channel = v
        # print(layers[0::4])
        seq.add_module(f'k{kernel_size}_Block_{idx}', nn.Sequential(*layers))
    
    layers: List[nn.Module] = []
    
    conv2d = nn.Conv2d(in_channels=in_channel, out_channels=outsize, kernel_size=3, padding="same")
    if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(outsize), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout)]
    else:
        layers += [conv2d, nn.ReLU(inplace=True), nn.Dropout2d(p=dropout)]
        
    seq.add_module(f'out_Block', nn.Sequential(*layers))
    return seq