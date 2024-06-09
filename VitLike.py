import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict
from functools import partial
import math
from math import lcm, gcd
import warnings

from PreBuilt import Encoder

class kernelwise_Vit(nn.Module):
    def __init__(self,  image_size:int, 
                 patch_size:float, 
                 num_layers:int, 
                 num_heads:int, 
                 hidden_dim:int, 
                 mlp_dim:int,
                 block_list:List[int],
                 kernel_list:List[int],
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), 
                 num_classes: int = 6, 
                 dropout_fc: float = 0.1, 
                 dropout_enc:float = 0.1, 
                 dropout_atten:float = 0.1, 
                 dropout_conv:float = 0.1,
                 adapt_size: List[int] = [9,9], 
                 conv_outsize: int = 512, 
                 include_features:bool = False,
                 batchnorm:bool = True,
                 representation_size:Optional[int] = None):
        super(kernelwise_Vit, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layer = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.dropout_enc = dropout_enc
        self.dropout_atten = dropout_atten
        self.dropout = dropout_conv
        self.batch_norm = batchnorm
        self.outsize = conv_outsize
        self.kernel_list = kernel_list

        self.feature_outsize = 2 if include_features else 1
        
        self.blocks = self._make_block_enc_combo(block_list=block_list, kernel_list=kernel_list)
        self.heads = self._make_per_kernel_linear(kernel_list=kernel_list)
                
        heads_layers_out: OrderedDict[str, nn.Module] = OrderedDict()
        
        heads_layers_out["pre_logits"] = nn.Linear(512*(len(self.unique_kernels)+1) + 512, representation_size)
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
        
    
    #from https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L143
    
    def _make_buckets(self, batch_size:int) -> Dict[int,torch.Tensor]:
        buckets:Dict[str:torch.Tensor] = {}
        
        for kernel in self.unique_kernels:
            buckets.update({kernel:torch.tensor((0,),device="cuda")})
            
        return buckets
    
    def _make_per_kernel_linear(self, kernel_list:List[int]) -> Dict[int,nn.Module]:
        heads:Dict[int:nn.Module] = {}
        unique_kernels:List[int] = []
        
        for kernel in kernel_list:
            if kernel not in unique_kernels:
                unique_kernels.append(kernel)
        
        self.unique_kernels = unique_kernels
        
        for kernel in unique_kernels:
            heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
            
            heads_layers["pre_logits"] = nn.Linear(self.per_kernel_hidden_sizes[kernel], self.representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(self.representation_size, 512)

            head = nn.Sequential(heads_layers).to(device="cuda")
            heads.update({kernel: head})
            
        return heads
        
    def _make_block_enc_combo(self, block_list:List[int], kernel_list:List[int]) -> Dict[str,Dict[str,Union[nn.Module, int]]] :
        block_dict:Dict[str:Dict[str:Union(nn.Module, int)]] = {}
        per_kernel_hidden_size_total:Dict[int,int] = {}
        # print(seq_length)
        in_channel = 3
        for idx, (block, kernel) in enumerate(zip(block_list, kernel_list)):
            layers: List[nn.Module] = []
            
            for v in block:
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=v, kernel_size=kernel, padding="same")
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True), nn.Dropout2d(p=self.dropout)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True), nn.Dropout2d(p=self.dropout)]
                    
                in_channel = v
            block_out_size = in_channel * self.image_size**2
            seq_length = int((self.image_size // self.patch_size)**2)
            # seq_length += 1
            p_h = int(self.image_size//self.patch_size)
            p_w = int(self.image_size//self.patch_size)
            hidden_size = int(block_out_size // (p_h*p_w))
            if kernel not in per_kernel_hidden_size_total.keys():
                per_kernel_hidden_size_total.update({kernel:hidden_size})
            else:
                per_kernel_hidden_size_total[kernel] += hidden_size
            # print(seq_length)
            # print(hidden_size)
            block_dict.update({f'k{kernel}_block{idx}': {"param": nn.Parameter(torch.zeros(1,1,hidden_size)),
                                                          "sequential": nn.Sequential(*layers).to(device="cuda"), 
                                                          "encoder": Encoder(seq_length=seq_length, 
                                                                    num_layers=self.num_layer, 
                                                                    num_heads=self.num_heads,
                                                                    hidden_dim=hidden_size, 
                                                                    mlp_dim=self.mlp_dim, 
                                                                    dropout=self.dropout_enc, 
                                                                    attention_dropout=self.dropout_atten,
                                                                    norm_layer=self.norm_layer).to(device="cuda"),
                                                          "hidden_size": hidden_size,
                                                          "kernel_size": kernel}})     
        self.per_kernel_hidden_sizes = per_kernel_hidden_size_total    
        layers: List[nn.Module] = []
        
        conv2d = nn.Conv2d(in_channels=in_channel, out_channels=self.outsize, kernel_size=3)
        if self.batch_norm:
            layers += [conv2d, nn.BatchNorm2d(self.outsize), nn.ReLU(inplace=True), nn.Dropout2d(p=self.dropout)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True), nn.Dropout2d(p=self.dropout)]
            
        block_dict.update({'final':{"sequential":nn.Sequential(*layers).to(device="cuda")}})
        
        return block_dict
                
    
    def forward(self, img:torch.Tensor, feature:torch.Tensor) -> torch.Tensor:
        x = img
        n = x.shape[0]
        buckets = self._make_buckets(n)
        for key, block in self.blocks.items():
            first = True
            if key != 'final':
                # print(block["hidden_size"])
                x = block["sequential"](x)
                enc_x = x.clone()
                enc_x = enc_x.reshape(n, block["hidden_size"], -1)
                enc_x = enc_x.permute(0,2,1)
                enc_x = block["encoder"](enc_x)
                # print(f'shape:{enc_x.shape} | hidden size: {block["hidden_size"]} | enc_x[:,0]:{enc_x[:,0].shape}')
                # print(len(buckets[block["kernel_size"]]))
                if len(buckets[block["kernel_size"]]) == 1:
                    buckets[block["kernel_size"]] = enc_x[:,0]
                else:
                    buckets[block["kernel_size"]] = torch.concat((buckets[block["kernel_size"]], enc_x[:,0]), dim=1)
            else:
                x = block["sequential"](x)
            
        out_head_x = torch.empty((n, 512)).to(device="cuda")
        for key, head in self.heads.items():
            # print(buckets[key].shape)
            x = head(buckets[key])
            out_head_x = torch.concat((out_head_x, x), dim=1)
        
        
        # feature = torch.cat([x.reshape(n, -1), feature], dim=1)
        feat_out = self.feature_nn(feature)
        # print(out_head_x.shape)
        # print(feat_out.shape)
        out_x = torch.concat((out_head_x, feat_out), dim=1)
        out = self.heads_out(out_x)

        head_out_aux = None
        return out, head_out_aux
    
class CnnBlock(nn.Module):
    """
    ----------------------------------------------CnnBlock----------------------------------------------

    Args:
        block_list:List[int] | List[List[int]]  -- List of ints or List of List of ints, describing the CNN depth and width                
        kernel_size:int | List[int]             -- Int or List of ints depicting kernel size for each CNN block or layer
        prev_channels:int , default -> 3        -- Int for the channel size of the previous Convolutional layer, defaults to 3 for RGB
        dropout:float , default -> 0.1          -- Float for 2d Convolutional layer dropout, defaults to 0.1
        batchnorm:bool , default -> True        -- Bool to include batchnorm or not, defaults to True
        reshape:bool , default -> True          -- Bool to reshape out size to be able to be passed to Encoder, adds positional Parameters as well
        phw:int , default -> 10                 -- Int, used only when reshape is True, is the per patch height and width, used for reshaping as the 3rd dimension, defaults to 10
        hidden_size:int , default -> 512        -- Int, used only when reshape is True, is the hidden size for the Encoder, used for reshaping as the 2nd dimension, defaults to 512
    """
    
    # TODO
    # Add feature to dynamically change hidden and phw sizes
    # this would require a getter so that the caller/parent function (CnnEncHeadBlock) could know the layer out sizes
    # this would also require change in the way and order the nn.Modules are instantiated in CnnEncHeadBlock since instantiating in list
    # while passing into the seqeuntial does not give it space to make the CNN block, calculate the hidden size, and send to Encoder
    
    def __init__(self, 
                 block_list:List[int] | List[List[int]], 
                 kernel_size:int | List[int],
                 prev_channels:int = 3,
                 dropout:float = 0.1,
                 batchnorm:bool = True,
                 reshape:bool = True,
                 hidden_size:int = 512,
                 phw:int = 10,
                 stride:int = 1,
                 padding:int = 0,
                 dilation:int = 1,
                 groups:int = 1,
                 adapt_size:int = 5
                 ):
        super(CnnBlock, self).__init__()
        self.reshape = reshape
        self.phw = phw
        self.hidden_size = hidden_size
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.adapt_size = adapt_size
        # shape (C, H, W)
        # self.shape = [prev_channels, image_size, image_size]
        
        if reshape:
            self.class_token = nn.Parameter(torch.zeros(1,1,hidden_size)).to("cuda")
            
        if isinstance(kernel_size, list):
            self.block:nn.Module = self._make_multi_kernel_block(block_list=block_list, kernel_list=kernel_size, dropout=dropout, batchnorm=batchnorm, prev_channels=prev_channels).to("cuda")
        else:
            self.block:nn.Module = self._make_block(block_list, kernel_size, dropout, batchnorm, prev_channels).to("cuda")
            
        self.block = self.block.to("cuda")
        
    def _make_multi_kernel_block(self, block_list:List[List[int]], kernel_list:List[int], dropout:float, batchnorm:bool, prev_channels:int) -> nn.Module:
        in_channel = prev_channels
        layers: List[nn.Module] = []
        
        for _, (block, kernel) in enumerate(zip(block_list, kernel_list)):
            layers += self._make_block(block_list=block, kernel_size=kernel, dropout=dropout, batchnorm=batchnorm, prev_channels=in_channel)
            in_channel = block[-1]
            
        return nn.Sequential(*layers).to("cuda")
        
    def _make_block(self, block_list:List[int], kernel_size:int, dropout:float, batchnorm:bool, prev_channels:int) -> nn.Module:
        in_channel = prev_channels
        layers: List[nn.Module] = []
        orig_kernel_size = kernel_size
        
        for idx, v in enumerate(block_list):
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=1, padding=1)]
                # self._update_shape(new_hw=self._calculated_maxpool_hw(kernel_size=2, stride=1, padding=1), new_channel=in_channel)
            elif v == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=1, padding=1)]
            elif v == "AD":
                layers += [nn.AdaptiveAvgPool2d((self.adapt_size, self.adapt_size))]
                # self._update_shape(new_hw=self._calculated_avgpool_hw(kernel_size=2, stride=1, padding=1), new_channel=in_channel)
            else:
                v = cast(int, v)
                #used to have padding = "same"
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=v, kernel_size=kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
                # self._update_shape(new_hw=self._calculated_conv2d_hw(kernel_size=kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation), new_channel=v)
                # print(self.shape)
                if batchnorm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True), nn.Dropout2d(p=dropout)]
                in_channel = v
                
                kernel_size = orig_kernel_size if idx == len(block_list)-2 else 3
        return nn.Sequential(*layers).to("cuda")

    def _reshape(self, x, hidden_size, phw) -> torch.Tensor:
        n,_,_,_ = x.shape
        x = x.reshape(n, hidden_size, phw)
        x = x.permute(0,2,1)
        
        batch_class_token = self.class_token.expand(n,-1,-1)
        
        x = torch.cat([batch_class_token, x], dim=1)
        
        return x
    #takes img as input 224x224 in this problems case
    
    def forward(self, x):
        x = self.block(x)
        if self.reshape:
            x = self._reshape(x, self.hidden_size, self.phw)
        return x
        
class EncHead(nn.Module):
    """
    ----------------------------------------------EncHead----------------------------------------------
    Args:
        in_size: int                               -- Int for the size of the input size, same as output of Encoder 
        representation_size:int                    -- Int for the head network width
        out_size:int                               -- Int for output size of the head network, this times the number of blocks is the input size for classifier (plus feature network output size)
        dropout:float , default -> 0.1             -- Float for dropout in the head network, defaults to 0.1
        act:Optional[nn.Module] , default -> None  -- nn.Module for activation layer in head network, default to None which is same as Tanh activation
    """
    def __init__(self, 
                 in_size:int, 
                 representation_size:int, 
                 out_size:int, 
                 dropout:float = 0.1, 
                 act:Optional[nn.Module] = None
                 ):
        super(EncHead, self).__init__()
        
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        
        heads_layers["pre_logits"] = nn.Linear(in_size, representation_size)
        if not act:
            heads_layers["act"] = nn.Tanh()
        else:
            heads_layers["act"] = act
        heads_layers["dropout"] = nn.Dropout1d(p=dropout)
        heads_layers["head"] = nn.Linear(representation_size, out_size)

        self.heads = nn.Sequential(heads_layers)
    
    def forward(self, x):
        return self.heads(x)
    
    
# _________________________ TODO _______________________
# make each cnn block point to encoder block
# make encoder block have the same input size, eg, patch, seq_length, hidden_size using adaptivemaxpool with fixed size
# maybe make encoder not in for loop but outside since it will be the same and non changing
# maybe instead of multiple parallel cnn blocks, have one connected deep one, which occationally outputs values to the encoder --
# after given adaptive layer and possibily a fixed "CNN" layer which has a specified size for each to give a specific size from the adaptive layer
# maybe not include the adaptive specific cnn and adaptive layer in the cnnblock, kind of detaching the output of the cnnblock using the same cnn and adaptive layer
class Cnn_Enc(nn.Module):

    def __init__(self, 
                 kernel_sizes:List[int] | List[List[int]],
                 block_list:List[List[int]] | List[List[List[int]]],
                 cnn_cfg:Dict[str,Union[float, bool]],
                 encoder_cfg:Dict[str,Union[int, float, Callable[..., nn.Module]]],
                 head_cfg:Dict[str,Union[int, float, Optional[nn.Module]]],
                 encoder_cnn_cfg:Dict[str,Union[int, float]],
                 patch_size:int,
                 in_channel:int = 3,
                 adapt_size:int = 24,
                 representation_size:int = 1024,
                 num_classes:int = 6
                 ):
        super(Cnn_Enc, self).__init__()
                
        self.CNNEnc: Dict[str:nn.Module] = self._makeCNNEnc(block_list=block_list,
                                                            kernel_sizes=kernel_sizes,
                                                            cnn_cfg=cnn_cfg,
                                                            patch_size=patch_size,
                                                            in_channel=in_channel)
        
        adapt_shape = adapt_size if isinstance(adapt_size, list) else [adapt_size, adapt_size]
                    
        block_out_size:float  = encoder_cnn_cfg["out_channels"] * adapt_shape[0]*adapt_shape[1]
        seq_length:int        = int((adapt_shape[0]*adapt_shape[1]//patch_size)**2) + 1
        p_h:int               = int(adapt_shape[0]*adapt_shape[1]//patch_size)
        p_w:int               = int(adapt_shape[0]*adapt_shape[1]//patch_size)
        hidden_size:int       = int(block_out_size//(p_h*p_w))
        print(f"hid: {hidden_size} | seq: {seq_length} | ph: {p_h} | pw: {p_w} | phpw: {p_h*p_w} | out: {encoder_cnn_cfg['out_channels']} | adapt: {adapt_shape[0]}{adapt_shape[1]} | block out: {block_out_size}")
        self.encoder_block = nn.Sequential(*[nn.AdaptiveAvgPool2d(output_size=adapt_shape),
                                             CnnBlock(block_list=encoder_cnn_cfg["block_list"], # will only contain one value, it is one layer only
                                                      kernel_size=encoder_cnn_cfg["kernel_size"], # can be changed, but information must not be skewed so kernel size 1 or 3 is optimal
                                                      prev_channels=encoder_cnn_cfg["in_channels"], # Every Cnnblock must have this final cnn layer out_channel
                                                      dropout=encoder_cnn_cfg["dropout"], # might not matter but 0 may be optimal to reduce information loss to encoder
                                                      batchnorm=encoder_cnn_cfg["batchnorm"], # might not matter but no batchnorm to again reduce information loss to encoder
                                                      reshape=encoder_cnn_cfg["reshape"], # MUST BE TRUE
                                                      phw=p_h*p_w,
                                                      hidden_size=hidden_size,
                                                      stride=encoder_cnn_cfg["stride"],
                                                      padding=encoder_cnn_cfg["padding"],
                                                      dilation=encoder_cnn_cfg["dilation"],
                                                      groups=encoder_cnn_cfg["groups"]),
                                             Encoder(seq_length=seq_length,
                                                     num_layers=encoder_cfg["num_layers"],
                                                     num_heads=encoder_cfg["num_heads"],
                                                     hidden_dim=hidden_size,
                                                     mlp_dim=encoder_cfg["mlp_dim"],
                                                     dropout=encoder_cfg["dropout"],
                                                     attention_dropout=encoder_cfg["dropout_atten"],
                                                     norm_layer=encoder_cfg["norm_layer"])]).to("cuda")
        
        self.enc_in_size = head_cfg["head_out"]
        
        self.EncHead = EncHead(in_size=self.enc_in_size + hidden_size,
                                representation_size=head_cfg["representation_size"],
                                out_size=head_cfg["head_out"],
                                dropout=head_cfg["dropout"],
                                act=head_cfg["act"])
        
        self.optimizer = torch.optim.Adam(self.EncHead.parameters(), lr=0.0001)
        self.loss_boost = nn.L1Loss()
        
        self.classifier = nn.Sequential(nn.Linear(in_features=head_cfg["head_out"], out_features=representation_size),
                                        nn.Dropout1d(p=0.1),
                                        nn.Tanh(),
                                        nn.Linear(in_features=representation_size, out_features=num_classes))
    
    def _makeCNNEnc(self, 
                    block_list:List[List[int]] | List[List[List[int]]], 
                    kernel_sizes:List[int] | List[List[int]], 
                    cnn_cfg:Dict[str,Union[bool,float,int]], 
                    patch_size:int | List[int], 
                    in_channel:int
                    ) -> Dict[str,nn.Module]:
        
        # blocks: Dict[str:nn.Module] = {}
        blocks: nn.Module = nn.ModuleList()
        prev_channels:int = in_channel
        
        torch._assert(len(block_list) == len(kernel_sizes), f"Expected number of blocks and number of kernel sizes or list of kernel sizes to be the same. Instead got {len(block_list)} blocks and {len(kernel_sizes)} kernel sizes")
        
        if not isinstance(patch_size, list):
            warnings.warn(f"Warning: Given patch size int, patch size extrapolated to be {patch_size} for all blocks")
            patch_size = [patch_size]*len(block_list)
            
        enum_list = zip(block_list, kernel_sizes, patch_size)
        
        
        
        for idx, (cnnblock, kernel, patch) in enumerate(enum_list):
            # ------------------- CAUTION ------------------- 
            # Block out size calculated fron cnnblock[-1], therefore the last value in cnnblock can never be a pool layer
            torch._assert(isinstance(cnnblock[-1], int) | isinstance(cnnblock[-1], list), f"Expected final layer of cnnblock to be int value assossiated with cnn layer, not pool layer")
            # print(cnnblock)
            # blocks[f"CnnBlock_{idx}"] = nn.Sequential(CnnBlock(block_list=cnnblock,
            #                                                    kernel_size=kernel,
            #                                                    prev_channels=prev_channels,
            #                                                    dropout=cnn_cfg["dropout"],
            #                                                    batchnorm=cnn_cfg["batchnorm"],
            #                                                    reshape=False,
            #                                                    phw=patch,
            #                                                    stride=cnn_cfg["stride"],
            #                                                    padding=cnn_cfg["padding"],
            #                                                    dilation=cnn_cfg["dilation"],
            #                                                    groups=cnn_cfg["groups"])).to("cuda")
            blocks.append(nn.Sequential(CnnBlock(block_list=cnnblock,
                                                            kernel_size=kernel,
                                                            prev_channels=prev_channels,
                                                            dropout=cnn_cfg["dropout"],
                                                            batchnorm=cnn_cfg["batchnorm"],
                                                            reshape=False,
                                                            phw=patch,
                                                            stride=cnn_cfg["stride"],
                                                            padding=cnn_cfg["padding"],
                                                            dilation=cnn_cfg["dilation"],
                                                            groups=cnn_cfg["groups"])))
            
            prev_channels = cnnblock[-1][-1] if isinstance(cnnblock[0], list) else cnnblock[-1]
        return blocks
    
    def forward(self, x):
        s = torch.zeros((x.shape[0], self.enc_in_size)).to("cuda")
        # loss = 0
        # -----------------------TODO---------------------
        # use the loss to give insight to which gradient to use
        # dont use backwards and compute degree 0 gradient by hand
        # update and make trees maybe
        # update enchead weights/bias/grads
        # compute graph intact
        for block in self.CNNEnc:
            # self.optimizer.zero_grad()
            x = block(x)
            s = torch.concat([self.encoder_block(x.clone()), s], dim=1)
            s = self.EncHead(s)
            # loss += self.loss_boost(s, y)
            # self.optimizer.step()
        # ret_graph = True if self.training else False
        # loss.backward(retain_graph=ret_graph)
        out = self.classifier(s)
        return out
    
class Feature_Lin(nn.Module):
    def __init__(self, input_size:int, representation_size:int, output_size:int):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, representation_size//2),
                                    nn.Dropout1d(p=0.1),
                                    nn.Tanh(),
                                    nn.Linear(representation_size//2, representation_size),
                                    nn.Tanh(),
                                    nn.Linear(representation_size, representation_size//2),
                                    nn.Dropout1d(p=0.1),
                                    nn.Tanh())
        
        self.classifier = nn.Linear(representation_size//2, output_size)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.classifier(x)
        return x
    
class Emsemble_cnn_feat(nn.Module):
    def __init__(self, 
                 output_size:int, 
                 representation_size:int, 
                 model_cfg:Dict[str,Union[int,float,bool]], 
                 feat_cfg:Dict[str,Union[int,float]]):
        super().__init__()
        
        self.cnn_model = Cnn_Enc(kernel_sizes=model_cfg["kernel_sizes"],
                                 block_list=model_cfg["block_list"],
                                 cnn_cfg=model_cfg["cnn_cfg"],
                                 encoder_cfg=model_cfg["encoder_cfg"],
                                 head_cfg=model_cfg["head_cfg"],
                                 encoder_cnn_cfg=model_cfg["encoder_cnn_cfg"],
                                 patch_size=model_cfg["patch_size"],
                                 in_channel=model_cfg["in_channel"],
                                 adapt_size=model_cfg["adapt_size"],
                                 representation_size=model_cfg["representation_size"],
                                 num_classes=model_cfg["num_classes"])
        
        self.feat_lin = Feature_Lin(input_size=feat_cfg["input_size"],
                                    representation_size=feat_cfg["representation_size"],
                                    output_size=feat_cfg["output_size"])
        
        self.connecter = nn.Bilinear(model_cfg["num_classes"], feat_cfg["output_size"], output_size)
        
        self.classifier = nn.Sequential(nn.Linear(output_size, representation_size),
                                        nn.Tanh(),
                                        nn.Linear(representation_size, output_size))
    
    def forward(self, img, feat, targets):
        x1 = self.cnn_model(img, targets)
        
        x2 = self.feat_lin(feat)
        
        x = self.connecter(x1, x2)
        out = self.classifier(x)
        return out
# encoder_cfg contains all parameter cfg for encoder except for hidden_size and seq_length which are computed in house on the fly
# cnn_cfg contains all parameter cfg for CnnBlock except for block_list and kernel_size (AKA only contains dropout and batchnorm)
    
class CnnEncHeadBlock(nn.Module):
    """
    ----------------------------------------------CnnEncHeadBlock----------------------------------------------
    Parameters:
        kernel_sizes:List[int] | List[List[int]]                           -- List of ints describing block wise kernel sizes, can be two types
        block_list:List[List[int]] | List[List[List[int]]]                 -- List of ints describing CNN block architecture, can be two types
        cnn_cfg:Dict[str,Union[float, bool]]                               -- Dict for generic CNN configuration, indepth description below in caller
        encoder_cfg:Dict[str,Union[int, float, Callable[..., nn.Module]]]  -- Dict for generic Encoder configuration, indepth descriptoin below in caller
        head_cfg:Dict[str,Union[int, float, Optional[nn.Module]]]          -- Dict for generic Encoder head configuration, indepth desciption below in caller
        patch_size:int | List[int]                                         -- List of ints describing block wise patch sizes, can be two types
        image_size:int , default -> 224                                    -- Int describing image size, defaults to 224 pixel height and width
        in_channel:int , default -> 3                                      -- Int describing the input channel number, defaults to 3 for RGB
        
    Returns:
        _type_: torch.Tensor
        
    Description:
        A generator for CNN-Encoder-Head blocks
        Returns torch.Tensor type for given input
        Stores nn.Sequential() of blocks of CNN, Encoder, Head
        
        CNN-Encoder-Head Architecture is based off ViT model
        Mainly the positional encoding of the CNN output
        
        Changes to architecture to allow parallized shallow CNN-Encoder models to train on 
        different kernel sizes, patch sizes, and CNN architecture while still incorporating ViT CNN concept
    """
    def __init__(self,
                 kernel_sizes:List[int] | List[List[int]],
                 block_list:List[List[int]] | List[List[List[int]]],
                 cnn_cfg:Dict[str,Union[float, bool]],
                 encoder_cfg:Dict[str,Union[int, float, Callable[..., nn.Module]]],
                 head_cfg:Dict[str,Union[int, float, Optional[nn.Module]]],
                 patch_size:int | List[int],
                 image_size:int = 224,
                 in_channel:int = 3,
                 _priority_type:str = "patch",
                 _recalc:bool = False
                 ):
        super(CnnEncHeadBlock, self).__init__()
        self._priorty_type = _priority_type
        self.hidden_limit = 512
        self.patch_limit = 24
        self.channel_limit = 128
        self._recalc = _recalc
        
        self.CEHblock:nn.Module = self._makeCEHblock(block_list, kernel_sizes, cnn_cfg, encoder_cfg, patch_size, image_size, in_channel, head_cfg)
    
    def _makeCEHblock(self, block_list:List[List[int]] | List[List[List[int]]], kernel_sizes:List[int] | List[List[int]], cnn_cfg, encoder_cfg, patch_size:int | List[int], image_size, in_channel, head_cfg) -> int:
        blocks: OrderedDict[str,nn.Module] = OrderedDict()
        prev_channels:int = in_channel
        
        torch._assert(len(block_list) == len(kernel_sizes), f"Expected number of blocks and number of kernel sizes or list of kernel sizes to be the same. Instead got {len(block_list)} blocks and {len(kernel_sizes)} kernel sizes")
        
        if not isinstance(patch_size, list):
            warnings.warn(f"Warning: Given patch size int, patch size extrapolated to be {patch_size} for all blocks")
            patch_size = [patch_size]*len(block_list)
            
        enum_list = zip(block_list, kernel_sizes, patch_size)
        
        for idx, (cnnblock, kernel, patch) in enumerate(enum_list):
            # ------------------- CAUTION ------------------- 
            # Block out size calculated fron cnnblock[-1], therefore the last value in cnnblock can never be a pool layer
            torch._assert(isinstance(cnnblock[-1], int) | isinstance(cnnblock[-1], list), f"Expected final layer of cnnblock to be int value assossiated with cnn layer, not pool layer")
            
            if isinstance(cnnblock[-1], list):
                out_size = cnnblock[-1][-1]
            else:
                out_size = cnnblock[-1]
                
            block_out_size:float  = out_size * image_size**2
            seq_length:int        = int((image_size//patch)**2) + 1
            p_h:int               = int(image_size//patch)
            p_w:int               = int(image_size//patch)
            hidden_size:int       = int(block_out_size//(p_h*p_w))
            
            
            # -------------------------TODO-----------------------
            # make hidden, channel, and patch recalculable
            # most ground work functions made, the brains is not implemented
            # no new ideas currently, as long as self._recalc is False, the network will work
            # if pool layers are not used
            if self._recalc:
                new_patch_size, new_hidden_size, new_channel = self._balance_hidden_patch_channel(patch_sizes=patch, 
                                                                                                    hidden_size=hidden_size, 
                                                                                                    kernel_sizes=kernel, 
                                                                                                    block_list=cnnblock,
                                                                                                    padding=cnn_cfg["padding"],
                                                                                                    dilation=cnn_cfg["dilation"],
                                                                                                    stride=cnn_cfg["stride"],
                                                                                                    in_shape=image_size,
                                                                                                    in_channel=in_channel)
                
                new_cnnblock = cnnblock
                if isinstance(cnnblock[-1], list):
                    new_cnnblock = new_cnnblock[-1][0]
                    for layer in reversed(new_cnnblock[-1]):
                        if not isinstance(layer, str):
                            layer = new_channel
                            break
                else:
                    new_cnnblock = new_cnnblock[-1]
                    for layer in reversed(new_cnnblock):
                        if not isinstance(layer, str):
                            layer = new_channel
                            break


            blocks[f"CEHblock_{idx}_PatchSize_{patch}"] = nn.Sequential(*[CnnBlock(block_list=cnnblock,
                                                                                  kernel_size=kernel,
                                                                                  prev_channels=prev_channels,
                                                                                  dropout=cnn_cfg["dropout"],
                                                                                  batchnorm=cnn_cfg["batchnorm"],
                                                                                  reshape=cnn_cfg["reshape"],
                                                                                  phw=patch,
                                                                                  hidden_size=hidden_size,
                                                                                  stride=cnn_cfg["stride"],
                                                                                  padding=cnn_cfg["padding"],
                                                                                  dilation=cnn_cfg["dilation"],
                                                                                  groups=cnn_cfg["groups"]),
                                                                          Encoder(seq_length=seq_length,
                                                                                  num_layers=encoder_cfg["num_layers"],
                                                                                  num_heads=encoder_cfg["num_heads"],
                                                                                  hidden_dim=hidden_size,
                                                                                  mlp_dim=encoder_cfg["mlp_dim"],
                                                                                  dropout=encoder_cfg["dropout"],
                                                                                  attention_dropout=encoder_cfg["dropout_atten"],
                                                                                  norm_layer=encoder_cfg["norm_layer"]),
                                                                          EncHead(in_size=hidden_size,
                                                                                  representation_size=head_cfg["representation_size"],
                                                                                  out_size=head_cfg["head_out"],
                                                                                  dropout=head_cfg["dropout"],
                                                                                  act=head_cfg["act"])])
        return nn.Sequential(blocks)
    
    def _balance_hidden_patch_channel(self, 
                                      patch_size:int, 
                                      hidden_size:int,
                                      kernel_sizes:int | List[int], 
                                      block_list:List[int] | List[List[int]], 
                                      padding:int, 
                                      dilation:int, 
                                      stride:int, 
                                      in_shape:int = 224, 
                                      in_channel:int = 3) -> int:
        if not isinstance(block_list[0], list):
            return self._calculate_HPC(patch_size, 
                                       hidden_size=hidden_size,
                                       last_shape=self._get_last_of_block(block_list=block_list, 
                                                                      kernel_size=kernel_sizes, 
                                                                      padding=padding, 
                                                                      dilation=dilation, 
                                                                      stride=stride, 
                                                                      in_channel=in_channel, 
                                                                      in_shape=in_shape))
        else:
            last_shape = self._get_last_shape(block_list=block_list, 
                                               kernel_size=kernel_sizes, 
                                               padding=padding, 
                                               dilation=dilation, 
                                               stride=stride, 
                                               in_channel=in_channel, 
                                               in_shape=in_shape)

            return self._calculate_HPC(patch=patch_size, last_shape=last_shape, hidden_size=hidden_size)
                
    def _calculate_HPC(self, patch:int, last_shape:List[int], hidden_size:int) -> List[int]:
        #returns list of int ordered [HiddenSize, PatchSize, ChannelSize]
        if self._priorty_type == "patch":
            return self._calc_patch_priority(patch=patch, last_shape=last_shape, hidden_size=hidden_size)
        elif self._priorty_type == "hidden":
            return self._calc_hidden_priority(hidden_size=hidden_size, last_shape=last_shape)
        elif self._priorty_type == "channel":
            return self._calc_channel_priority(last_shape=last_shape, patch=patch)
    
    def _calc_patch_priority(self, patch, last_shape, hidden_size) -> List[int]:
        # This calculates the hidden and channel size with a set patch size
        # It will start changing the channel size first, if it cannot be done 
        # with channel size alone it will change hidden size
        in_limit = False
        hid_size = hidden_size
        while in_limit:
            # If numerator is even and denominator is even, then does not matter what parity x is
            if hid_size*patch % 2 == 0 and last_shape[1]**2 % 2 == 0:
                x = 1
                incrementor = 1
            # If numerator is odd and denominator is even, then the parity of x must be even
            elif hid_size*patch % 2 == 0 and last_shape[1]**2 % 2 == 1:
                x = 2
                incrementor = 2
            # If numerator is even and denominator is odd, then break because it will never evenly divide whatever number numerator is multiplied by 
            elif hid_size*patch % 2 == 1 and last_shape[1]**2 % 2 == 0:
                break
            
            while (x*last_shape[1]**2) % (hid_size*patch) != 0 and x < self.channel_limit+2:
                x+=incrementor
            
            if x > self.channel_limit:
                if hid_size < self.hidden_limit:
                    #takes small step towards hidden_limit
                    hid_size += 1
                    if hid_size*patch % 2 != last_shape[1]**2 % 2:
                        # hid_size += max(hid_size*patch % 2, last_shape[1]**2 % 2)
                        hid_size += 1
                else:
                    warnings.warn("Cannot compute channel size via patch priority, with fixed patch and diff hid. Please choose another priority type or change hidden size")
                    break
            else:
                in_limit = True
        return [hid_size, patch, x]
                
    def _calc_hidden_priority(self, patch, last_shape, hidden_size) -> List[int]:
        # This calculates the hidden and channel size with a set patch size
        # It will start changing the channel size first, if it cannot be done 
        # with channel size alone it will change hidden size
        in_limit = False
        hid_size = hidden_size
        while in_limit:
            # If numerator is even and denominator is even, then does not matter what parity x is
            if hid_size*patch % 2 == 0 and last_shape[1]**2 % 2 == 0:
                x = 1
                incrementor = 1
            # If numerator is odd and denominator is even, then the parity of x must be even
            elif hid_size*patch % 2 == 0 and last_shape[1]**2 % 2 == 1:
                x = 2
                incrementor = 2
            # If numerator is even and denominator is odd, then break because it will never evenly divide whatever number numerator is multiplied by 
            elif hid_size*patch % 2 == 1 and last_shape[1]**2 % 2 == 0:
                x = self.channel_limit*2
            
            while (last_shape[0]*last_shape[1]**2) % (x*patch) != 0 and x < self.hidden_limit+2:
                x+=incrementor
            
            if x > self.hidden_limit:
                if patch < self.patch_limit:
                    #takes small step towards hidden_limit
                    patch += 1
                    if patch % 2 != last_shape[0]*last_shape[1]**2 % 2:
                        # hid_size += max(hid_size*patch % 2, last_shape[1]**2 % 2)
                        patch += 1
                else:
                    warnings.warn("Cannot compute channel size via patch priority, with fixed patch and diff hid. Please choose another priority type or change hidden size")
                    break
            else:
                in_limit = True
        return [hid_size, patch, x]
    
    def _calc_channel_priority(self, patch, last_shape, hidden_size) -> List[int]:
        # This calculates the hidden and channel size with a set patch size
        # It will start changing the channel size first, if it cannot be done 
        # with channel size alone it will change hidden size
        in_limit = False
        hid_size = hidden_size
        while in_limit:
            # If numerator is even and denominator is even, then does not matter what parity x is
            if hid_size*patch % 2 == 0 and last_shape[1]**2 % 2 == 0:
                x = 1
                incrementor = 1
            # If numerator is odd and denominator is even, then the parity of x must be even
            elif hid_size*patch % 2 == 0 and last_shape[1]**2 % 2 == 1:
                x = 2
                incrementor = 2
            # If numerator is even and denominator is odd, then break because it will never evenly divide whatever number numerator is multiplied by 
            elif hid_size*patch % 2 == 1 and last_shape[1]**2 % 2 == 0:
                x = self.channel_limit*2
            
            while (last_shape[0]*last_shape[1]**2) % (x*patch) != 0 and x < self.hidden_limit+2:
                x+=incrementor
            
            if x > self.hidden_limit:
                if patch < self.patch_limit:
                    #takes small step towards hidden_limit
                    patch += 1
                    if patch % 2 != last_shape[0]*last_shape[1]**2 % 2:
                        # hid_size += max(hid_size*patch % 2, last_shape[1]**2 % 2)
                        patch += 1
                else:
                    warnings.warn("Cannot compute channel size via patch priority, with fixed patch and diff hid. Please choose another priority type or change hidden size")
                    break
            else:
                in_limit = True
        return [hid_size, patch, x]
    
    def _get_last_shape(self, 
                        block_list:List[List[int]], 
                        kernel_sizes:List[int], 
                        padding:int, 
                        dilation:int, 
                        stride:int, 
                        in_channel:int = 3, 
                        in_shape:int = 224
                        ) -> list[int]:

        in_shape = in_shape
        for _, (block, kernel) in enumerate(zip(block_list, kernel_sizes)):
            in_shape = self._get_last_of_block(block_list=block, 
                                               kernel_size=kernel, 
                                               padding=padding, 
                                               dilation=dilation, 
                                               stride=stride, 
                                               in_channel=in_channel[0], 
                                               in_shape=in_shape[1:])
            
        return in_shape

    def _get_last_of_block(self, block_list, kernel_size, padding, dilation, stride, in_channel, in_shape) -> list[int]:
        shape = [in_channel, in_shape, in_shape]
        for v in block_list:
            if v == "M":
                hw = self._calculated_maxpool_hw(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, shape=shape[1])
            elif v == "A":
                hw = self._calculated_avgpool_hw(kernel_size=kernel_size, stride=stride, padding=padding, shape=shape[1])
            else:
                v = cast(int, v)
                shape[0] = v
                hw = self._calculated_conv2d_hw(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, shape=shape[1])
            shape[1] = hw
            shape[2] = hw
        return shape
    
    def _calculated_conv2d_hw(self, kernel_size, shape, stride = 1, padding = 0, dilation = 1) -> int:
        return math.floor((shape + (2 * padding) - dilation*(kernel_size-1) - 1) / stride) + 1
    
    def _calculated_maxpool_hw(self, kernel_size, shape, padding = 0, dilation = 1, stride = 1) -> int:
        return math.floor((shape + (2 * padding) - dilation*(kernel_size -1) - 1) / stride ) + 1
    
    def _calculated_avgpool_hw(self, kernel_size, shape, padding = 0, stride = 1) -> int:
        return math.floor((shape + 2 * padding - kernel_size) / stride) + 1
    
    def forward(self, x):
        return self.CEHblock(x)
    
    
class Kernel_wise_parallel_Vit(nn.Module):
    """
    -------------------------------------------- INPUT DICTIONARY DESCRIPTION --------------------------------------------
    CnnEncHeadBlock -> Dict[str:Union(List[int], List[List[int]], int, Optional[int])]
    keys: 
        kernel_sizes:List[int], 
        block_list:List[List[int]], 
        patch_size:int, 
        image_size:int, 
        in_channel:int

    cnn_cfg -> Dict[str:Union(float, bool)]
    keys:  
        dropout:float, 
        batchnorm:bool, 
        reshape:bool

    encoder_cfg -> Dict[str:Union(int, float, Callable[..., torch.nn.Module])]
    keys: 
        num_layers:int, 
        num_heads:int, 
        mlp_dim:int, 
        dropoput:float, 
        dropout_atten:float, 
        norm_layer:Optional[int]

    head_cfg -> Dict[str:Union(int, float, Optional[nn.Module])]
    keys: 
        in_size:int, 
        representation_size:int, 
        out_size:int, 
        dropout:float, 
        act:Optional[nn.Module]

    feat_cfg -> Dict[str:Union(bool, int, float)]
    keys: 
        include_features:bool, 
        representation_size:int, 
        dropout:float, 
        out_size:int
    """
    def __init__(self,
                 CnnEncHeadBlock_cfg:Dict[str,Union[List[int], List[List[int]], int, Optional[int]]],
                 cnn_cfg:Dict[str,Union[float, bool]],
                 encoder_cfg:Dict[str,Union[int, float, Callable[..., torch.nn.Module]]],
                 head_cfg:Dict[str,Union[int, float, Optional[nn.Module]]],
                 feat_cfg:Dict[str,Union[bool, int, float]],
                 classifier_size:int = 1024,
                 classifier_dropout:float = 0.1,
                 num_classes:int = 6
                 ):
        super(Kernel_wise_parallel_Vit, self).__init__()
        self.include_features = feat_cfg["include_features"]
        
        self.CEHblock = CnnEncHeadBlock(kernel_sizes=CnnEncHeadBlock_cfg["kernel_sizes"],
                                        block_list=CnnEncHeadBlock_cfg["block_list"],
                                        cnn_cfg=cnn_cfg,
                                        encoder_cfg=encoder_cfg,
                                        head_cfg=head_cfg,
                                        patch_size=CnnEncHeadBlock_cfg["patch_size"],
                                        image_size=CnnEncHeadBlock_cfg["image_size"],
                                        in_channel= CnnEncHeadBlock_cfg["in_channel"])
        
        if feat_cfg["include_features"]:
            self.feature_nn = nn.Sequential(
                nn.Linear(feat_cfg["feat_size"], feat_cfg["representation_size"]),
                nn.ReLU(True),
                nn.Dropout(p=feat_cfg["dropout"]),
                nn.Linear(feat_cfg["representation_size"], feat_cfg["out_size"])
            )
        classifier_insize = head_cfg["head_out"] * len(CnnEncHeadBlock_cfg["kernel_sizes"]) + feat_cfg["out_size"] if feat_cfg["include_features"] else head_cfg["head_out"] * len(CnnEncHeadBlock_cfg["kernel_sizes"])
        self.classifier = nn.Sequential(
            nn.Linear(classifier_insize, classifier_size),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(classifier_size, classifier_size),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(classifier_size, num_classes)
        )     
    
    def forward(self, x, features):
        combine = None
        # print(self.CEHblock.CEHblock.CEHblock_0)
        # print(self.CEHblock.CEHblock._)
        for idx, block_name in enumerate(self.CEHblock.CEHblock._modules):
            # print(block)
            # print(self.CEHblock.CEHblock._modules[block])
            if combine == None:
                combine = self.CEHblock.CEHblock._modules[block_name](x)
            else:
                combine = torch.cat([combine, self.CEHblock.CEHblock._modules[block_name](x)], dim=1)
        
        if self.include_features:
            feat_x = self.feature_nn(features)

            x = torch.cat([combine, feat_x], dim=1)
        else:
            x = combine
            
        x = self.classifier(x)
        
        return x
    
class R2Loss(nn.Module):
            def __init__(self, use_mask=False):
                super(R2Loss, self).__init__()
                self.use_mask = use_mask

            def forward(self, y_pred, y_true):
                if self.use_mask:
                    mask = (y_true != -1)
                    y_true = torch.where(mask, y_true, torch.zeros_like(y_true))
                    y_pred = torch.where(mask, y_pred, torch.zeros_like(y_pred))

                SS_res = torch.sum((y_true - y_pred)**2, dim=0)  # (B, C) -> (C,)
                SS_tot = torch.sum((y_true - torch.mean(y_true, dim=0))**2, dim=0)  # (B, C) -> (C,)
                r2_loss = SS_res / (SS_tot + 1e-6)  # (C,)
                return torch.mean(r2_loss)  # ()
            
class Weighted_Emsemble(nn.Module):
        
    def __init__(self, cfgs:List[Dict], num_classes:int):
        super().__init__()  
        self.num_classes = num_classes
        # self.ModelList = nn.ModuleList([Cnn_Enc(kernel_sizes=        cfg["kernel_sizes"],
        #                                         block_list=          cfg["block_list"],
        #                                         cnn_cfg=             cfg["cnn_cfg"],
        #                                         encoder_cfg=         cfg["encoder_cfg"],
        #                                         head_cfg=            cfg["head_cfg"],
        #                                         encoder_cnn_cfg=     cfg["encoder_cnn_cfg"],
        #                                         patch_size=          cfg["patch_size"],
        #                                         in_channel=          cfg["in_channel"],
        #                                         adapt_size=          cfg["adapt_size"],
        #                                         representation_size= cfg["representation_size"],
        #                                         num_classes=         cfg["num_classes"]) for cfg in cfgs])
        adapt_size = cfgs['adapt_size']
        
        self.ModList = nn.ModuleList([nn.Sequential(CnnBlock(reshape=True,
                                                             block_list=          cfg["block_list"], # will only contain one value, it is one layer only
                                                             kernel_size=         cfg["kernel_sizes"], # can be changed, but information must not be skewed so kernel size 1 or 3 is optimal
                                                             prev_channels=       cfg["in_channels"], # Every Cnnblock must have this final cnn layer out_channel
                                                             dropout=             cfg["dropout"], # might not matter but 0 may be optimal to reduce information loss to encoder
                                                             batchnorm=           cfg["batchnorm"], # might not matter but no batchnorm to again reduce information loss to encoder
                                                             stride=              cfg["stride"],
                                                             padding=             cfg["padding"],
                                                             dilation=            cfg["dilation"],
                                                             groups=              cfg["groups"],
                                                             adapt_size=adapt_size,
                                                             hidden_size = 64,
                                                             phw=4)) for cfg in cfgs['model_cfg']])
        
        self.encoder = Encoder(seq_length=          4+1,
                               num_layers=          cfgs['encoder_cfg']["num_layers"],
                               num_heads=           cfgs['encoder_cfg']["num_heads"],
                               hidden_dim=          64,
                               mlp_dim=             cfgs['encoder_cfg']["mlp_dim"],
                               dropout=             cfgs['encoder_cfg']["dropout"],
                               attention_dropout=   cfgs['encoder_cfg']["dropout_atten"],
                               norm_layer=          cfgs['encoder_cfg']["norm_layer"])
        
        self.EncHead = EncHead(in_size=             64,
                               representation_size= cfgs['head_cfg']["representation_size"],
                               out_size=            cfgs['head_cfg']["head_out"],
                               dropout=             cfgs['head_cfg']["dropout"],
                               act=                 cfgs['head_cfg']["act"])
        
        self.cnn_embedder = nn.Sequential(nn.Linear(1*adapt_size**2, 512), #adapt size should be small, like 5x5 -> 10x10 max
                                          nn.ReLU6(),
                                          nn.Dropout(p=0.1),
                                          nn.Linear(512, 32))
        
        # self.bilinear_connector = nn.Bilinear(32, 32, 128)
        self.bilin_modlist = nn.ModuleList([nn.Bilinear(32,32,64) for _ in range(len(cfgs))])
        self.bilin_seq = nn.Sequential(nn.Linear(64, 256),
                                       nn.ReLU6(),
                                       nn.Dropout1d(p=0.1),
                                       nn.Linear(256, 32))
        
        self.image_feat_bilin = nn.Bilinear(32, 32, 128)
        self.image_feat_connector = nn.Sequential(nn.Linear(128, 64),
                                                  nn.ReLU6(),
                                                  nn.Dropout(p=0.1),
                                                  nn.Linear(64, 12),
                                                  )
        
        # for model in self.ModelList:
        #     model.classifier = nn.Identity()
        
        self.model_weights = nn.Parameter(torch.ones(len(cfgs), device='cuda'))
        self.img_weights = nn.Parameter(torch.ones(32, device='cuda'))
        self.feat_weights = nn.Parameter(torch.ones(32, device='cuda'))
        self.class_token = nn.Parameter(torch.zeros(1, 1, 128, device='cuda'))
        
        # self.head_out = cfgs[0]["head_cfg"]["head_out"]
        # self.hidden_dim = cfgs['cnn_cfgs']['representation_size'] + cfgs[0]["head_cfg"]["head_out"]
        self.feat_nn = nn.Sequential(nn.Linear(163, 512),
                                     nn.ReLU(),
                                     nn.Dropout1d(0.1),
                                    #  nn.BatchNorm1d(512),
                                     nn.Linear(512,128),
                                     nn.ReLU(),
                                     nn.Dropout1d(0.1),
                                    #  nn.BatchNorm1d(128),
                                     nn.Linear(128, 32))
        
        self.feat_classifier = nn.Sequential(nn.Linear(32, 128),
                                             nn.Linear(128,128),
                                             nn.Linear(128,6))
        self.feat_embedder = nn.Linear(6, 32)
        self.head_embedder = nn.Linear(6, 32)
        # self.EncHead_aux = EncHead(in_size=             128,
        #                            representation_size= cfgs[0]['head_cfg']["representation_size"],
        #                            out_size=            cfgs[0]['head_cfg']["head_out"],
        #                            dropout=             cfgs[0]['head_cfg']["dropout"],
        #                            act=                 cfgs[0]['head_cfg']["act"])
        
        # self.classifier = nn.Linear(cfgs[0]["head_cfg"]["head_out"] + 128, num_classes)
        # self.optimizer = torch.optim.Adam(self.classifier.parameters, lr=0.001)
        #these two must be the same embed size
        
    def forward(self, img, feat):
        xs = None
        n = img.shape[0]
        
        xs = self.ModList[0](img)
        # batch_class_token = self.class_token.expand(n,-1,-1)
        # print(f'xs: {xs.shape}')
        # tokenized = torch.cat([batch_class_token, xs], dim=1)
        enced = self.encoder(xs)
        head_out = self.EncHead(enced)
        head_embeded = self.head_embedder(head_out)
        # for idx, model in enumerate(self.ModList):
        #     if xs is None:
        #         # k = model(img).flatten(start_dim=1)
        #         # print(k.shape)
        #         xs = self.cnn_embedder(model(img.clone()).flatten(start_dim=1))  * self.model_weights[idx]
        #     else:
        #         mid = self.cnn_embedder(model(img.clone()).flatten(start_dim=1)) * self.model_weights[idx]
        #         # print(mid.shape)
        #         # print(xs.shape)
        #         xs = self.bilin_modlist[idx](mid, xs)
        #         xs = self.bilin_seq(xs)
        
        #make cnn_enc cfg numclasses 128
        # summed = xs/len(self.ModelList)
        feat_x = self.feat_nn(feat)
        feat_out = self.feat_classifier(feat_x)
        # ------------------TODO---------------
        # connect features after image is passed through encoder
        
        # print(f'xs: {summed.shape} | feat_x: {feat_x.shape} | imgw: {self.img_weights.shape} | feat: {self.feat_weights.shape}')
        # print(f'xs: {summed.shape} | feat_x: {feat_x.shape} | summed: {catted.shape}')
        # catted = torch.concat([summed*self.img_weights, feat_embed_x*self.feat_weights], dim=1)
        # catted = torch.concat([xs*self.img_weights, self.feat_embedder(feat_out)*self.feat_weights], dim=1)
        connected = self.image_feat_bilin(head_embeded*self.img_weights, self.feat_embedder(feat_out)*self.feat_weights)
        connected = self.image_feat_connector(connected)
        
        # batch_class_token = self.class_token.expand(n,-1,-1)
        # connected = connected.reshape(n, 128, -1)
        # connected = connected.permute(0,2,1)
        # tokenized = torch.cat([batch_class_token, connected], dim=1)
        # print(f'bct: {batch_class_token.shape} | ct: {self.class_token.shape} | cat: {catted.shape} | head out: {self.head_out} | tokenized: {tokenized.shape}')
        
        # print(f'tokenized: {tokenized.shape}')
        # enced = self.encoder(tokenized)
        
        # head_out = self.EncHead(enced)
        # aux_out = self.EncHead_aux(enced)
        # out = self.classifier(torch.concat([summed*self.img_weights, feat_x*self.feat_weights], dim=1))
        
        #ouput size: 12, 6, 6
        return connected, head_out, feat_out
    
    
class CNNLIN(nn.Module):
    def __init__(self, cnn_cfg:Dict[str,Union[List[int],List[List[int]],bool,int]]):
        super().__init__()  
        self.cnn_cfg = cnn_cfg
        print(self.cnn_cfg["block_list"])
        self.cnn_layer = make_block_layers(block_list=      self.cnn_cfg["block_list"],
                                           kernel_sizes=    self.cnn_cfg['kernel_sizes'],
                                            batch_norm=     self.cnn_cfg["batchnorm"],
                                            dropout=        self.cnn_cfg["dropout"],
                                            in_channel=     self.cnn_cfg["in_channels"],
                                            outsize=        self.cnn_cfg["outsize"])
        outsize_shape = cnn_cfg["outsize"] * cnn_cfg['adapt_size']**2
        self.cnn_classifier = nn.Sequential(nn.AdaptiveAvgPool2d(cnn_cfg["adapt_size"]),
                                            nn.Flatten(),
                                            nn.Linear(outsize_shape, 256),
                                            nn.Linear(256, 6),
                                            nn.Sigmoid()
                                            )
        
        self.cnn_lin_con = nn.Bilinear(6, 163, 64)
        
        self.lin = nn.Sequential(nn.ReLU6(),
                                 nn.Linear(64, 128),
                                 nn.ReLU6(),
                                 nn.Dropout(0.1),
                                 nn.Linear(128, 64),
                                 nn.ReLU6(),
                                 nn.Dropout(0.1),
                                 nn.Linear(64, 32),
                                 nn.ReLU6(),
                                 nn.Dropout(0.1),
                                 nn.Linear(32, 128),
                                 nn.ReLU6(),
                                 nn.Dropout(0.1),
                                 nn.Linear(128, 64))
        
        self.lin_classifier_head = nn.Sequential(nn.Linear(64, 32),
                                                 nn.Linear(32, 6))
    
    
    
    def forward(self, img, features):
        x = self.cnn_layer(img)
        cnn_out = self.cnn_classifier(x)
        
        connect = self.cnn_lin_con(cnn_out.clone(), features)
        x2 = self.lin(connect)
        
        lin_out = self.lin_classifier_head(x2.clone())
        return lin_out, cnn_out
    
def make_block_layers(block_list: List[List[Union[str,int]]], kernel_sizes:List[int], batch_norm:bool = False, dropout: float = 0.1, in_channel:int = 3, outsize: int = 512, max_kernel_size: int = 3, calc_padding: bool = False) -> nn.Sequential:
    seq:nn.Module = nn.Sequential()
    in_channel = in_channel
    for idx, (block, kernel_size) in enumerate(zip(block_list, kernel_sizes)):
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