import torch
import torch.nn as nn
import torch.functional as F
import numpy
import math
import random
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict

class NonLinWeight(nn.Module):
    def __init__(self, function:Callable, shape:int, num_channels:int, batch_first:bool = True, negative:bool = False):
        super(NonLinWeight, self).__init__()
        self.batch_first = batch_first
        self.function = function
        self.negative = negative
        self.num_channels = num_channels
        
        self.mul_shift_weight = nn.Parameter(torch.empty(size=(1,), device='cuda', requires_grad=True), requires_grad=True)
        self.add_shift_weight = nn.Parameter(torch.empty(size=(1,), device='cuda', requires_grad=True), requires_grad=True)
        
        self._create_map(shape)
        
        self.map_param = nn.Parameter(self.map, requires_grad=True)
        
        # print(nn.Parameter(torch.tensor(1.0), requires_grad=True).grad_fn)
        # print(f'pre map_param grad: {self.map_param.grad_fn}')
        # print(f'map_param grad: {self.map_param.grad_fn}')
        # print(torch.__version__)
        # self._nonlin_function()
        # self.map = None
    
    def _create_map(self, size:int) -> torch.Tensor:
        linspace = torch.linspace(-(size//2)+1,size//2, steps=size, device="cuda", requires_grad=True)
        map = torch.outer(linspace, linspace.T)
        map = map.expand(self.num_channels, map.shape[0], map.shape[1])

        start_std = 0.5
        end_std = 0
        step_std = -(start_std-end_std)/self.num_channels

        mean_step = 2/self.num_channels
        mult_vals = torch.normal(mean=torch.arange(-1.,1.,mean_step), std=torch.arange(start_std, end_std, step_std)).to(device='cuda')
        
        rand_perm = torch.randperm(self.num_channels, device='cuda')
        mult_vals = mult_vals[rand_perm]

        multed = torch.einsum('cxy,c->cxy', map, mult_vals)
        
        if self.negative:
            self.map = self.function(multed)
        else:
            self.map = self.function(multed)
        # print(self.map.shape)
        
    def apply_map(self, input:torch.tensor) -> torch.tensor:
        if self.batch_first:
            # print(input[:][:].shape)
            # print(self.map_param)
            expanded = self.map_param.expand(input.shape[0], self.map_param.shape[0], self.map_param.shape[1], self.map_param.shape[2])
            # expanded.retain_grad()
            # print(expanded.shape)
            return torch.mul(input, expanded)
            # print(input.grad_fn)
            # print(input[:][:].shape)
        else:
            # expanded = self.map_param.expand(input.shape[0], self.map_param.shape[0], self.map_param.shape[1])
            # expanded.retain_grad()
            return torch.mul(input, self.map_param)
        # return input
            
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return self.apply_map(input)
    

# ins = torch.ones((10, 1, 224, 224), device='cuda', dtype=torch.float32)
# weighter = NonLinWeight(batch_first=True).to(device='cuda')
# conv = nn.Conv2d(1,4, kernel_size=(3,3), padding='same', device='cuda')
# avg = nn.AvgPool2d(kernel_size=3)
# out2 = avg(conv(ins))
# out = avg(weighter(conv(ins)))
# print(out2.shape)
# print(out2)
# print(out.shape)
# print(out)
# h = torch.empty(10, requires_grad=True)
# emp = torch.empty(20, requires_grad=True, device='cuda')
# linspace = torch.linspace(-(10//2)+1,10//2, steps=20, device="cuda", requires_grad=True)
# map = torch.outer(linspace, linspace.T)
# si = torch.sin(map)
# nsi = -1*torch.sin(map)
# comb = emp*si
# comb2 = emp*nsi
# print(si.grad_fn)
# print(nsi.grad_fn)
# print(comb.grad_fn)
# print(comb2.grad_fn)

# size=10
# linspace = torch.linspace(-(size//2)+1,size//2, steps=size, device="cuda", requires_grad=True)
# map = torch.outer(linspace, linspace.T)
# # print(map.expand(64, 16, map.shape[0], map.shape[1]).shape)
# channel_size = 63

# start_std = 0.5
# end_std = 0
# step_std = -(start_std-end_std)/channel_size

# mean_step = 2/channel_size
# mult_vals = torch.normal(mean=torch.arange(-1.,1.,mean_step), std=torch.arange(start_std, end_std, step_std)).to(device='cuda')
# # print()
# remap = torch.cos(map)*torch.tensor(-1, device='cuda').view(1, 1)
# remap = remap.expand(64, channel_size, remap.shape[0], remap.shape[1])
# # print(remap.shape)

# # print(torch.randint(0,1, (1,)))
# # list_num = torch.arange(1, 10, 1).tolist()
# # random.shuffle(list_num)
# # import random
# # for _ in range(30):
# #     val = list_num.pop()
# #     print(val)
# rand_perm = torch.randperm(channel_size, device='cuda')
# # print(rand_perm)
# # print(mult_vals)
# mult_vals = mult_vals[rand_perm]
# # print(mult_vals[rand_perm])

# multed = torch.einsum('bcxy,c->bcxy', remap, mult_vals)
# print(multed.shape)
# print(multed)
# # permed = torch.permute(remap[:], rand_perm.tolist())