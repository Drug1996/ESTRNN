import torch
import torch.nn as nn
from importlib import import_module
# from thop import profile


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        model_name = para.model
        module = import_module('model.' + model_name)
        self.model = module.Model(para)

    def __repr__(self):
        return self.model.__repr__() + '\n'

    def profile(self):
        device = torch.device('cpu' if self.para.cpu else 'cuda')
        frames = self.para.frames
        H = self.para.profile_H
        W = self.para.profile_W
        x = torch.randn(1, frames, 3, H, W).to(device)
        profile_flag = True
        flops, params = profile(self.model.to(device), inputs=(x, profile_flag), verbose=False)

        return flops, params


    def forward(self, x):
        return self.model(x)