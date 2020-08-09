import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision import models
import torch.optim as optim
from importlib import import_module

# L2 loss
def MSE(para):
    return nn.MSELoss()

# L1 loss
def L1(para):
    return nn.L1Loss()
    
# GAN loss
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # x: (n,3,256,256)
        c = 3
        h, w = 256, 256
        n_feats = 8
        n_middle_blocks = 6
        self.start_module = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=n_feats, kernel_size=3, stride=1, padding=1), # (n,8,256,256)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            BasicBlock(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1), # (n,8,128,128)
        ) 
        middle_module_list = []
        for i in range(n_middle_blocks):
            middle_module_list.append(BasicBlock(in_channels=n_feats*(2**i), out_channels=n_feats*(2**(i+1)), kernel_size=3, stride=1, padding=1))
            middle_module_list.append(BasicBlock(in_channels=n_feats*(2**(i+1)), out_channels=n_feats*(2**(i+1)), kernel_size=3, stride=2, padding=1))
        self.middle_module = nn.Sequential(*middle_module_list)
        self.end_module = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (n,3,256,256)
        n, _, _, _ = x.shape
        h = self.start_module(x) # (n,8,256,256)
        h = self.middle_module(h) # (n,512,2,2)
        h = h.reshape(n,-1) # (n,2048)
        out = self.end_module(h) # (n,1)
        return out

class GAN(nn.Module):
    def __init__(self, para):
        super(GAN, self).__init__()
        self.device = torch.device('cpu' if para.cpu else 'cuda')
        self.D = Discriminator().to(self.device)
        self.criterion = nn.BCELoss().to(self.device)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=para.lr)
        self.real_label = 1
        self.fake_label = 0
    
    def forward(self, x, y, valid_flag=False):
        self.D.zero_grad()
        b = y.size(0)
        label = torch.full((b,), self.real_label, device=self.device)
        if not valid_flag:
        ############################################
        # update D network: maximize log(D(y) + log(1-D(G(x))))
        # train with all-real batch
            output = self.D(y).view(-1) # forward pass of real batch through D
            errD_real = self.criterion(output, label) # calculate loss on all-real batch
            errD_real.backward() # calculate gradients for D in backward pass 
            ## D_y = output.mean().item()
            # train with all-fake batch
            label.fill_(self.fake_label)
            output = self.D(x.detach()).view(-1) # classify all fake batch with D
            errD_fake = self.criterion(output, label) # calculate D's loss on the all-fake batch
            errD_fake.backward() # calculate gradients for all-fake batch
            ## D_G_x1 = output.mean().item()
            # add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # update D
            self.D_optimizer.step()
        ############################################
        # generate loss for G network: maximize log(D(G(x)))
        label.fill_(self.real_label) # fake labels are all real for generator cost
        # since we just updated D, perform another forward pass of all-fake batch through D
        output = self.D(x).view(-1)
        errG = self.criterion(output, label) # calculate G's loss on this output
        ## D_G_x2 = output.mean().item()
        
        return errG

# Training loss
class Loss(nn.Module):
    def __init__(self, para):
        super(Loss, self).__init__()
        ratios, losses = self.loss_parse(para.loss)
        self.losses_name = losses
        self.ratios = ratios
        self.losses = nn.ModuleList()
        for loss in losses:
            # module = import_module('train.loss')
            # self.losses.append(getattr(module, loss)(para).cuda())
            loss_fn = eval('{}(para).cuda()'.format(loss))
            self.losses.append(loss_fn)
            
    def loss_parse(self, loss_str):
        ratios = []
        losses = []
        str_temp = loss_str.split('|')
        for item in str_temp:
            substr_temp = item.split('*')
            ratios.append(float(substr_temp[0]))
            losses.append(substr_temp[1])
        return ratios, losses
    
    def forward(self, x, y, valid_flag=False):
        if len(x.shape) == 5:
            b,n,c,h,w = x.shape
            x = x.reshape(b*n,c,h,w)
            y = y.reshape(b*n,c,h,w)
        # print(x.shape, y.shape)
        for i in range(len(self.losses)):
            if valid_flag==True and self.losses_name[i] == 'GAN':
                loss_sub = self.ratios[i]*self.losses[i](x,y,valid_flag)
            else:
                loss_sub = self.ratios[i]*self.losses[i](x,y)
            if i==0: loss_all = loss_sub
            else: loss_all += loss_sub
        return loss_all