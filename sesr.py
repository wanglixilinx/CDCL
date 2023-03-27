import torch
import sys 
import mlsr_training.sesr_common as common
import copy
import numpy as np
import torch.nn as nn
from collections import OrderedDict
# from .convolution import reset_state, detach_state
# def make_model(args, parent=False):
#     return SESR(args)

def Conv(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    conv_seq = nn.Sequential()
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
    conv_seq.add_module('conv', conv)
    # conv_seq.add_module('bn', nn.BatchNorm2d(out_channels))
    return conv_seq

## Residual Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        res_scale=1, deploy=False):

        super(RCAB, self).__init__()
        self.in_channels = n_feat
        self.groups = 1
        self.res_scale = res_scale
        self.deploy = deploy
        self.se = nn.Identity() 
        r = 16
  
        if deploy:
            self.body_reparam = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=kernel_size, stride=1,
                                      padding=1, dilation=1, groups=1, bias=False)
            
        else:
            self.body_identity = None
            
            self.body_dense = Conv(n_feat, r*n_feat, kernel_size, 1, 1, 1)
            self.body_dense_1x1 = Conv(r*n_feat, n_feat, 1, 1, 0, 1)
            self.body_1x1 = Conv(n_feat, n_feat, 1, 1, 0, 1)
            #print('Rep Block, identity = ', self.body_identity)

    def forward(self, x):
        if hasattr(self, 'body_reparam'):
             return self.body_reparam(x)
        else:
            if self.body_identity is None:
                id_out = 0
            else: 
                id_out = self.body_identity(x)
            y = self.body_dense(x)
            y = self.body_dense_1x1(y)
            return y + self.body_1x1(x) + id_out
      
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_tensor(self.body_dense)
        kernel1x1, bias1x1  = self._fuse_tensor(self.body_1x1)
        kernelid, biasid = self._fuse_tensor(self.body_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,  None#bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])


    def _fuse_tensor(self, branch):
        
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight 
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
        return kernel, None 

    def switch_to_deploy(self):
        if hasattr(self, 'body_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.body_reparam = nn.Conv2d(in_channels=self.body_dense.conv.in_channels, out_channels=self.body_dense.conv.out_channels,
                                         kernel_size=self.body_dense.conv.kernel_size, stride=self.body_dense.conv.stride,
                                         padding=self.body_dense.conv.padding, dilation=self.body_dense.conv.dilation, groups=self.body_dense.conv.groups, bias=False)

        self.body_reparam.weight.data = kernel
        for para in self.parameters():
            para.detach_()
        self.__delattr__('body_dense')
        self.__delattr__('body_1x1')
        if hasattr(self, 'body_identity'):
            self.__delattr__('body_identity')

## RonvGroup
class ConvGroup(nn.Module):
    def __init__(
        self, conv, in_feat, mid_feat, out_feat, kernel_size, deploy=False):

        super(ConvGroup, self).__init__()
        self.deploy = deploy
  
        if deploy:
            self.body_reparam = nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2, dilation=1, groups=1, bias=False)
        else:
            self.body_dense = Conv(in_feat, mid_feat, kernel_size, 1, (kernel_size - 1) // 2, 1)
            self.body_1x1 = Conv(mid_feat, out_feat, 1, 1, 0, 1)
            
    def forward(self, x):
        if hasattr(self, 'body_reparam'):
            return self.body_reparam(x)
        else:
            x = self.body_dense(x)
            x = self.body_1x1(x) 
            return x

    def merge_tensor(self):
        kernel5x5 = self.body_dense.conv.weight
        kernel1x1 = self.body_1x1.conv.weight
        return torch.conv2d(kernel5x5.permute(1, 0, 2, 3), kernel1x1.flip(-1, -2), padding=0).permute(1, 0, 2, 3)

    def switch_to_deploy(self):
        if hasattr(self, 'body_reparam'):
            return
        kernel = self.merge_tensor()
        self.body_reparam = nn.Conv2d(in_channels=self.body_dense.conv.in_channels, out_channels=self.body_1x1.conv.out_channels,
                                         kernel_size=self.body_dense.conv.kernel_size, stride=self.body_dense.conv.stride,
                                         padding=self.body_dense.conv.padding, dilation=self.body_dense.conv.dilation, groups=self.body_dense.conv.groups, bias=False)

        self.body_reparam.weight.data = kernel
        for para in self.parameters():
            para.detach_()
        self.__delattr__('body_dense')
        self.__delattr__('body_1x1')
 
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, res_scale, n_resblocks, deploy):
        super(ResidualGroup, self).__init__()
        self.deploy = deploy 
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, res_scale=1, deploy=self.deploy) \
            for _ in range(n_resblocks)]
        self.body = nn.Sequential(*modules_body)
        self.act = nn.ReLU()

    def forward(self, x):
        res = self.body(x)
        res += x
        return self.act(res)

class SESR(nn.Module):
    def __init__(self, in_channels:int, return_outs:bool, gen_cfg=None, scale:int=2, conv=common.default_conv):
        super(SESR, self).__init__()
        
        # n_resgroups = 11
        # n_resblocks = 1
        # n_feats = 16
        # kernel_size = 3
        # reduction = 16
        # scale = 1
        # act = nn.PReLU()
        # deploy = False
        # res_scale = 1
        # self.deploy = deploy 

        n_resgroups = 7
        n_resblocks = 1
        n_feats = 16
        kernel_size = 3
        reduction = 16
        scale = 1
        act = nn.PReLU()
        deploy = False
        res_scale = 1
        self.deploy = deploy 

        # define head module
        self.head = ConvGroup(conv, in_feat=15, mid_feat=256, out_feat=n_feats, kernel_size=5, deploy=deploy)
                           
        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, res_scale=res_scale, n_resblocks=n_resblocks, deploy=deploy) \
            for _ in range(n_resgroups)]

        self.body = nn.Sequential(*modules_body)
 
        # define tail module
        self.tail = ConvGroup(conv, in_feat=n_feats, mid_feat=256, out_feat=scale*scale*16, kernel_size=5, deploy=deploy)

    def forward(self, x): 
        x = self.head(x)
        res = self.body(x)
        res = res + x
        res = self.tail(res)
        return res 

    def reset(self):
        self.head.reset()
        for m in self.body:
            m.reset()
        self.tail.reset()
    
    def detach(self):
        self.head.detach()
        for m in self.body:
            m.detach()
        self.tail.detach()

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print('Save converted model in: ', save_path)
    return model
