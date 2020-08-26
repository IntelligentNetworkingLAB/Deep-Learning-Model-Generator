import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


__all__ = ['Conv2dLSQ', 'InputConv2dLSQ', 'LinearLSQ', 'LSQQuantizer']


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LSQQuantizer(t.nn.Module):
    def __init__(self, bit, is_activation=True):
        super(LSQQuantizer,self).__init__()

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.bit = bit
        self.is_activation = is_activation
        self.register_buffer('init_state', torch.zeros(1))        

        if is_activation:
            self.Qn = 0
            self.Qp = 2 ** self.bit - 1
        else:
            self.Qn = -2 ** (self.bit - 1)
            self.Qp = 2 ** (self.bit - 1) - 1

    def forward(self, x):
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.detach().abs().mean() / math.sqrt(self.Qp))
            self.init_state.fill_(1)
            print ("Initializing step-size value ...")
        
        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        self._alpha = grad_scale(self.alpha, g)
        x_q = round_pass((x / self._alpha).clamp(self.Qn, self.Qp))# * self._alpha
       
        return x_q
    
    def get_alpha(self):
        return self._alpha


    def __repr__(self):
        return "LSQQuantizer (bit=%s, is_activation=%s)" % (self.bit, self.is_activation)


   


class Conv2dLSQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=4):


        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.quan_w = LSQQuantizer(bit=bit, is_activation=False)
        self.quan_a = LSQQuantizer(bit=bit, is_activation=True)
        self.quan_b = LSQQuantizer(bit=bit, is_activation=False)
        self.bit = bit

    def forward(self, x):
        
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            inp = self.quan_a(x)
            weigh = self.quan_w(self.weight)
            bias = self.quan_b(self.bias, self.quan_a.get_alpha()* self.quan_w.get_alpha())
            return F.conv2d(self.quan_a(x), self.quan_w(self.weight), self.quan_b(self.bias), self.stride, self.padding, self.dilation, self.groups)*self.quan_a.get_alpha()*self.quan_w.get_alpha()
    




class InputConv2dLSQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=4):


        super(InputConv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.quan_w = LSQQuantizer(bit=bit, is_activation=False)
        self.quan_a = LSQQuantizer(bit=bit, is_activation=False)
        self.bit = bit

    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.quan_w(self.weight), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

class LinearLSQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=4):
        super(LinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.bit = bit
        self.quan_w = LSQQuantizer(bit=bit, is_activation=False)
        self.quan_a = LSQQuantizer(bit=bit, is_activation=True)

        self.quan_b = LSQQuantizer(bit=bit, is_activation=False)
        
    def forward(self, x):
        if self.bit == 32:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(self.quan_a(x), self.quan_w(self.weight), self.bias)*self.quan_a.get_alpha()*self.quan_w.get_alpha()