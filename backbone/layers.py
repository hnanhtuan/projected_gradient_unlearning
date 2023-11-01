import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    def __init__(self,   
                in_channels, 
                out_channels,              
                kernel_size, 
                padding=0, 
                stride=1, 
                dilation=1,
                groups=1,                                                   
                bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels,
              kernel_size, stride=stride, padding=padding, bias=bias)
        # define the scale v
        size = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
        scale = self.weight.data.new(size, size)
        scale.fill_(0.)
        # initialize the diagonal as 1
        scale.fill_diagonal_(1.)
        # self.scale1 = scale.cuda()
        self.scale1 = nn.Parameter(scale, requires_grad=True)


    def forward(self, input, space1=None):
        if space1 is not None:
            sz =  self.weight.grad.data.size(0)
                
            real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
            # print(real_scale1.type(), space1.type(), self.weight.type())
            norm_project = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
            #[chout, chinxkxk]  [chinxkxk, chinxkxk]
            proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())

            diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space1, space1.transpose(1,0))).view(self.weight.size())

            masked_weight = proj_weight + self.weight - diag_weight 

        else:
            masked_weight = self.weight

        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# Define specific linear layer
class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)

        # define the scale v
        # scale = self.weight.data.new(self.weight.size(1), self.weight.size(1))
        self.next_topk = 400
        scale = self.weight.data.new(self.next_topk, self.next_topk)
        scale.fill_(0.)
        # initialize the diagonal as 1
        scale.fill_diagonal_(1.)
        # self.scale1 = scale.cuda()
        self.scale = nn.Parameter(scale, requires_grad=True)

    def forward(self, input, space=None):
        if space is not None:
            space = space.detach()
            # sz =  self.weight.data.size(0)
            k = space.size(1)
            pad_sz = max(0, k-self.next_topk)
            tmp = torch.ones(k, dtype=input.dtype).cuda()
            tmp[pad_sz:] = 0
            tmp = torch.diag(tmp).detach()
            real_scale = tmp.add(F.pad(self.scale, [pad_sz, 0, pad_sz, 0]))
            norm_project = torch.mm(torch.mm(space, real_scale), space.transpose(1, 0))

            proj_weight = torch.mm(self.weight, norm_project)

            diag_weight = torch.mm(self.weight, torch.mm(space, space.transpose(1,0)))

            masked_weight = self.weight + proj_weight - diag_weight 
        else:
            masked_weight = self.weight

        return F.linear(input, masked_weight, self.bias)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            # model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)