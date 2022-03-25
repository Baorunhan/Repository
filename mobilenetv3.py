import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
import numpy as np
from collections import deque
from torch.autograd import Function

def Enablecertainlayer_fortrain(net,req_name):
    for name,p in net.named_parameters():
        if req_name not in name:
            p.requires_grad=False#固定参数
            print("%s fixed\n"%(name))


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out
		
class SaModule(nn.Module):
    def __init__(self):
        super(SaModule,self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
	
    def forward(self,x):
        out1 = torch.mean(x, dim=1, keepdim=True)
        out2,_ = torch.torch.max(x, dim=1, keepdim=True)
        out = torch.cat([out1, out2], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return x * out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),#全连接层
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=False),#全连接层
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
		
class Block_2(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, samodule, stride):
        super(Block_2, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()#先定义一个空的
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )
         #这里加一个SAmodule吗
        self.sa = samodule
    def forward(self, x):  
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        if self.sa != None:
            out = self.sa(out)
        return out
		
class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()#先定义一个空的
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )
    def forward(self, x):  
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

class MNV3_large2(nn.Module):
    def __init__(self, numclasses):
        super(MNV3_large2, self).__init__()
        self.featurelook = None
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #self.spatialattention =  SaModule()#加入一个空域注意力机制
        self.bneck = nn.Sequential(
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=1),
            #nn.Dropout2d(0.2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), stride=2),
            Block(kernel_size=3, in_size=32, expand_size=128, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), stride=1),
            #nn.Dropout2d(0.3),
            Block(kernel_size=3, in_size=32, expand_size=128, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=2),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=1),
            #nn.Dropout2d(0.5),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=2),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=1),

            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.5),
        )
        self.classifier = nn.Conv2d(in_channels=48, out_channels=numclasses, kernel_size=1,
                      stride=1, padding=0, bias=True)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        #x = self.spatialattention(x)#广播相乘#[64,16,21,17]
        feature = self.bneck(x)
        #self.featurelook = feature.detach()
        output = self.classifier(feature)
		#将多维度tensor平展成一维
        featrue = feature.view(feature.size(0), -1)
        output = output.view(output.size(0), -1)
        return featrue,output
       
class MNV3_large2_v2(nn.Module):
    def __init__(self, numclasses):
        super(MNV3_large2_v2, self).__init__()
        self.featurelook = None
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.spatialattention =  SaModule()#加入一个空域注意力机制
        self.bneck = nn.Sequential(
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=1),
            #nn.Dropout2d(0.2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), stride=2),
            Block(kernel_size=3, in_size=32, expand_size=128, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), stride=1),
            #nn.Dropout2d(0.3),
            Block(kernel_size=3, in_size=32, expand_size=128, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=2),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=1),
            #nn.Dropout2d(0.5),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=2),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=1),

            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.5),
        )
        self.middlelayer = nn.Conv2d(in_channels=48, out_channels=12, kernel_size=1,stride=1,padding=0,bias=True)
        self.classifier = nn.Conv2d(in_channels=12, out_channels=numclasses, kernel_size=1,stride=1, padding=0, bias=True)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.spatialattention(x)#广播相乘#[64,16,21,17]
        #print(x.shape)
        feature = self.bneck(x)
        #self.featurelook = feature.detach()
        feature = self.middlelayer(feature)
        output = self.classifier(feature)
		#将多维度tensor平展成一维
        feature = feature.view(output.size(0), -1)
        output = output.view(output.size(0), -1)
        return feature, output       

class MNV3_large2_v3(nn.Module):
    def __init__(self, numclasses):
        super(MNV3_large2_v3, self).__init__()
        self.featurelook = None
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #self.spatialattention =  SaModule()#加入一个空域注意力机制
        self.bneck = nn.Sequential()#前面4个block加上SAmodule
            
        self.bneck.add_module("0",Block_2(kernel_size=3, in_size=16, expand_size=64, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), samodule = None,stride=2))
        self.bneck.add_module("1",Block_2(kernel_size=3, in_size=24, expand_size=96, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), samodule = None, stride=1))
        self.bneck.add_module("B1",Block_2(kernel_size=3, in_size=24, expand_size=96, out_size=24,#多加一层
                   nolinear=hswish(), semodule=SeModule(24), samodule = None, stride=1))
            #nn.Dropout2d(0.2),
        self.bneck.add_module("2",Block_2(kernel_size=3, in_size=24, expand_size=96, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), samodule = None, stride=2))
        self.bneck.add_module("3",Block_2(kernel_size=3, in_size=32, expand_size=128, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), samodule = None, stride=1))
        self.bneck.add_module("B3",Block_2(kernel_size=3, in_size=32, expand_size=128, out_size=32,#多加一层
                  nolinear=hswish(), semodule=SeModule(32), samodule = None, stride=1))
            #nn.Dropout2d(0.3),
        self.bneck.add_module("4",Block_2(kernel_size=3, in_size=32, expand_size=128, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), samodule=None, stride=2))
        self.bneck.add_module("5",Block_2(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), samodule=None,stride=1))
            #nn.Dropout2d(0.5),
        self.bneck.add_module("6",Block_2(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), samodule=None, stride=2))
        self.bneck.add_module("7",Block_2(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), samodule=None, stride=1))
				  
        self.bneck.add_module("AvgPooling",nn.AdaptiveAvgPool2d(1))
        self.bneck.add_module("Dropout",nn.Dropout2d(0.5))

        self.classifier = nn.Conv2d(in_channels=48, out_channels=numclasses, kernel_size=1,stride=1, padding=0, bias=True)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        #print(x.shape)
        feature = self.bneck(x)
        #self.featurelook = feature.detach()
        output = self.classifier(feature)
		#将多维度tensor平展成一维
        feature = feature.view(output.size(0), -1)
        output = output.view(output.size(0), -1)
        return feature, output


class Uncertainty_block(nn.Module):
    def __init__(self,in_size):
        super(Uncertainty_block, self).__init__()
        self.uncertainty = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=in_size//2, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=in_size//2, out_channels=48, kernel_size=1,
                 stride=1, padding=0, bias=False)
        )
    def forward(self,x):
        return self.uncertainty(x)


class MNV3_large2_uncertainty(nn.Module):
    def __init__(self, numclasses):
        super(MNV3_large2_uncertainty, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bneck = nn.Sequential(
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=1),
            #nn.Dropout2d(0.2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), stride=2),
            Block(kernel_size=3, in_size=32, expand_size=128, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), stride=1),
            #nn.Dropout2d(0.3),
            Block(kernel_size=3, in_size=32, expand_size=128, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=2),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=1),
            #nn.Dropout2d(0.5),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=2),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=1),

            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.5),
        )
        self.uncertainty = Uncertainty_block(576)
        self.classifier = nn.Conv2d(in_channels=48, out_channels=numclasses, kernel_size=1,
                      stride=1, padding=0, bias=True)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        count = 0
        for submodule in self.bneck:
            count += 1
            if count == 8: #第8个block
                x_in = x.view(x.size(0),-1,1,1)#铺成全连接层
                x_uncertainty = self.uncertainty(x_in)
                x = submodule(x)
            else:
                x = submodule(x)

        log_sigma2 = x_uncertainty.view(x_uncertainty.size(0), -1)
        feature = x.view(x.size(0), -1) #这个是mu
        output = self.classifier(x)
		#将多维度tensor平展成一维
        output = output.view(output.size(0), -1)
        return output, feature, log_sigma2


# def test():
#     x = torch.randn(128,1,200,180) 
   
#     net = SVDD(48)
#     #net = MNV3_large2(48)
#     y = net(x)
#     layers = get_parameters_layer(net)
#     #clayers = get_parameters_layer(C)
#     print(layers)
     
#     params = list(net.parameters())
#     k = 0
#     for i in params:
#         l = 1
#         print("该层的结构：" + str(list(i.size())))
#         for j in i.size():
#                 l *= j
#         print("该层参数和：" + str(l))
#         k = k + l
#     print("总参数数量和：" + str(k))
     
#     print(y.detach().numpy()[0])

# if __name__ == '__main__':
#     test()

