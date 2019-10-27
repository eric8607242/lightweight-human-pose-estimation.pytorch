import torch
from torch import nn

class ConvBNRelu(nn.Sequential):
    def __init__(self, input_depth, 
                       output_depth, 
                       kernel,
                       stride,
                       pad,
                       *args,
                       **kwargs):
        super(ConvBNRelu, self).__init__()
        
        operation = nn.Conv2d(input_depth,
                           output_depth,
                           kernel_size=kernel,
                           stride=stride,
                           padding=pad,
                           bias=False)
        self.add_module("conv", operation)
        
        bn_operation = nn.BatchNorm2d(output_depth)
        self.add_module("bn", bn_operation)

        self.add_module("relu", nn.ReLU(inplace=True))

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #self.conv1 = ConvBNRelu(512, 512, 3, 2, 3//2)
        self.conv3 = ConvBNRelu(512, 720, 3, 2, 3//2)
        self.conv4 = ConvBNRelu(720, 720, 3, 1, 3//2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = nn.Sequential(nn.Linear(720, 200),
                                nn.BatchNorm1d(200),
                                nn.ReLU(),
                                nn.Linear(200, 1),
                                nn.Sigmoid())
    def forward(self, x):
        out = x
        #out = self.conv1(out)
        #out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        #out = self.conv5(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
