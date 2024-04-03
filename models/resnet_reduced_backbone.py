import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import  conv1x1, BasicBlock, Bottleneck

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}

class ModifiedResNet(nn.Module):
    def __init__(self, block, layers,img_size=256, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, downscale=False):
        super(ModifiedResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])

        self.avgpool1 = nn.AdaptiveAvgPool2d((img_size//8, img_size//8)) 
        self.avgpool2 = nn.AdaptiveAvgPool2d((img_size//16, img_size//16))
        self.avgpool3 = nn.AdaptiveAvgPool2d((img_size//32, img_size//32))
        self.avgpool4 = nn.AdaptiveAvgPool2d((img_size//64, img_size//64))
        
        self.downscale = downscale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x1 = self.layer1(x)

        x1Reduced=self.avgpool1(x1)

        x2 = self.layer2(x1)
        x2Reduced = self.avgpool2(x2)

        x3 = self.layer3(x2)
        x3Reduced = self.avgpool3(x3)

        x4 = self.layer4(x3)
        x4Reduced = self.avgpool4(x4)


        return x,x1Reduced,x2Reduced,x3Reduced,x4Reduced

    def forward(self, x):
        x,x1,x2,x3,x4= self._forward_impl(x)

        return x,x1,x2


def modified_resnet(arch, block, layers, pretrained, progress,img_size=256, **kwargs):
    model = ModifiedResNet(block, layers,img_size, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load("/home/aquilae/SC/Data/FranceTeinture/models/Teacher_resnet18/resnet18-5c106cde.pth"), strict=False)
    return model


def modified_resnet18(pretrained=True, progress=True,img_size=256, **kwargs):
    return modified_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,img_size=img_size,
                   **kwargs)

def modified_resnet34(pretrained=True, progress=True,img_size=256, **kwargs):
    return modified_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,img_size=img_size,
                   **kwargs)

def modified_resnet50(pretrained=True, progress=True,img_size=256, **kwargs):
    return modified_resnet('resnet50', BasicBlock, [3, 4, 6, 3], pretrained, progress,img_size=img_size,
                   **kwargs)


class ReducedStudent(nn.Module):
    def __init__(self, block, layers,img_size=256, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, downscale=False):
        super(ReducedStudent, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)




        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],stride=2)  # ajout stride=2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])
        
        
        self.downscale = downscale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x4 = self.layer4(x3)

        return x,x1,x2,x3,x4

    def forward(self, x):
        x, x1, x2, x3,x4 = self._forward_impl(x)
        return x,x1,x2 


def reduced_student(arch, block, layers, pretrained, progress,img_size=256, **kwargs):
    model = ReducedStudent(block, layers,img_size=256, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def reduce_student18(pretrained=True, progress=True,img_size=256, **kwargs):
    return reduced_student('resnet18', BasicBlock, [1, 1, 1, 1], pretrained, progress,img_size,
                   **kwargs)


