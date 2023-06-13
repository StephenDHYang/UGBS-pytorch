import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo


affine_par = True
class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """
    def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_features, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_features * (len(sizes)//4 + 1), out_features)
        self.relu = nn.ReLU()

    def _make_stage_1(self, in_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, in_features//4, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_features//4, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_features, out_features):
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))
        return bottle

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class FineNet_Basic(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class):
        super(FineNet_Basic, self).__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade-i-1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self._predict(4*lateral_channel, num_class)    

    def _make_layer(self, input_channel, num, output_shape):
        layers = []
        for i in range(num):
            layers.append(BasicBlock(input_channel, 64))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class):
        layers = []
        layers.append(BasicBlock(input_channel, 256))
        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_class))
        return nn.Sequential(*layers)

    def forward(self, x):
        fine_fms = []
        for i in range(4):
            fine_fms.append(self.cascade[i](x[i]))
        out = torch.cat(fine_fms, dim=1) 
        out = self.final_predict(out)
        return out

class CoarseNet_Basic(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(CoarseNet_Basic, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 64,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(64, 64,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(64, 64,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))
        return nn.Sequential(*layers)

    def forward(self, x):
        coarse_fms, coarse_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])            
                coarse_fms.append(feature)
                if i != len(self.channel_settings) - 1:
                    up = feature                 
                feature = self.predict[i](feature)
                coarse_outs.append(feature)                              
            else:             
                feature = self.laterals[i](x[i])
                feature = feature+ up                      
                coarse_fms.append(feature)
                if i == len(self.channel_settings) - 2:
                    up = self.upsamples[i](feature)
                feature = self.predict[i](feature)
                coarse_outs.append(feature)        
        return coarse_fms, coarse_outs

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18_CPN(nn.Module):
    """
    ResNet18_CPN8(CPN Structure for refining)
    """
    def __init__(self, nInputChannels, block, layers=(3, 4, 23, 3), pretrained=True, cascade_pretrain_dir='', criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(ResNet18_CPN, self).__init__()
        self.cascade_pretrain_dir = cascade_pretrain_dir
        self.inplanes = 64
        output_shape = 128
        n_classes = 1
        self.criterion = criterion
        channel_settings = [512, 256, 128, 64]
        self.global_net = CoarseNet_Basic(channel_settings, output_shape, n_classes)
        self.refine_net = FineNet_Basic(channel_settings[-1], output_shape, n_classes)
        self.psp4 = PSPModule(in_features=512, out_features=512, sizes=(1, 2, 3, 6), n_classes=1)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True) 

        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.nInputChannels = nInputChannels
        if pretrained and len(self.cascade_pretrain_dir):
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        print('Load pretrained model for cascade subnet:', self.cascade_pretrain_dir)
        pretrain_dict = torch.load(self.cascade_pretrain_dir)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if self.nInputChannels != 3 and "conv1.weight" == k:
                model_dict[k] = state_dict[k]
                model_dict[k][:,:3,:,:] = v
            #wether copy other
            elif k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x, gts=None, void_pixels=None, step='step1', select_ratio=[1.0,1.0,1.0,1.0,1.0]):
        x = self.conv1(x)  # /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # /2

        x1 = self.layer1(x)
        x2 = self.layer2(x1)   # /2
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.psp4(x4)
        feats = [x4,x3,x2,x1]
        coarse_fms, coarse_outs = self.global_net(feats)
        fine_out = self.refine_net(coarse_fms)

        coarse_outs[0] = self.upsample(coarse_outs[0])
        coarse_outs[1] = self.upsample(coarse_outs[1])
        coarse_outs[2] = self.upsample(coarse_outs[2])
        coarse_outs[3] = self.upsample(coarse_outs[3])
        fine_out = self.upsample(fine_out)

        if self.training:
            pred_list_step1 = [coarse_outs[0], coarse_outs[1], coarse_outs[2] ,coarse_outs[3], fine_out]
            refine_loss = self.criterion(pred_list_step1, gts, void_pixels=void_pixels, step=step)
            return fine_out, refine_loss
        else:
            return fine_out


def res18_cpn(nInputChannels=6, pretrained=True, cascade_pretrain_dir='', criterion=nn.CrossEntropyLoss(ignore_index=255)):
    """
    refine the cpn-output with res18, use the coarse-to-fine predict (CPN style)
    """
    model = ResNet18_CPN(nInputChannels, BasicBlock, [2, 2, 2, 2], pretrained, cascade_pretrain_dir, criterion=criterion)
    return model

