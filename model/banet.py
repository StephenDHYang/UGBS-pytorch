import math
import numpy as np
import pdb

import torch.nn as nn
import torch
import torch.nn.functional as F

from model.basenet import res18_cpn
from util.sobel_op import SobelComputer
import util.boundary_modification as boundary_modification

sobel_compute = SobelComputer()
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# BottleneckII is used in FineNet
class BottleneckII(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleneckII, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 2,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 2),
            )
 
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


class BANet_BBOX(nn.Module):
    """ Three-Step Approach for Interative Semantic Segmentaion
        Step1: Get Segmentation Confidence Map
	Step2: Generate Semantic Prediction
	Step3: Boundary Refinement
    """
    def __init__(self, step1_net='res18', step1_loss_func='mse', layers=101, nInputChannels=3, n_classes=21, os=16, pretrained=True, pretrain_dir='', 
            cascade_pretrain_dir='', _print=True, criterion=nn.CrossEntropyLoss(ignore_index=255) ):
        if _print:
            print("Constructing BANet(bbox) model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(BANet_BBOX, self).__init__()
        if layers == 101:
            print('Pretrained model: res101')
            self.resnet = ResNet101(5, os, pretrained=pretrained, pretrain_dir=pretrain_dir)
        else:
            print('Pretrained model: res50')
            self.resnet = ResNet50(5, os, pretrained=pretrained, pretrain_dir=pretrain_dir)
        output_shape = 128
        channel_settings = [512, 1024, 512, 256]
        self.step1_loss_func = step1_loss_func
        self.step1_net = step1_net
        self.global_net = CoarseNet(channel_settings, output_shape, n_classes)
        self.refine_net = FineNet(channel_settings[-1], output_shape, n_classes)
        self.psp4 = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=256)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        self.up128 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        self.criterion = criterion
        self.n_classes = n_classes
        if step1_net == 'res18':
            self.confidence_prediction = res18_cpn(nInputChannels=4, pretrained=pretrained, cascade_pretrain_dir=cascade_pretrain_dir, criterion=self.criterion)
        else:
            raise NotImplementedError("confidence prediction can only be res18")
        self.boundary_refine = res18_cpn(nInputChannels=5, pretrained=pretrained, cascade_pretrain_dir=cascade_pretrain_dir, criterion=self.criterion)
        # seg perturbation
        iou_max = 1.0
        iou_min = 0.8
        self.iou_target = np.random.rand()*(iou_max-iou_min) + iou_min

    def forward(self, x, gts=None, noise_gts=None, void_pixels=None, select_ratio=[1.0,1.0,1.0,1.0,1.0]):
	# x:[0-2:rgb,3:in_point,4:out_point,5:fore_centroid,6:back_centroid]
        input_step1 = torch.cat((x[:,:3,:,:], x[:,4:5,:,:]), 1)
        centroid_map = x[:,5:6,:,:]/255.0
        
        # Step1: Get confidence map
        if self.step1_net == 'res18':
            if self.training:
                pred_centroid, loss_step1 = self.confidence_prediction(input_step1, centroid_map)
            else:
                pred_centroid = self.confidence_prediction(input_step1)
        else:
            raise NotImplementedError("confidence prediction can only be res18")
        
        # Step2: Get coarse segmentation prediction
        input_step2 = torch.cat((x[:,:3,:,:], (pred_centroid*255).detach(), x[:,4:5,:,:]), 1)
        [low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1] = self.resnet(input_step2)
        low_level_feat_4 = self.psp4(low_level_feat_4)
        res_out = [low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1]
        coarse_fms, coarse_outs = self.global_net(res_out)
        fine_out = self.refine_net(coarse_fms)

        coarse_outs[0] = self.upsample(coarse_outs[0])
        coarse_outs[1] = self.upsample(coarse_outs[1])
        coarse_outs[2] = self.upsample(coarse_outs[2])
        coarse_outs[3] = self.upsample(coarse_outs[3])

        if self.training:
            # Step3: Boundary Refinement
            output = torch.sigmoid(fine_out).detach().cpu().numpy()
            b = output.shape[0]
            seg = []
            for ii in range(0, b):
                # output shape [b, 1, h, w]
                output_sample = np.squeeze(output[ii]) # h, w
                seg_sample = torch.Tensor(boundary_modification.modify_boundary((output_sample>0.5).astype('uint8')*255, iou_target=self.iou_target))
                seg_sample = seg_sample.view(-1, seg_sample.shape[0], seg_sample.shape[1])
                seg.append(seg_sample.unsqueeze(0))
            seg = torch.cat(seg, dim = 0).cuda()
            fine_out_pert = self.upsample(seg)
            input_step3 = torch.cat((x[:,:3,:,:], x[:,4:5,:,:], fine_out_pert), 1)
            input_step3 = input_step3.detach()
            fine_out = self.upsample(fine_out)

            out_step3, loss_step3 = self.boundary_refine(input_step3, gts, void_pixels)
            pred_list_step2 = [coarse_outs[0], coarse_outs[1], coarse_outs[2] ,coarse_outs[3], fine_out]
            loss_step2 = self.criterion(pred_list_step2, gts, void_pixels=void_pixels, step='step2')
            #print('centroid_pred.sum:{:.2f}, centroid_gt.sum:{:.2f}, loss_step0:{:.3f}, loss_step1:{:.6f}, loss_step1:{:.3f}'.format(pred_centroid.sum(), centroid_map.sum(), loss_step0, loss_step1, loss_step1))
            main_loss = loss_step1 +  loss_step2 + loss_step3
            aux_loss = torch.tensor(0.0).cuda()
            return out_step3, main_loss, aux_loss
        else:
            fine_out = torch.sigmoid(fine_out)*255
            fine_out = self.upsample(fine_out)
            input_step3 = torch.cat((x[:,:3,:,:], x[:,4:5,:,:], fine_out), 1)
            out_step3 = self.boundary_refine(input_step3, gts, void_pixels, select_ratio)
            return out_step3


class BANet_IOG(nn.Module):
    """ Three-Step Approach for Interative Semantic Segmentaion
        Step1: Get Segmentation Confidence Map
	Step2: Generate Semantic Prediction
	Step3: Boundary Refinement
    """
    def __init__(self, step1_net='res18', step1_loss_func='mse', layers=101, nInputChannels=3, n_classes=21, os=16, pretrained=True, pretrain_dir='', 
            cascade_pretrain_dir='', _print=True, criterion=nn.CrossEntropyLoss(ignore_index=255) ):
        if _print:
            print("Constructing BANet_IOG model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(BANet_IOG, self).__init__()
        if layers == 101:
            print('Pretrained model: res101')
            self.resnet = ResNet101(6, os, pretrained=pretrained, pretrain_dir=pretrain_dir)
        else:
            print('Pretrained model: res50')
            self.resnet = ResNet50(6, os, pretrained=pretrained, pretrain_dir=pretrain_dir)
        output_shape = 128
        channel_settings = [512, 1024, 512, 256]
        self.step1_loss_func = step1_loss_func
        self.step1_net = step1_net
        self.global_net = CoarseNet(channel_settings, output_shape, n_classes)
        self.refine_net = FineNet(channel_settings[-1], output_shape, n_classes)
        self.psp4 = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=256)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        self.up128 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        self.criterion = criterion
        self.n_classes = n_classes
        if step1_net == 'res18':
            self.confidence_prediction = res18_cpn(nInputChannels=5, pretrained=pretrained, cascade_pretrain_dir=cascade_pretrain_dir, criterion=self.criterion)
        else:
            raise NotImplementedError("confidence prediction can only be res18")
        self.boundary_refine = res18_cpn(nInputChannels=6, pretrained=pretrained, cascade_pretrain_dir=cascade_pretrain_dir, criterion=self.criterion)
        # seg perturbation
        iou_max = 1.0
        iou_min = 0.8
        self.iou_target = np.random.rand()*(iou_max-iou_min) + iou_min

    def forward(self, x, gts=None, noise_gts=None, void_pixels=None, select_ratio=[1.0,1.0,1.0,1.0,1.0]):
	# x:[0-2:rgb,3:in_point,4:out_point,5:fore_centroid,6:back_centroid]
        #input_step1 = torch.cat((x[:,:3,:,:], x[:,4:5,:,:]), 1)
        input_step1 = x[:,:5,:,:]
        centroid_map = x[:,5:6,:,:]/255.0
        
        # Step1: Generate centroid_map
        #pdb.set_trace()
        if self.step1_net == 'res18':
            if self.training:
                pred_centroid, loss_step1 = self.confidence_prediction(input_step1, gts, void_pixels=void_pixels, step='step_seg')
            else:
                pred_centroid = self.confidence_prediction(input_step1)
        else:
            raise NotImplementedError("confidence prediction can only be res18")
        
        # Step2: Generate coarse segmentation_pred
        input_step2 = torch.cat((x[:,:5,:,:], (pred_centroid*255).detach()), 1)
        [low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1] = self.resnet(input_step2)
        low_level_feat_4 = self.psp4(low_level_feat_4)
        res_out = [low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1]
        coarse_fms, coarse_outs = self.global_net(res_out)
        fine_out = self.refine_net(coarse_fms)

        coarse_outs[0] = self.upsample(coarse_outs[0])
        coarse_outs[1] = self.upsample(coarse_outs[1])
        coarse_outs[2] = self.upsample(coarse_outs[2])
        coarse_outs[3] = self.upsample(coarse_outs[3])

        if self.training:
            # Step3: Boundary Refinement
            output = torch.sigmoid(fine_out).detach().cpu().numpy()
            b = output.shape[0]
            seg = []
            for ii in range(0, b):
                # output shape [b, 1, h, w]
                output_sample = np.squeeze(output[ii]) # h, w
                seg_sample = torch.Tensor(boundary_modification.modify_boundary((output_sample>0.5).astype('uint8')*255, iou_target=self.iou_target))
                seg_sample = seg_sample.view(-1, seg_sample.shape[0], seg_sample.shape[1])
                seg.append(seg_sample.unsqueeze(0))
            seg = torch.cat(seg, dim = 0).cuda()
            fine_out_pert = self.upsample(seg)
            input_step3 = torch.cat((x[:,:5,:,:], fine_out_pert), 1)
            input_step3 = input_step3.detach()
            fine_out = self.upsample(fine_out)

            out_step3, loss_step3 = self.boundary_refine(input_step3, gts, void_pixels)
            pred_list_step2 = [coarse_outs[0], coarse_outs[1], coarse_outs[2] ,coarse_outs[3], fine_out]
            loss_step2 = self.criterion(pred_list_step2, gts, void_pixels=void_pixels, step='step2')
            #print('centroid_pred.sum:{:.2f}, centroid_gt.sum:{:.2f}, loss_step0:{:.3f}, loss_step1:{:.6f}, loss_step1:{:.3f}'.format(pred_centroid.sum(), centroid_map.sum(), loss_step0, loss_step1, loss_step1))
            main_loss = loss_step1 + loss_step2 + loss_step3
            aux_loss = torch.tensor(0.0).cuda()
            return out_step3, main_loss, aux_loss
        else:
            fine_out = torch.sigmoid(fine_out)*255
            fine_out = self.upsample(fine_out)
            input_step3 = torch.cat((x[:,:5,:,:], fine_out), 1)
            out_step3 = self.boundary_refine(input_step3, gts, void_pixels, select_ratio)
            return out_step3
            # return fine_out


class BANet_DEXTR(nn.Module):
    """ Three-Step Approach for Interative Semantic Segmentaion
        Step1: Get Segmentation Confidence Map
	Step2: Generate Semantic Prediction
	Step3: Boundary Refinement
    """
    def __init__(self, step1_net='res18', step1_loss_func='mse', layers=101, nInputChannels=3, n_classes=21, os=16, pretrained=True, pretrain_dir='', 
            cascade_pretrain_dir='', _print=True, criterion=nn.CrossEntropyLoss(ignore_index=255) ):
        if _print:
            print("Constructing BANet_DEXTR model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(BANet_DEXTR, self).__init__()
        if layers == 101:
            print('Pretrained model: res101')
            self.resnet = ResNet101(5, os, pretrained=pretrained, pretrain_dir=pretrain_dir)
        else:
            print('Pretrained model: res50')
            self.resnet = ResNet50(5, os, pretrained=pretrained, pretrain_dir=pretrain_dir)
        output_shape = 128
        channel_settings = [512, 1024, 512, 256]
        self.step1_loss_func = step1_loss_func
        self.step1_net = step1_net
        self.global_net = CoarseNet(channel_settings, output_shape, n_classes)
        self.refine_net = FineNet(channel_settings[-1], output_shape, n_classes)
        self.psp4 = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=256)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        self.up128 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        self.criterion = criterion
        self.n_classes = n_classes
        if step1_net == 'res18':
            self.confidence_prediction = res18_cpn(nInputChannels=4, pretrained=pretrained, cascade_pretrain_dir=cascade_pretrain_dir, criterion=self.criterion)
        else:
            raise NotImplementedError("confidence prediction can only be res18")
        self.boundary_refine = res18_cpn(nInputChannels=5, pretrained=pretrained, cascade_pretrain_dir=cascade_pretrain_dir, criterion=self.criterion)
        # seg perturbation
        iou_max = 1.0
        iou_min = 0.8
        self.iou_target = np.random.rand()*(iou_max-iou_min) + iou_min

    def forward(self, x, gts=None, noise_gts=None, void_pixels=None, select_ratio=[1.0,1.0,1.0,1.0,1.0]):
	# x:[0-2:rgb,3:in_point,4:out_point,5:fore_centroid,6:back_centroid]
        input_step1 = x 
        
        # Step1: Generate centroid_map
        #pdb.set_trace()
        if self.step1_net == 'res18':
            if self.training:
                pred_centroid, loss_step1 = self.confidence_prediction(input_step1, gts, void_pixels=void_pixels, step='step_seg')
            else:
                pred_centroid = self.confidence_prediction(input_step1)
        else:
            raise NotImplementedError("confidence prediction can only be res18")
        
        # Step2: Generate coarse segmentation_pred
        input_step2 = torch.cat((x, (pred_centroid*255).detach()), 1)
        [low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1] = self.resnet(input_step2)
        low_level_feat_4 = self.psp4(low_level_feat_4)
        res_out = [low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1]
        coarse_fms, coarse_outs = self.global_net(res_out)
        fine_out = self.refine_net(coarse_fms)

        coarse_outs[0] = self.upsample(coarse_outs[0])
        coarse_outs[1] = self.upsample(coarse_outs[1])
        coarse_outs[2] = self.upsample(coarse_outs[2])
        coarse_outs[3] = self.upsample(coarse_outs[3])

        if self.training:
            # Step3: Boundary Refinement
            output = torch.sigmoid(fine_out).detach().cpu().numpy()
            b = output.shape[0]
            seg = []
            for ii in range(0, b):
                # output shape [b, 1, h, w]
                output_sample = np.squeeze(output[ii]) # h, w
                seg_sample = torch.Tensor(boundary_modification.modify_boundary((output_sample>0.5).astype('uint8')*255, iou_target=self.iou_target))
                seg_sample = seg_sample.view(-1, seg_sample.shape[0], seg_sample.shape[1])
                seg.append(seg_sample.unsqueeze(0))
            seg = torch.cat(seg, dim = 0).cuda()
            fine_out_pert = self.upsample(seg)
            input_step3 = torch.cat((x, fine_out_pert), 1)
            input_step3 = input_step3.detach()
            fine_out = self.upsample(fine_out)

            out_step3, loss_step3 = self.boundary_refine(input_step3, gts, void_pixels)
            pred_list_step2 = [coarse_outs[0], coarse_outs[1], coarse_outs[2] ,coarse_outs[3], fine_out]
            loss_step2 = self.criterion(pred_list_step2, gts, void_pixels=void_pixels, step='step2')
            #print('centroid_pred.sum:{:.2f}, centroid_gt.sum:{:.2f}, loss_step0:{:.3f}, loss_step1:{:.6f}, loss_step1:{:.3f}'.format(pred_centroid.sum(), centroid_map.sum(), loss_step0, loss_step1, loss_step1))
            main_loss =  loss_step1 + loss_step2 + loss_step3
            aux_loss = torch.tensor(0.0).cuda()
            return out_step3, main_loss, aux_loss
        else:
            fine_out = torch.sigmoid(fine_out)*255
            fine_out = self.upsample(fine_out)
            input_step3 = torch.cat((x, fine_out), 1)
            out_step3 = self.boundary_refine(input_step3, gts, void_pixels, select_ratio)
            return out_step3


class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False, pretrain_dir=''):
        self.inplanes = 64
        self.pretrain_dir = pretrain_dir
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()
        self.nInputChannels = nInputChannels

        if pretrained and len(self.pretrain_dir):
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x4, x3, x2, x1]

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        #pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        pretrain_dict = torch.load(self.pretrain_dir)
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

def ResNet18(nInputChannels=3, os=16, pretrained=False, pretrain_dir=''):
    model = ResNet(nInputChannels, Bottleneck, [2, 2, 2, 2], os, pretrained=pretrained, pretrain_dir=pretrain_dir)
    return model

def ResNet50(nInputChannels=3, os=16, pretrained=False, pretrain_dir=''):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os, pretrained=pretrained, pretrain_dir=pretrain_dir)
    return model

def ResNet101(nInputChannels=3, os=16, pretrained=False, pretrain_dir=''):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained, pretrain_dir=pretrain_dir)
    return model


class FineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class):
        super(FineNet, self).__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade-i-1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self._predict(4*lateral_channel, num_class)    

    def _make_layer(self, input_channel, num, output_shape):
        layers = []
        for i in range(num):
            layers.append(BottleneckII(input_channel, 128))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class):
        layers = []
        layers.append(BottleneckII(input_channel, 128))
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

class CoarseNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(CoarseNet, self).__init__()
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
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, num_class,
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
                if i != len(self.channel_settings) - 1:
                    up = self.upsamples[i](feature)
                feature = self.predict[i](feature)
                coarse_outs.append(feature)        
        return coarse_fms, coarse_outs
