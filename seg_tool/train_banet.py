import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
try:
    import apex
except:
    pass
from torchvision import transforms
from tensorboardX import SummaryWriter
from util.custom_transforms import mixScribbleTransform
from util import dataset, transform, config, helpers
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_cuda_devices, class_balanced_cross_entropy_loss, class_cross_entropy_loss
from util.util import LovaszSoftmax_LOSS, SSIM_LOSS, IOU_LOSS, IOU_LOSS_VOID, CE_LOSS
from util.util import exponential_triangle_lr

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
mp.set_sharing_strategy('file_system')

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_select_ratio(focus_start, focus_end, cur_iter, max_iter):
    focus_w =  [(loss_i+(cur_iter)/(max_iter)*(loss_j-loss_i)) for \
		    loss_i, loss_j in zip(focus_start, focus_end)]
    return focus_w 

def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    if args.train_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    else:
        args.train_gpu = get_cuda_devices()
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def mixed_loss(pred_list, target, void_pixels=None, show_loss=True, step='step0'):
    loss_dict = {'CE':[]}
    if args.use_iou_loss:
        loss_dict['IOU_VOID'] = []
    if args.use_ssim_loss:
        loss_dict['SSIM'] = []
    if args.use_lovasz_loss:
        loss_dict['LovaszSoftmax'] = []
    
    # side loss
    for pred in pred_list:
        loss_dict['CE'].append(CE_LOSS()(pred, target, void_pixels=void_pixels))
        if args.use_iou_loss:
            loss_dict['IOU_VOID'].append(0.5*IOU_LOSS_VOID()(torch.sigmoid(pred), target, void_pixels=void_pixels))
        if args.use_ssim_loss:
            loss_dict['SSIM'].append(1 - SSIM_LOSS(window_size=11)(torch.sigmoid(pred), target))
        if args.use_lovasz_loss:
            loss_dict['LovaszSoftmax'].append(LovaszSoftmax_LOSS()(torch.sigmoid(pred), target))
    
    if show_loss:
        len_pred = len(pred_list)
        for key in loss_dict:
            line = '{0}_loss[{1:<12}]=['.format(step, key)
            for i in range(len_pred-1):
                line += '{:.4f}, '.format(loss_dict[key][i])
            line += '{:.4f}] sum = {:.4f}'.format(loss_dict[key][-1], sum(loss_dict[key]))
            #print(line)
    loss = sum(loss_dict['CE'])
    for key in loss_dict:
        if key != 'CE':
            loss += sum(loss_dict[key])
    return loss

def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.use_apex and args.sync_bn and args.multiprocessing_distributed:
        BatchNorm = apex.parallel.SyncBatchNorm
    else:
        BatchNorm = nn.BatchNorm2d
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    criterion = mixed_loss

    if args.arch == 'psp':
        from model.pspnet import PSPNet
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, 
                       BatchNorm=BatchNorm, deep_base=args.deep_base, in_channels=args.in_channels, pretrained=args.pretrained, 
                       deeplab_pretrained_model=args.deeplab_pretrained_model, BN_requires_grad = args.BN_requires_grad)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.ppm, model.cls, model.aux]
    elif args.arch == 'psa':
        from model.psanet import PSANet
        model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, psa_type=args.psa_type,
                       compact=args.compact, shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                       normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax,
                       criterion=criterion,
                       BatchNorm=BatchNorm)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.psa, model.cls, model.aux]
    elif args.arch == 'deeplabv3p':
        from model.deeplabv3p_resnet import DeepLabv3_plus
        model = DeepLabv3_plus(nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True,
                                pretrained=args.pretrained, pretrain_dir=args.deeplab_pretrained_model, criterion=criterion)
        modules_ori = [model.resnet_features]
        modules_new = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    elif args.arch == 'cpn':
        from model.cpn import CPN
        model = CPN(nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True,
                                pretrained=args.pretrained, pretrain_dir=args.deeplab_pretrained_model, criterion=criterion)
        modules_ori = [model.resnet]
        modules_new = [model.global_net, model.refine_net]
    elif args.arch == 'iog':   # IOG method
        from model.banet import IOG
        model = IOG(layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True,
                                pretrained=args.pretrained, pretrain_dir=args.deeplab_pretrained_model, criterion=criterion)
        modules_ori = [model.resnet]
        modules_new = [model.global_net, model.refine_net]
    elif args.arch == 'iog_pert_gt':   # IOG + gt_pert + res18
        from model.banet import IOG_pert_gt
        model = IOG_pert_gt(layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True,
                                pretrained=args.pretrained, pretrain_dir=args.deeplab_pretrained_model,
                                cascade_pretrain_dir=args.cascade_pretrained_model, criterion=criterion)
        modules_ori = [model.resnet, model.boundary_refine]
        modules_new = [model.global_net, model.refine_net]
    elif args.arch == 'iog_pert_pred':   # IOG + pred_pert + res18
        from model.banet import IOG_pert_pred
        model = IOG_pert_pred(layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True,
                                pretrained=args.pretrained, pretrain_dir=args.deeplab_pretrained_model,
                                cascade_pretrain_dir=args.cascade_pretrained_model, criterion=criterion)
        modules_ori = [model.resnet, model.boundary_refine]
        modules_new = [model.global_net, model.refine_net]
    elif args.arch == 'brnet_iog':   # CPN+cascade refine baseline
        from model.banet import BRNet_IOG
        model = BRNet_IOG(layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True,
                                pretrained=args.pretrained, pretrain_dir=args.deeplab_pretrained_model,
                                cascade_pretrain_dir=args.cascade_pretrained_model, criterion=criterion)
        modules_ori = [model.resnet, model.boundary_refine]
        modules_new = [model.global_net, model.refine_net]
    elif args.arch == 'banet_bbox':   # CPN+cascade refine baseline
        from model.banet import BANet_BBOX
        print('step1_net:{}, step1_loss_func:{}'.format(args.step1_net, args.step1_loss_func))
        model = BANet_BBOX(step1_net=args.step1_net, step1_loss_func=args.step1_loss_func, layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True,
                                pretrained=args.pretrained, pretrain_dir=args.deeplab_pretrained_model,
                                cascade_pretrain_dir=args.cascade_pretrained_model, criterion=criterion)
        modules_ori = [model.resnet, model.boundary_refine]
        modules_new = [model.global_net, model.refine_net, model.confidence_prediction]
    elif args.arch == 'banet_iog':   # CPN+cascade refine baseline
        from model.banet import BANet_IOG
        print('step1_net:{}, step1_loss_func:{}'.format(args.step1_net, args.step1_loss_func))
        model = BANet_IOG(step1_net=args.step1_net, step1_loss_func=args.step1_loss_func, layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True,
                                pretrained=args.pretrained, pretrain_dir=args.deeplab_pretrained_model,
                                cascade_pretrain_dir=args.cascade_pretrained_model, criterion=criterion)
        modules_ori = [model.resnet, model.boundary_refine]
        modules_new = [model.global_net, model.refine_net, model.confidence_prediction]
    elif args.arch == 'banet_dextr':   # CPN+cascade refine baseline
        from model.banet import BANet_DEXTR
        print('step1_net:{}, step1_loss_func:{}'.format(args.step1_net, args.step1_loss_func))
        model = BANet_DEXTR(step1_net=args.step1_net, step1_loss_func=args.step1_loss_func, layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True,
                                pretrained=args.pretrained, pretrain_dir=args.deeplab_pretrained_model,
                                cascade_pretrain_dir=args.cascade_pretrained_model, criterion=criterion)
        modules_ori = [model.resnet, model.boundary_refine]
        modules_new = [model.global_net, model.refine_net, model.confidence_prediction]
    else:
        raise RuntimeError('Wrong arch.')
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if not args.use_apex and args.sync_bn and args.multiprocessing_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        # logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        if args.use_apex:
            model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level=args.opt_level, keep_batchnorm_fp32=args.keep_batchnorm_fp32, loss_scale=args.loss_scale)
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Preparation of the data loaders
    # transform
    trimap_by_percent = True if args.data_type == 'voc' or args.data_type == 'mig_instance' else False
    _mixtransform = mixScribbleTransform(channel=args.in_channels, relax_crop=args.relax_crop,
                                        use_scribble=args.use_scribble, use_roimasking=args.use_roimasking,
                                        use_trimap=args.use_trimap, use_iogpoints=args.use_iogpoints, no_resize=False, trimap_by_percent = trimap_by_percent,
                                         use_iogdextr=args.use_iogdextr, use_centroid_map=args.use_centroid_map)
    composed_transforms_tr = _mixtransform.getTrainTransform()
    composed_transforms_ts = _mixtransform.getTestTransform()
    # training data
    if args.data_type == 'context':
        train_data = dataset.ContextSegmentation(root = args.data_root, split = 'train',
                    transform = composed_transforms_tr,use_context_bg = args.use_context_bg,
                    use_context_inst = args.use_context_inst,
                    connected_components = args.connected_components, channel = args.in_channels)
    elif args.data_type == 'voc':
        voc_train = dataset.VOCSegmentation(root=args.data_root, split='train', transform=composed_transforms_tr)
        if args.use_sbd:
            voc_val = dataset.VOCSegmentation(root=args.data_root, split='val', transform=composed_transforms_ts)
            sbd = dataset.SBDSegmentation(root=args.sbd_root, split=['train', 'val'], transform=composed_transforms_tr, retname=True)
            train_data = dataset.CombineDBs([voc_train, sbd], excluded=[voc_val])
            #train_data = dataset.CombineDBs([sbd], excluded=[voc_val])
        else:
            train_data = voc_train
    elif args.data_type == 'ade20k':
        train_data = dataset.Ade20kSegmentation(root=args.data_root, split='train', transform=composed_transforms_tr)
    elif args.data_type == 'ade20k_0.05':
        train_data = dataset.Ade20kSegmentation(root=args.data_root, split='train_0.05', transform=composed_transforms_tr)
    elif args.data_type == 'ade20k_0.1':
        train_data = dataset.Ade20kSegmentation(root=args.data_root, split='train_0.1', transform=composed_transforms_tr)
    elif args.data_type == 'ade20k_0.2':
        train_data = dataset.Ade20kSegmentation(root=args.data_root, split='train_0.2', transform=composed_transforms_tr)
    elif args.data_type == 'cityscape_full':
        train_data = dataset.CityScapesSegmentation(root=args.data_root, split='train', label_type='full', transform=composed_transforms_tr)
    elif args.data_type == 'cityscape_select':
        train_data = dataset.CityScapesSegmentation(root=args.data_root, split='train', label_type='select', transform=composed_transforms_tr)
    elif args.data_type == 'vegas':
        train_data = dataset.VegasSegmentation(root=args.data_root, split='train', transform=composed_transforms_tr)
    elif args.data_type == 'inria':
        train_data = dataset.InriaSegmentation(root=args.data_root, split='train', transform=composed_transforms_tr)
    elif args.data_type == 'coco':
        coco_data = dataset.CocoSegmentation(root=args.data_root, split='train', transform=composed_transforms_tr)
        if args.use_sbd and args.use_voc:
            voc_train = dataset.VOCSegmentation(root=args.data_root, split='train', transform=composed_transforms_tr)
            sbd = dataset.SBDSegmentation(root=args.sbd_root, split=['train', 'val'], transform=composed_transforms_tr, retname=True)
            train_data = dataset.CombineDBs([voc_train, sbd, coco_data])
        else:
            train_data = coco_data
    elif args.data_type == 'business':
        sub_files = args.train_sub_file_list
        total_train = []
        for f_name in sub_files:
            sub_train = dataset.BusinessDataSegmentation(root=args.data_root, split='train', transform=composed_transforms_tr, sub_file=f_name)
            total_train.append(sub_train)
        train_data = dataset.CombineDBs(total_train)

    elif args.data_type == 'mig_instance':
        train_data = dataset.MigInstanceSegmentation(root='/mnt/lustre/share_data/hexianhua/mig_process_data', split='train',
                    transform=composed_transforms_tr)
    else:
        raise RuntimeError('Wrong data_type.')
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=args.rank)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    # evaluate data
    if args.evaluate:
        if args.data_type == 'context':
            val_data = dataset.ContextSegmentation(root=args.data_root, split='val', transform=composed_transforms_ts,
                                                   use_context_bg = args.use_context_bg, use_context_inst = args.use_context_inst,
                                                   connected_components = args.connected_components, channel = args.in_channels)
        elif args.data_type == 'voc':
            val_data = dataset.VOCSegmentation(root=args.data_root, split='val', transform=composed_transforms_ts)
        else:
            raise RuntimeError('Wrong data_type.')
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=args.world_size, rank=args.rank)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                            num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    best_loss_val = float('inf')
    best_miou_val = 0
    best_loss_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            if epoch_log / args.save_freq > 100:
                deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                os.remove(deletename)
        if args.evaluate:
            if args.evaluate == 'loss':
                loss_val, _ = validate(val_loader, model, criterion)
                if main_process():
                    writer.add_scalar('loss_val', loss_val, epoch_log)
                    is_best = loss_val < best_loss_val
                    best_loss_val = min(loss_val, best_loss_val)
                    if is_best:
                        best_loss_epoch = epoch_log
                        logger.info('update best val loss in epoch {} : {:.4f}'.format(best_loss_epoch, best_loss_val))
                        filename = args.save_path + '/best_checkpoint.pth'
                        torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            elif args.evaluate == 'miou':
                _, miou_val = validate(val_loader, model, criterion)
                if main_process():
                    writer.add_scalar('miou', miou_val, epoch_log)
                    is_best = miou_val > best_miou_val
                    best_miou_val = max(miou_val, best_miou_val)
                    if is_best:
                        best_loss_epoch = epoch_log                        
                        logger.info('update best val iou in epoch {} : {:.4f}'.format(best_loss_epoch, best_miou_val))
                        filename = args.save_path + '/best_checkpoint.pth'
                        torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, sample_batched in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, targets, noise_targets, void_pixels = sample_batched['concat'], sample_batched['crop_gt'], sample_batched['noise_gt'], sample_batched['crop_void_pixels']

        view_input = False
        if view_input == True:
            ### add show input
            image_show_path = "image_show"
            if not os.path.exists(image_show_path):
                os.makedirs(image_show_path)
            if i == 0:
                for j in range(48):
                    print("***********************************************")
                    image = inputs[j, :3, :, :]
                    image = image.numpy()
                    image = image.astype("uint8")
                    image = np.transpose(image, [1, 2, 0])
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    ch4 = inputs[j, 3, :, :].unsqueeze(dim=-1)
                    ch4 = ch4.numpy().astype("uint8")
                    print("ch4", ch4.shape)
                    ch5 = inputs[j, 4, :, :].unsqueeze(dim=-1)
                    ch5 = ch5.numpy().astype("uint8")
                    ch6 = inputs[j, 5, :, :].unsqueeze(dim=-1)
                    ch6 = ch6.numpy().astype("uint8")
                    print("ch6", ch6.shape)
                    mask = targets[j, ...]
                    mask = mask.numpy() * 255
                    mask = mask.astype("uint8")
                    mask = np.transpose(mask, [1, 2, 0])
                    print("mask shape", mask.shape)
                    cv2.imwrite(image_show_path + "/{}_image.png".format(j), image)
                    cv2.imwrite(image_show_path + "/{}_ch4.png".format(j), ch4)
                    cv2.imwrite(image_show_path + "/{}_ch5.png".format(j), ch5)
                    cv2.imwrite(image_show_path + "/{}_ch6.png".format(j), ch6)
                    cv2.imwrite(image_show_path + "/{}_mask.png".format(j), mask)
                    print("***********************************************")

        inputs.requires_grad_()
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        cur_iter = epoch * len(train_loader) + i + 1
        focus_start = args.focus_start
        focus_end =  args.focus_end
        select_ratio = get_select_ratio(focus_start, focus_end, cur_iter, max_iter)
        
        output, main_loss, aux_loss = model(inputs, targets, noise_targets, void_pixels, select_ratio=select_ratio)
        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + args.aux_weight * aux_loss

        optimizer.zero_grad()
        if args.use_apex and args.multiprocessing_distributed:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        n = inputs.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  # not considering ignore pixels
            count = targets.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n

        main_loss_meter.update(main_loss.item(), n)
        #aux_loss_meter.update(aux_loss.item(), n)
        aux_loss_meter.update(args.aux_weight*aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        if args.lr_policy:
            if args.lr_policy == 'poly':
                current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
                for index in range(0, args.index_split):
                    optimizer.param_groups[index]['lr'] = current_lr
                for index in range(args.index_split, len(optimizer.param_groups)):
                    optimizer.param_groups[index]['lr'] = current_lr * 10
            elif args.lr_policy == 'exponential_triangle':
                current_lr = exponential_triangle_lr(args.base_lr, current_iter, max_iter, restart_period = 4 * len(train_loader))
                for index in range(0, args.index_split):
                    optimizer.param_groups[index]['lr'] = current_lr
                for index in range(args.index_split, len(optimizer.param_groups)):
                    optimizer.param_groups[index]['lr'] = current_lr * 10
            elif args.lr_policy == 'const':
                current_lr = args.base_lr
                for index in range(0, args.index_split):
                    optimizer.param_groups[index]['lr'] = current_lr
                for index in range(args.index_split, len(optimizer.param_groups)):
                    optimizer.param_groups[index]['lr'] = current_lr * 10
            elif args.lr_policy == 'step':
                if epoch < args.epochs*0.25:
                    current_lr = args.base_lr
                elif epoch < args.epochs*0.5:
                    current_lr = args.base_lr*0.3
                elif epoch < args.epochs*0.75:
                    current_lr = args.base_lr*(0.3**2)
                else:
                    current_lr = args.base_lr*(0.3**3)
                for index in range(0, args.index_split):
                    optimizer.param_groups[index]['lr'] = current_lr
                for index in range(args.index_split, len(optimizer.param_groups)):
                    optimizer.param_groups[index]['lr'] = current_lr * 10
            else:
                raise RuntimeError('Wrong lr_policy.')
                
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time,
                                                            remain_time=remain_time,
                                                            main_loss_meter=main_loss_meter,
                                                            aux_loss_meter=aux_loss_meter,
                                                            loss_meter=loss_meter))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
    return main_loss_meter.avg


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, sample_batched in enumerate(val_loader):
        data_time.update(time.time() - end)
        inputs, targets = sample_batched['concat'], sample_batched['crop_gt']
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        outputs = model(inputs)
        # loss
        loss = criterion(outputs, targets)
        n = inputs.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = targets.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)
        loss_meter.update(loss.item(), inputs.size(0))
        # iou
        outputs = 1 / (1 + torch.exp(-outputs))
        pred_thres = args.pred_th
        outputs = (outputs > pred_thres).float()
        intersection = torch.sum(targets*outputs)
        union = torch.sum(targets) + torch.sum(outputs) - intersection
        iou = intersection / union
        n = inputs.size(0)
        if args.multiprocessing_distributed:
            iou = iou * n
            count = targets.new_tensor([n], dtype=torch.long)
            dist.all_reduce(iou), dist.all_reduce(count)
            n = count.item()
            iou = iou / n
        else:
            iou = torch.mean(iou)
        iou_meter.update(iou.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'iou {iou_meter.val:.4f} ({iou_meter.avg:.4f}) '.format(i + 1, len(val_loader),
                                                                                   data_time=data_time,
                                                                                   batch_time=batch_time,
                                                                                   loss_meter=loss_meter,
                                                                                   iou_meter=iou_meter))
    return loss_meter.avg, iou_meter.avg


if __name__ == '__main__':
    main()
