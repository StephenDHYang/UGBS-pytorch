#coding=utf-8
import numpy as np
import os.path
import logging
import argparse
import cv2
import torch.nn.parallel
import numpy as np
from PIL import Image
import util.helpers as helpers
from util import dataset
from util.util import AverageMeter, get_cuda_devices
from torch.nn.functional import upsample
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
from torchvision import transforms
from tensorboardX import SummaryWriter
from util.custom_transforms import mixScribbleTransform
from util import dataset, transform, config, helpers
from util.compute_boundary_acc import compute_boundary_acc


def sort_dict(dict_src):
    dict_new = {}
    for k in sorted(dict_src):
        dict_new.update({k: dict_src[k]})
    return dict_new


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


def jaccard(annotation, segmentation, void_pixels=None):
    # return iou of annotation and segmentation

    assert(annotation.shape == segmentation.shape)

    if void_pixels is None:
        void_pixels = np.zeros_like(annotation)
    assert(void_pixels.shape == annotation.shape)

    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)
    void_pixels = void_pixels.astype(np.bool)
    if np.isclose(np.sum(annotation & np.logical_not(void_pixels)), 0) and np.isclose(np.sum(segmentation & np.logical_not(void_pixels)), 0):
        return 1, 1, 1
    else:
        intersection = np.sum(((annotation & segmentation) & np.logical_not(void_pixels)))
        union = np.sum(((annotation | segmentation) & np.logical_not(void_pixels)), dtype=np.float32)
        return intersection / union, intersection, union


def compute_iou(gt_mask, predict_mask, void_pixels=None):
    # iou computation by CVNet: https://github.com/xzq-njust/CVNet/blob/13df2d307b5164027cbf925a50809564dce6ad7b/package/utils/data_utils.py#L35
    intersection = np.count_nonzero(
        np.logical_and(predict_mask, gt_mask)
    )
    union = np.count_nonzero(
        np.logical_or(predict_mask, gt_mask)
    )
    return intersection / union, intersection, union


def mAPr(per_cat, thresholds):# per_cat为dict,key为category,value为该category对应的所有mask的jaccard
    n_cat = len(per_cat)
    all_apr = np.zeros(len(thresholds))
    for ii, th in enumerate(thresholds):
        per_cat_recall = np.zeros(n_cat)
        for jj, categ in enumerate(per_cat.keys()):
            per_cat_recall[jj] = np.sum(np.array(per_cat[categ]) > th)/len(per_cat[categ]) #每个类别中IoU大于阈值的mask占比
        all_apr[ii] = per_cat_recall.mean() # 该阈值下所有类别的mask占比的平均
    return all_apr.mean()


def mmIoU(per_cat):
    # firstly compuse the ave iou for per categ
    # then calculate the average of the average iou of each category 
    n_cat = len(per_cat)
    all_miou = np.zeros(n_cat)
    for ii, categ in enumerate(per_cat.keys()):
        all_miou[ii] = np.sum(np.array(per_cat[categ]))/len(per_cat[categ]) # the ave iou for per categ
    return all_miou

def get_bound(boundary, bound_th=0.008, bound_max=3):
    y1,y0,x1,x0 = boundary
    mask_w = x1-x0+1
    mask_h = y1-y0+1
    bound_res = bound_th if bound_th >= 1 else \
                min(np.ceil(bound_th*np.linalg.norm((mask_h, mask_w))), bound_max)
    return bound_res


def db_eval_boundary(foreground_mask, gt_mask, ignore_mask=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))
    if ignore_mask:
        foreground_mask[ignore_mask] = 0
        gt_mask[ignore_mask] = 0

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)
    from skimage.morphology import binary_dilation,disk

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))
    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall)

    return F, precision, recall, np.sum(fg_match), n_fg, np.sum(gt_match), n_gt


def boundary_iou(foreground_mask, gt_mask, ignore_mask=None, bound_th=0.008):
    """
    Computeboundary iou from per-frame evaluation.
    Calculates ioul for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        IoU (float): boundaries IoU
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))
    if ignore_mask:
        foreground_mask[ignore_mask] = 0
        gt_mask[ignore_mask] = 0

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)
    from skimage.morphology import binary_dilation,disk

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix)) # Pd
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix)) # Gd

    # Get the intersection
    gt_i = (gt_dil * gt_mask).astype(np.bool)
    fg_i = (fg_dil * foreground_mask).astype(np.bool)

    # Computer Boundary IoU
    intersection = np.sum(gt_i & fg_i)
    union = np.sum(gt_i|fg_i, dtype=np.float32)

    return intersection/union


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
        David Martin <dmartin@eecs.berkeley.edu>
        January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1] = seg[:,1:]
    s[:-1,:] = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap


def get_relax_pad(relax_pad, extreme_points):
    if relax_pad <= 0:
        return 0
    if relax_pad >= 1:
        return int(relax_pad)

    x_min, y_min = np.min(extreme_points, axis=0)
    x_max, y_max = np.max(extreme_points, axis=0)
    x_len = x_max - x_min + 1
    y_len = y_max - y_min + 1
    return max(20, int(relax_pad * max(x_len, y_len)))

def main():
    global args, logger, writer
    use_void_pixels=True
    logger = get_logger()
    args = get_parser()
    # writer = SummaryWriter(args.save_folder)
    if args.test_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    else:
        args.test_gpu = get_cuda_devices()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(args)# 在屏幕上打印信息
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    # transform and dataloader
    trimap_by_percent = True if args.data_type == 'voc' or args.data_type == 'mig_instance' else False
    _mixtransform = mixScribbleTransform(channel = args.in_channels, relax_crop = args.relax_crop,
                                         use_scribble = args.use_scribble, del_elem = True, 
                                         use_roimasking = args.use_roimasking, use_trimap = args.use_trimap, use_iogpoints = args.use_iogpoints,
                                         no_resize = args.no_resize, trimap_by_percent = trimap_by_percent, use_iogdextr=args.use_iogdextr, use_centroid_map=args.use_centroid_map)
    composed_transforms_ts = _mixtransform.getTestTransform()
    if args.data_type == 'vegas':
        val_data = dataset.VegasSegmentation(root=args.data_root, split='test', transform=composed_transforms_ts)
    elif args.data_type == 'inria':
        val_data = dataset.InriaSegmentation(root=args.data_root, split='test', transform=composed_transforms_ts)
    else:
        raise RuntimeError('Wrong data_type.')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_eval, shuffle=False,
                                             num_workers=args.eval_workers, pin_memory=True, sampler=None)

    # model
    if args.arch == 'banet_bbox':
        print('step1_net:{}, step1_loss_func:{}'.format(args.step1_net, args.step1_loss_func))
        print('-'*50)
        from model.banet import BANet_BBOX
        model = BANet_BBOX(step1_net=args.step1_net, step1_loss_func=args.step1_loss_func, layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True)
    elif args.arch == 'banet_iog':
        print('step1_net:{}, step1_loss_func:{}'.format(args.step1_net, args.step1_loss_func))
        print('-'*50)
        from model.banet import BANet_IOG
        model = BANet_IOG(step1_net=args.step1_net, step1_loss_func=args.step1_loss_func, layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True)
    elif args.arch == 'banet_dextr':
        print('step1_net:{}, step1_loss_func:{}'.format(args.step1_net, args.step1_loss_func))
        print('-'*50)
        from model.banet import BANet_DEXTR
        model = BANet_DEXTR(step1_net=args.step1_net, step1_loss_func=args.step1_loss_func, layers=args.layers, nInputChannels=args.in_channels, n_classes=args.classes, os=16, _print=True)
    else:
        raise RuntimeError('Wrong arch.')

    logger.info(model)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model = model.to(device)
    model.eval()

    # checkpoint
    model_path = args.model_path
    if os.path.isfile(model_path):
        logger.info("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
    
    # evaluate
    print('evaluating Network')
    eval_result = dict()
    eval_result["all_jaccards"] = AverageMeter()
    eval_result['all_precision'] = AverageMeter()
    eval_result['all_recall'] = AverageMeter()
    eval_result['all_fscore'] = AverageMeter()
    eval_result['all_boundary_iou'] = AverageMeter()
    eval_result['all_boundary_acc'] = AverageMeter()
    eval_result['all_area'] = AverageMeter()
    eval_result['all_wcov'] = AverageMeter()
    eval_result["meta"] = []
    eval_result["per_categ_jaccard"] = dict()
    eval_result["per_categ_fscore"] = dict()
    eval_result["per_categ_precision"] = dict()
    eval_result["per_categ_recall"] = dict()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    with torch.no_grad():
        # Main Testing Loop
        num_category_id_dict = {}
        for ii, sample_batched in enumerate(val_loader):
            if ii % 10 == 0:
                print('Evaluating: {} of {} batches'.format(ii, len(val_loader)))
            # predict result and gt
            inputs = sample_batched['concat'].type(torch.FloatTensor)
            crop_gts = sample_batched['crop_gt']
            gts = sample_batched['gt']
            metas = sample_batched["meta"]
            preds = model.forward(inputs)
            preds = preds.to(torch.device('cpu'))
            void_pixels = sample_batched["crop_void_pixels"]
            if args.data_type != 'mig_stuff_artifical':
                metas["category"] = metas["category"].cpu().numpy()
            for jj in range(preds.size()[0]):
                crop_gt = helpers.tens2image(crop_gts[jj])
                gt = helpers.tens2image(gts[jj])
                pred = helpers.tens2image(preds[jj])
                input = helpers.tens2image(inputs[jj])
                pred = 1 / (1 + np.exp(-pred))                
                # Restore the image to its original size
                h = metas['im_size'][0][jj].item()
                w =  metas['im_size'][1][jj].item()
                boundary = [metas['boundary'][0][jj].item(), metas['boundary'][1][jj].item(), 
                            metas['boundary'][2][jj].item(), metas['boundary'][3][jj].item()]                
                points= np.array([[boundary[2], boundary[0]], 
                                  [boundary[2], boundary[1]], 
                                  [boundary[3], boundary[0]], 
                                  [boundary[3], boundary[1]]])
                relax_pad = get_relax_pad(args.relax_crop, points)
                if args.no_resize:
                    pred = helpers.align2fullmask(pred, (h,w), points, relax=relax_pad)
                    #gt = helpers.align2fullmask(gt, (h,w), points, relax=relax_pad)
                else:
                    bbox = helpers.get_bbox(mask=None, points=points, pad=relax_pad, 
                                            zero_pad=args.zero_pad_crop)
                    pred = helpers.crop2fullmask(pred, bbox, im_size=(h,w), zero_pad=args.zero_pad_crop, 
                                                 relax=relax_pad, mask_relax=False)
                    #gt = helpers.crop2fullmask(gt, bbox, im_size=(h,w), zero_pad=args.zero_pad_crop, 
                    #                           relax=relax_pad)
                if use_void_pixels:
                    void_pixel = np.squeeze(helpers.tens2image(sample_batched['void_pixels']))
                    void_pixel = (void_pixel > 0.5)
                if pred.shape != gt.shape:
                    print("pred.shape != gt.shape")
                    pred = cv2.resize(pred, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)
                if void_pixel.shape != gt.shape:
                    print("void_pixel.shape != gt.shape")
                    void_pixel = cv2.resize(void_pixel, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)
                # Threshold
                pred_thres = args.pred_th
                pred = (pred > pred_thres)
                # save per image visualization result
                visualization = False
                if visualization == True:
                    # to do modify
                    test_set_name = 'business'
                    method = 'iogdextr'
                    # #####################
                    visual_save_path = 'visualization_files/result_' + test_set_name
                    if not os.path.exists(visual_save_path):
                        os.makedirs(visual_save_path)
                    pic_id = sample_batched["meta"]["image"][0]
                    if test_set_name == 'business':
                        sub_dir = str(pic_id).split('_')[0]

                    gt_dir = os.path.join(visual_save_path, pic_id + '_' + '001' + '_gt.png')
                    pred_dir = os.path.join(visual_save_path, pic_id + '_' + '002' + '_pred_{}.png'.format(method))
                    pred_image_dir = os.path.join(visual_save_path, pic_id + '_' + '003' +'_{}.png'.format(method))
                    diff_dir = os.path.join(visual_save_path, pic_id + '_' + '004' + '_diff_{}.png'.format(method))
                    pred = pred.astype(np.uint8) * 255
                    pred_mask = pred
                    cv2.imwrite(pred_dir, pred_mask)
                    if test_set_name == 'voc':
                        image_dir = cv2.imread("/mnt/lustre/share_data/sensebee/VOCdevkit/VOC2012/JPEGImages/"+pic_id+".jpg")
                    elif test_set_name == 'mig':
                        image_dir = cv2.imread("/mnt/lustre/share_data/sensebee_panoptic/mig_annotation_data/mig_processed/PNGImages/"+pic_id+".png")
                    elif test_set_name == 'business':
                        image_dir = cv2.imread("/mnt/lustre/share_data/sensebee/sensebee_annotation_data/image/" + sub_dir + "/" + pic_id + ".jpg")
                    mask = cv2.imread(pred_dir)
                    pred_with_image = cv2.addWeighted(image_dir, 0.5, mask, 0.5, 0)

                    gt = gt.astype(np.uint8) * 255
                    cv2.imwrite(gt_dir, gt)
                    cv2.imwrite(pred_image_dir, pred_with_image)
                    cv2.imwrite(diff_dir, (np.abs(pred.astype(np.float32)-gt.astype(np.float32))).astype(np.uint8))

                if args.save_pic:
                    # get object id
                    category_id = str(sample_batched["meta"]["category"][jj])
                    pic_id = sample_batched["meta"]["image"][jj]
                    if pic_id not in num_category_id_dict.keys():
                        num_category_id_dict[pic_id] = {category_id : 1}
                    elif category_id not in num_category_id_dict[pic_id].keys():
                        num_category_id_dict[pic_id][category_id] = 1
                    else:
                        num_category_id_dict[pic_id][category_id] += 1
                    num_sum_obj = num_category_id_dict[pic_id][category_id]

                    gt_dir = os.path.join(args.save_folder,pic_id + '_' + category_id + '_' + str(num_sum_obj) + '_gt.png')
                    void_pixels_dir = os.path.join(args.save_folder,pic_id + '_' + category_id + '_' + str(num_sum_obj) + '_void_pixels.png')
                    pred_dir = os.path.join(args.save_folder,pic_id + '_' + category_id + '_' + str(num_sum_obj) + '_pred.png')
                    dextr_dir = os.path.join(args.save_folder,pic_id + '_' + category_id + '_' + str(num_sum_obj) + '_dextr.png')
                    scribble_dir = os.path.join(args.save_folder,pic_id + '_' + category_id + '_' + str(num_sum_obj) + '_scribble.png')
                    trimap_dir = os.path.join(args.save_folder,pic_id + '_' + category_id + '_' + str(num_sum_obj) + '_trimap.png')
                    cv2.imwrite(pred_dir, pred * 255)
                    cv2.imwrite(gt_dir, gt * 255)
                    if use_void_pixels:
                        cv2.imwrite(void_pixels_dir, void_pixel * 255)
                    # save annotation                    
                    if args.use_scribble and args.in_channels == 5:
                        if args.no_resize:
                            scribble_map = helpers.align2fullmask(input[...,3], (h,w), points, relax=relax_pad)
                            extreme_point_map = helpers.align2fullmask(input[...,4], (h,w), points, relax=relax_pad)
                        else:
                            scribble_map = helpers.crop2fullmask(input[...,3], bbox, im_size=(h,w), zero_pad=args.zero_pad_crop, 
                                                                 relax=relax_pad)
                            extreme_point_map = helpers.crop2fullmask(input[...,4], bbox, im_size=(h,w), zero_pad=args.zero_pad_crop, 
                                                                      relax=relax_pad)
                        cv2.imwrite(scribble_dir, scribble_map)
                        cv2.imwrite(dextr_dir, extreme_point_map)
                    elif args.use_trimap and args.in_channels == 4:
                        if args.no_resize:
                            trimap_map = helpers.align2fullmask(input[...,3], (h,w), points, relax=relax_pad)
                        else:
                            trimap_map = helpers.crop2fullmask(input[...,3], bbox, im_size=(h,w), zero_pad=args.zero_pad_crop, 
                                                               relax=relax_pad)
                        cv2.imwrite(trimap_dir, trimap_map)
                    elif args.use_scribble == False and args.use_trimap == False and args.in_channels == 4:
                        if args.no_resize:
                            extreme_point_map = helpers.align2fullmask(input[...,3], (h,w), points, relax=relax_pad)
                        else:
                            extreme_point_map = helpers.crop2fullmask(input[...,3], bbox, im_size=(h,w), zero_pad=args.zero_pad_crop, 
                                                                      relax=relax_pad)
                        cv2.imwrite(dextr_dir, extreme_point_map)

                # Store in per category
                if "category" in sample_batched["meta"]:
                    cat = sample_batched["meta"]["category"][jj]
                else:
                    cat = 1
                if cat not in eval_result["per_categ_jaccard"]:
                    eval_result["per_categ_jaccard"][cat] = [0, 0]
                    eval_result["per_categ_recall"][cat] = [0, 0]
                    eval_result["per_categ_precision"][cat] = [0, 0]

                bound_th = get_bound(boundary, args.bound_th,  3)

                f_bound_n_fg = [0] * 5
                f_bound_fg_match = [0] * 5
                f_bound_gt_match= [0] * 5
                f_bound_n_gt = [0] * 5
                # Evaluate
                if use_void_pixels:
                    j_s, j_nu, j_de = compute_iou(gt, pred, void_pixel)
                    # f_s, p_s, r_s, p_nu, p_de, r_nu, r_de = db_eval_boundary(pred.copy(), 
                    #                gt, np.where(void_pixel > 0.5), bound_th)

                    for bounds in range(5):
                        _, _, _, fg_match, n_fg, gt_match, n_gt = db_eval_boundary(pred.copy(), 
                                                gt, np.where(void_pixel > 0.5), bound_th=bounds + 1)
                        f_bound_fg_match[bounds] += fg_match
                        f_bound_n_fg[bounds] += n_fg
                        f_bound_gt_match[bounds] += gt_match
                        f_bound_n_gt[bounds] += n_gt
                
                    b_iou = boundary_iou(pred.copy(), 
                                    gt, np.where(void_pixel > 0.5), bound_th)
                    b_a = compute_boundary_acc(gt, pred, np.where(void_pixel > 0.5))
                else:
                    j_s, j_nu, j_de = compute_iou(gt, pred)
                    # f_s, p_s, r_s, p_nu, p_de, r_nu, r_de = db_eval_boundary(pred, gt, bound_th)
                    for bounds in range(5):
                        _, _, _, fg_match, n_fg, gt_match, n_gt = db_eval_boundary(pred, gt, bound_th=bounds + 1)
                        f_bound_fg_match[bounds] += fg_match
                        f_bound_n_fg[bounds] += n_fg
                        f_bound_gt_match[bounds] += gt_match
                        f_bound_n_gt[bounds] += n_gt

                    b_iou = boundary_iou(pred, gt, bound_th)
                    b_a = compute_boundary_acc(gt, pred)

                gt_area = np.count_nonzero(gt)
                wcov = gt_area * j_s
                eval_result["all_area"].update(gt_area)
                eval_result["all_wcov"].update(wcov)

                eval_result["all_jaccards"].update(j_s)
                # eval_result["all_fscore"].update(f_s)
                # eval_result["all_precision"].update(p_s)
                # eval_result["all_recall"].update(r_s)
                eval_result['all_boundary_iou'].update(b_iou)
                eval_result["all_boundary_acc"].update(b_a)
                eval_result["per_categ_jaccard"][cat][0] += j_nu
                eval_result["per_categ_jaccard"][cat][1] += j_de
                # eval_result["per_categ_recall"][cat][0] += r_nu
                # eval_result["per_categ_recall"][cat][1] += r_de
                # eval_result["per_categ_precision"][cat][0] += p_nu
                # eval_result["per_categ_precision"][cat][1] += p_de
            # Compute some stats
        for cat in eval_result["per_categ_jaccard"]:
            eval_result["per_categ_jaccard"][cat] = round(float(eval_result["per_categ_jaccard"][cat][0]) / \
                max(1e-8, float(eval_result["per_categ_jaccard"][cat][1])), 4)
            # eval_result["per_categ_recall"][cat] = round(float(eval_result["per_categ_recall"][cat][0]) / \
            #    max(1e-8, float(eval_result["per_categ_recall"][cat][1])), 4)
            # eval_result["per_categ_precision"][cat] = round(float(eval_result["per_categ_precision"][cat][0]) / \
            #    max(1e-8, float(eval_result["per_categ_precision"][cat][1])), 4)
            # eval_result["per_categ_fscore"][cat] = round(2*eval_result["per_categ_precision"][cat] * \
            #        eval_result["per_categ_recall"][cat]/ \
            #        max(1e-8, (eval_result["per_categ_precision"][cat]+eval_result["per_categ_recall"][cat])), 4)
    f_bound = [None] * 5
    for bounds in range(5):
        precision = f_bound_fg_match[bounds] / f_bound_n_fg[bounds]
        recall = f_bound_gt_match[bounds] / f_bound_n_gt[bounds]
        f_bound[bounds] = 2 * precision * recall / (precision + recall)

    logger.info("=========result==========")
    logger.info("IoU: ")
    logger.info(eval_result["all_jaccards"].avg)
    logger.info("WCov: ")
    logger.info(eval_result["all_wcov"].sum / eval_result["all_area"].sum)
    logger.info("boundary F-score: ")
    logger.info(sum(f_bound)/5)
    logger.info("Dice: ")
    logger.info((2*eval_result["all_jaccards"].avg)/(1+eval_result["all_jaccards"].avg))
    logger.info("boundary IoU: ")
    logger.info(eval_result['all_boundary_iou'].avg)   
    logger.info("boundary Accuracy: ")
    logger.info(eval_result['all_boundary_acc'].avg)

if __name__ == '__main__':
    main()
