import os
import os.path
import sys
import errno
import hashlib
import tarfile
import json
import cv2
import math
import copy
import scipy.io
import numpy as np
from PIL import Image
from six.moves import urllib
import util.helpers as helpers
from util.util import load_image_file

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms


interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(cv2_interp, random_interp=False):
    if random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split('\t')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class VegasSegmentation(Dataset):

    def __init__(self,
                 root='vegas',
                 split='train',
                 transform=None,
                 default=False):

        self.root = root
        self.split = split
        _mask_dir = os.path.join(self.root, 'vegas_' + self.split + '_label_224_png_01')
        _image_dir = os.path.join(self.root, 'vegas_' + self.split + '_img_224_jpg')
        self.transform = transform
        self.default = default

        self.sample_list = []
        self.images = []
        self.masks = []

        filenames = os.listdir(_image_dir)
        for filename in filenames:
            if '.jpg' in filename:
                self.sample_list.append(filename.split('.')[0])

        # if self.split == 'train':
        #     self.sample_list = self.sample_list[:100]

        for sample in self.sample_list:
            _image = os.path.join(_image_dir, sample + '.jpg')
            _mask = os.path.join(_mask_dir, sample + '.png')
            assert os.path.isfile(_image)
            assert os.path.isfile(_mask)
            self.images.append(_image)
            self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(len(self.sample_list), len(self.sample_list)))

    def __getitem__(self, index):
        _img = load_image_file(self.images[index])
        _target = (np.array(Image.open(self.masks[index]))).astype(np.float32)
        _void_pixels = np.zeros_like(_target).astype(np.float32)

        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels}

        _target_area = np.where(_target>0)
        sample['meta'] = {'image': str(self.sample_list[index]),
                            'object': str(1),
                            'category': 1,
                            'im_size': (_img.shape[0], _img.shape[1]),
                            'boundary': [_target_area[0].max(), _target_area[0].min(),
                                        _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sample_list)

    def __str__(self):
        return 'Vegas remote sensing data'


class InriaSegmentation(Dataset):
    MEAN = np.array([0.42825925, 0.4411106, 0.40593693])
    STD = np.array([0.21789166, 0.20679809, 0.20379359])

    def __init__(self,
                 root='inria',
                 split='train',
                 transform=None,
                 default=False):

        self.root = root
        self.split = split
        _mask_dir = os.path.join(self.root, 'mask_one')
        _image_dir = os.path.join(self.root, 'img')
        self.transform = transform
        self.default = default

        self.sample_list = []
        self.images = []
        self.masks = []

        with open(os.path.join(self.root, 'inria_split', self.split + '.txt'), "r") as f:
            print(os.path.join(self.root, self.split + '.txt'))
            lines = f.read().splitlines()

        for filename in lines:
            if '.tif' in filename:
                self.sample_list.append(filename.split('.')[0])

        # if self.split == 'train':
        #     self.sample_list = self.sample_list[:100]

        for sample in self.sample_list:
            _image = os.path.join(_image_dir, sample + '.tif')
            _mask = os.path.join(_mask_dir, sample + '.tif')
            assert os.path.isfile(_image)
            assert os.path.isfile(_mask)
            self.images.append(_image)
            self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(len(self.sample_list), len(self.sample_list)))

    def __getitem__(self, index):
        _img = np.array(Image.open(self.images[index])).astype(np.float32)
        _target = np.array(Image.open(self.masks[index]).convert('L')).astype(np.float32)
        _void_pixels = np.zeros_like(_target).astype(np.float32)

        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels}

        _target_area = np.where(_target>0)
        sample['meta'] = {'image': str(self.sample_list[index]),
                            'object': str(1),
                            'category': 1,
                            'im_size': (_img.shape[0], _img.shape[1]),
                            'boundary': [_target_area[0].max(), _target_area[0].min(),
                                        _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sample_list)

    def __str__(self):
        return 'Inria remote sensing data'
    
    
class Ade20kSegmentation(Dataset):
    BASE_DIR = 'ade20k_pascal'

    def __init__(self,
                 root='dataset',
                 split='val',
                 transform=None,
                 download=False,
                 preprocess=False,
                 area_thres=0,
                 retname=True,
                 suppress_void_pixels=True,
                 default=False):

        self.root = root
        _ade20k_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_ade20k_root, 'SegmentationObject')
        _cat_dir = os.path.join(_ade20k_root, 'SegmentationClass')
        _image_dir = os.path.join(_ade20k_root, 'Images')
        self.transform = transform
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.area_thres = area_thres
        self.retname = retname
        self.suppress_void_pixels = suppress_void_pixels
        self.default = default

        # Build the ids file
        area_th_str = ""
        if self.area_thres != 0:
            area_th_str = '_area_thres-' + str(area_thres)

        self.obj_list_file = os.path.join(self.root, self.BASE_DIR, 'ImageSets', 'Segmentation',
                                          '_'.join(self.split) + '_instances' + area_th_str + '.txt')

        if download:
            self._download()

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_ade20k_root, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.masks = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(_image_dir, line + ".jpg")
                _cat = os.path.join(_cat_dir, line + ".png")
                _mask = os.path.join(_mask_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_mask)
                self.im_ids.append(line.rstrip('\n'))
                self.images.append(_image)
                self.categories.append(_cat)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))
        assert (len(self.images) == len(self.categories))

        # Precompute the list of objects and their categories for each image 
        if (not self._check_preprocess()) or preprocess:
            print('Preprocessing of ADE20K dataset, this will take long, but it will be done only once.')
            self._preprocess()

        # Build the list of objects
        self.obj_list = []
        num_images = 0
        for ii in range(len(self.im_ids)):
            flag = False
            for jj in range(len(self.obj_dict[self.im_ids[ii]])):
                if self.obj_dict[self.im_ids[ii]][jj] != -1:
                    self.obj_list.append([ii, jj])
                    flag = True
            if flag:
                num_images += 1

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):
        _img, _target, _void_pixels = self._make_img_gt_point_pair(index)
        weight = np.ones_like(_target)
        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels, 'weight': weight}
        # print(_target.min(), _target.max())
        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.obj_list[index][1]
            _target_area = np.where(_target>0)
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'object': str(_obj_ii),
                              'category': self.obj_dict[self.im_ids[_im_ii]][_obj_ii],
                              'im_size': (_img.shape[0], _img.shape[1]),
                              'boundary': [_target_area[0].max(), _target_area[0].min(),
                                           _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.obj_list)

    def _check_preprocess(self):
        _obj_list_file = self.obj_list_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))

            return list(np.sort([str(x) for x in self.obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess(self):
        self.obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            _mask = np.array(Image.open(self.masks[ii]))
            _mask_ids = np.unique(_mask)
            if _mask_ids[-1] == 255:
                n_obj = _mask_ids[-2]
            else:
                n_obj = _mask_ids[-1]

            # Get the categories from these objects
            _cats = np.array(Image.open(self.categories[ii]))
            _cat_ids = []
            for jj in range(n_obj):
                tmp = np.where(_mask == jj + 1)
                obj_area = len(tmp[0])

                if obj_area > self.area_thres:
                    _cat_ids.append(int(_cats[tmp[0][0], tmp[1][0]]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.obj_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

    def _download(self):
        _fpath = os.path.join(self.root, self.FILE)

        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        else:
            print('Downloading ' + self.URL + ' to ' + _fpath)

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (_fpath, float(count * block_size) /
                                  float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, _fpath, _progress)

        # extract file
        cwd = os.getcwd()
        print('Extracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def _make_img_gt_point_pair(self, index):
        _im_ii = self.obj_list[index][0]
        _obj_ii = self.obj_list[index][1]

        # Read Image
        _img = np.array(Image.open(self.images[_im_ii]).convert('RGB')).astype(np.float32)

        # Read Target object
        _tmp = (np.array(Image.open(self.masks[_im_ii]))).astype(np.float32)
        _void_pixels = (_tmp == 255)
        _tmp[_void_pixels] = 0

        _other_same_class = np.zeros(_tmp.shape)
        _other_classes = np.zeros(_tmp.shape)

        if self.default:
            _target = _tmp
        else:
            _target = (_tmp == (_obj_ii + 1)).astype(np.float32)

        return _img, _target, _void_pixels.astype(np.float32)

    def __str__(self):
        return 'ADE20K(split=' + str(self.split) + ',area_thres=' + str(self.area_thres) + ')'
    

class CityScapesSegmentation(Dataset):
    BASE_DIR = 'cityscapes_processed'

    def __init__(self,
                 root='dataset',
                 split='val',
                 label_type:str='select', # the label type of cityscapes gtFine, 'full' means using all 39 classes, 'select' means using selected classes.
                 transform=None,
                 preprocess=False,
                 area_thres=0,
                 retname=True,
                 suppress_void_pixels=True,
                 default=False):
        assert split in ['train', 'val', 'test']
        assert label_type in ['select', 'full']
        self.root = root
        self.split = split
        self.label_type = label_type

        _cityscapes_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_cityscapes_root, split, 'obj_masks')
        _cat_dir = os.path.join(_cityscapes_root, split, 'cat_masks')
        _image_dir = os.path.join(_cityscapes_root, split, 'images')
        self.transform = transform
        self.area_thres = area_thres
        self.retname = retname
        self.suppress_void_pixels = suppress_void_pixels
        self.default = default

        # Build the ids file
        area_th_str = ""
        if self.area_thres != 0:
            area_th_str = '_area_thres-' + str(area_thres)

        self.cat_list_file = os.path.join(_cityscapes_root, self.split, self.split + '_' + self.label_type + '_instances_cat' + area_th_str + '.txt')
        self.ins_list_file = os.path.join(_cityscapes_root, self.split, self.split + '_' + self.label_type + '_instances_ins' + area_th_str + '.txt')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.masks = []

        with open(os.path.join(os.path.join(_cityscapes_root, split, split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(_image_dir, line + "_leftImg8bit.png")
            if self.label_type == 'full':
                _cat = os.path.join(_cat_dir, line + "_gtFine_labelIds.png")
                _mask = os.path.join(_mask_dir, line + "_gtFine_instanceIds.png")
            else:
                _cat = os.path.join(_cat_dir, line + "_gtFine_labelTrainIds.png")
                _mask = os.path.join(_mask_dir, line + "_gtFine_instanceTrainIds.png") 
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            assert os.path.isfile(_mask)
            self.im_ids.append(line.rstrip('\n'))
            self.images.append(_image)
            self.categories.append(_cat)
            self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))
        assert (len(self.images) == len(self.categories))

        # Precompute the list of objects and their categories for each image 
        if (not self._check_preprocess()) or preprocess:
            print('Preprocessing of CityScapes dataset, this will take long, but it will be done only once.')
            self._preprocess()

        # Build the list of objects
        self.obj_list = []
        self.cat_list = []
        # self.im_ids = self.im_ids[:10]
        num_images = 0
        for ii in range(len(self.im_ids)):
            flag = False
            for jj in range(len(self.cat_dict[self.im_ids[ii]])):
                if self.cat_dict[self.im_ids[ii]][jj] != -1:
                    self.obj_list.append([ii, self.ins_dict[self.im_ids[ii]][jj]])
                    self.cat_list.append([ii, jj])
                    flag = True
            if flag:
                num_images += 1

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):
        _img, _target, _void_pixels = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels}

        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.cat_list[index][1]
            _target_area = np.where(_target>0)
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'object': str(_obj_ii),
                              'category': self.cat_dict[self.im_ids[_im_ii]][_obj_ii],
                              'im_size': (_img.shape[0], _img.shape[1]),
                              'boundary': [_target_area[0].max(), _target_area[0].min(),
                                           _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.obj_list)

    def _check_preprocess(self):
        _cat_list_file = self.cat_list_file
        _ins_list_file = self.ins_list_file
        if not os.path.isfile(_cat_list_file) or not os.path.isfile(_ins_list_file):
            return False
        else:
            self.cat_dict = json.load(open(_cat_list_file, 'r'))
            self.ins_dict = json.load(open(_ins_list_file, 'r'))

            return list(np.sort([str(x) for x in self.cat_dict.keys()])) == list(np.sort(self.im_ids)) and list(np.sort([str(x) for x in self.ins_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess(self):
        self.cat_dict = {}
        self.ins_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            _mask = np.array(Image.open(self.masks[ii]))
            _mask_ids = np.unique(_mask)

            # Get the categories from these objects
            _cats = np.array(Image.open(self.categories[ii]))
            _cat_ids = []
            _ins_ids = []
            for obj_id in _mask_ids:
                if obj_id == 255:
                    continue
                if obj_id < 10 and len(_mask_ids) > 2:
                    continue
                tmp = np.where(_mask == obj_id)
                obj_area = len(tmp[0])

                if obj_area > self.area_thres:
                    _cat_ids.append(int(_cats[tmp[0][0], tmp[1][0]]))
                    _ins_ids.append(int(obj_id))
                else:
                    _cat_ids.append(-1)
                    _ins_ids.append(-1)
                obj_counter += 1
            self.ins_dict[self.im_ids[ii]] = _ins_ids
            self.cat_dict[self.im_ids[ii]] = _cat_ids

        with open(self.cat_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.cat_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.cat_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        with open(self.ins_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.ins_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.ins_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

    def _make_img_gt_point_pair(self, index):
        _im_ii = self.obj_list[index][0]
        _obj_ii = self.obj_list[index][1]
        # Read Image
        _img = np.array(Image.open(self.images[_im_ii]).convert('RGB')).astype(np.float32)

        # Read Target object
        _tmp = (np.array(Image.open(self.masks[_im_ii]))).astype(np.float32)
        _void_pixels = np.zeros_like(_tmp).astype(np.uint8)
        _tmp[_void_pixels] = 0

        _other_same_class = np.zeros(_tmp.shape)
        _other_classes = np.zeros(_tmp.shape)

        if self.default:
            _target = _tmp
        else:
            _target = (_tmp == _obj_ii).astype(np.float32)

        return _img, _target, _void_pixels.astype(np.float32)

    def __str__(self):
        return 'CityScapes(split=' + str(self.split) + ',area_thres=' + str(self.area_thres) + ')'
    

class GrabCutSegmentation(Dataset):
    """
        Include 50 samples, only use for evaluation.
    """
    BASE_DIR = 'grabcut'

    def __init__(self,
                 root='dataset',
                 transform=None,
                 preprocess=False,
                 area_thres=0,
                 retname=True,
                 suppress_void_pixels=True,
                 default=False):

        self.root = root
        _grabcut_dir = os.path.join(self.root, self.BASE_DIR)
        self.transform = transform
        self.area_thres = area_thres
        self.retname = retname
        self.suppress_void_pixels = suppress_void_pixels
        self.default = default

        self.sample_list = os.listdir(_grabcut_dir)
        self.im_ids = []
        self.images = []
        self.masks = []
        for sample in self.sample_list:
            if '.png' in sample:
                self.im_ids.append(sample.split('.')[-2])
                self.masks.append(os.path.join(_grabcut_dir, sample))
            else:
                self.images.append(os.path.join(_grabcut_dir, sample))

        assert (len(self.images) == len(self.masks))

        # Build the ids file
        area_th_str = ""
        if self.area_thres != 0:
            area_th_str = '_area_thres-' + str(area_thres)

        # Build the list of objects
        self.obj_list = []
        num_images = 0
        for ii in range(len(self.im_ids)):
            self.obj_list.append([ii, 1])
            num_images += 1

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):
        _img, _target, _void_pixels = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels}

        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.obj_list[index][1]
            _target_area = np.where(_target>0)
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'object': str(_obj_ii),
                              'category': 1,
                              'im_size': (_img.shape[0], _img.shape[1]),
                              'boundary': [_target_area[0].max(), _target_area[0].min(),
                                           _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.obj_list)

    def _make_img_gt_point_pair(self, index):
        _im_ii = self.obj_list[index][0]
        _obj_ii = self.obj_list[index][1]

        # Read Image
        _img = np.array(Image.open(self.images[_im_ii]).convert('RGB')).astype(np.float32)

        # Read Target object
        _tmp = cv2.imread(self.masks[_im_ii], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        _void_pixels = (_tmp == 128)
        _tmp[_void_pixels] = 0

        _other_same_class = np.zeros(_tmp.shape)
        _other_classes = np.zeros(_tmp.shape)

        if self.default:
            _target = _tmp
        else:
            _target = (_tmp == 255).astype(np.float32)

        return _img, _target, _void_pixels.astype(np.float32)

    def __str__(self):
        return 'GrapCut(split=' + str(self.split) + ',area_thres=' + str(self.area_thres) + ')'


class VOCSegmentation(Dataset):
    BASE_DIR = 'VOCdevkit/VOC2012'

    category_names = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self,
                 root='pascal',
                 split='val',
                 transform=None,
                 preprocess=False,
                 area_thres=0,
                 retname=True,
                 suppress_void_pixels=True,
                 default=False):

        self.root = root
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationObject')
        _cat_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self.transform = transform
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.area_thres = area_thres
        self.retname = retname
        self.suppress_void_pixels = suppress_void_pixels
        self.default = default

        # Build the ids file
        area_th_str = ""
        if self.area_thres != 0:
            area_th_str = '_area_thres-' + str(area_thres)

        self.obj_list_file = os.path.join(self.root, self.BASE_DIR, 'ImageSets', 'Segmentation',
                                          '_'.join(self.split) + '_instances' + area_th_str + '.txt')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.masks = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(_image_dir, line + ".jpg")
                _cat = os.path.join(_cat_dir, line + ".png")
                _mask = os.path.join(_mask_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_mask)
                self.im_ids.append(line.rstrip('\n'))
                self.images.append(_image)
                self.categories.append(_cat)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))
        assert (len(self.images) == len(self.categories))

        # Precompute the list of objects and their categories for each image 
        if (not self._check_preprocess()) or preprocess:
            print('Preprocessing of PASCAL VOC dataset, this will take long, but it will be done only once.')
            self._preprocess()

        # Build the list of objects
        self.obj_list = []
        num_images = 0
        for ii in range(len(self.im_ids)):
            flag = False
            for jj in range(len(self.obj_dict[self.im_ids[ii]])):
                if self.obj_dict[self.im_ids[ii]][jj] != -1:
                    self.obj_list.append([ii, jj])
                    flag = True
            if flag:
                num_images += 1

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):
        _img, _target, _void_pixels = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels}

        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.obj_list[index][1]
            _target_area = np.where(_target>0)
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'object': str(_obj_ii),
                              'category': self.obj_dict[self.im_ids[_im_ii]][_obj_ii],
                              'im_size': (_img.shape[0], _img.shape[1]),
                              'boundary': [_target_area[0].max(), _target_area[0].min(),
                                           _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.obj_list)

    def _check_preprocess(self):
        _obj_list_file = self.obj_list_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))

            return list(np.sort([str(x) for x in self.obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess(self):
        self.obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            _mask = np.array(Image.open(self.masks[ii]))
            _mask_ids = np.unique(_mask)
            if _mask_ids[-1] == 255:
                n_obj = _mask_ids[-2]
            else:
                n_obj = _mask_ids[-1]

            # Get the categories from these objects
            _cats = np.array(Image.open(self.categories[ii]))
            _cat_ids = []
            for jj in range(n_obj):
                tmp = np.where(_mask == jj + 1)
                obj_area = len(tmp[0])

                if obj_area > self.area_thres:
                    _cat_ids.append(int(_cats[tmp[0][0], tmp[1][0]]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.obj_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

    def _make_img_gt_point_pair(self, index):
        _im_ii = self.obj_list[index][0]
        _obj_ii = self.obj_list[index][1]

        # Read Image
        _img = np.array(Image.open(self.images[_im_ii]).convert('RGB')).astype(np.float32)

        # Read Target object
        _tmp = (np.array(Image.open(self.masks[_im_ii]))).astype(np.float32)
        _void_pixels = (_tmp == 255)
        _tmp[_void_pixels] = 0

        _other_same_class = np.zeros(_tmp.shape)
        _other_classes = np.zeros(_tmp.shape)

        if self.default:
            _target = _tmp
        else:
            _target = (_tmp == (_obj_ii + 1)).astype(np.float32)

        return _img, _target, _void_pixels.astype(np.float32)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ',area_thres=' + str(self.area_thres) + ')'

        
class SBDSegmentation(Dataset):
    URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
    FILE = "benchmark.tgz"
    MD5 = '82b4d87ceb2ed10f6038a1cba92111cb'

    def __init__(self,
                 root='sbd',
                 split='val',
                 transform=None,
                 download=False,
                 preprocess=False,
                 area_thres=0,
                 retname=True):

        # Store parameters
        self.root = root
        self.transform = transform
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.area_thres = area_thres
        self.retname = retname

        # Where to find things according to the author's structure
        self.dataset_dir = os.path.join(self.root, 'benchmark_RELEASE', 'dataset')
        _mask_dir = os.path.join(self.dataset_dir, 'inst')
        _image_dir = os.path.join(self.dataset_dir, 'img')

        if self.area_thres != 0:
            self.obj_list_file = os.path.join(self.dataset_dir, '_'.join(self.split) + '_instances_area_thres-' +
                                              str(area_thres) + '.txt')
        else:
            self.obj_list_file = os.path.join(self.dataset_dir, '_'.join(self.split) + '_instances' + '.txt')

        # Download dataset?
        if download:
            self._download()
            if not self._check_integrity():
                raise RuntimeError('Dataset file downloaded is corrupted.')

        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.masks = []
        for splt in self.split:
            with open(os.path.join(self.dataset_dir, splt + '.txt'), "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                _image = os.path.join(_image_dir, line + ".jpg")
                _mask = os.path.join(_mask_dir, line + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.im_ids.append(line)
                self.images.append(_image)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

        # Precompute the list of objects and their categories for each image 
        if (not self._check_preprocess()) or preprocess:
            print('Preprocessing SBD dataset, this will take long, but it will be done only once.')
            self._preprocess()

        # Build the list of objects
        self.obj_list = []
        num_images = 0
        for ii in range(len(self.im_ids)):
            if self.im_ids[ii] in self.obj_dict.keys():
                flag = False
                for jj in range(len(self.obj_dict[self.im_ids[ii]])):
                    if self.obj_dict[self.im_ids[ii]][jj] != -1:
                        self.obj_list.append([ii, jj])
                        flag = True
                if flag:
                    num_images += 1

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):

        _img, _target = self._make_img_gt_point_pair(index)
        _void_pixels = (_target == 255).astype(np.float32)
        sample = {'image': _img, 'gt': _target, 'void_pixels': _void_pixels}

        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.obj_list[index][1]
            _target_area = np.where(_target>0)
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'object': str(_obj_ii),
                              'im_size': (_img.shape[0], _img.shape[1]),
                              'category': self.obj_dict[self.im_ids[_im_ii]][_obj_ii],
                              'boundary': [_target_area[0].max(), _target_area[0].min(),
                                           _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.obj_list)

    def _check_integrity(self):
        _fpath = os.path.join(self.root, self.FILE)
        if not os.path.isfile(_fpath):
            print("{} does not exist".format(_fpath))
            return False
        _md5c = hashlib.md5(open(_fpath, 'rb').read()).hexdigest()
        if _md5c != self.MD5:
            print(" MD5({}) did not match MD5({}) expected for {}".format(
                _md5c, self.MD5, _fpath))
            return False
        return True

    def _check_preprocess(self):
        # Check that the file with categories is there and with correct size
        _obj_list_file = self.obj_list_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))
            return list(np.sort([str(x) for x in self.obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess(self):
        # Get all object instances and their category
        self.obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            tmp = scipy.io.loadmat(self.masks[ii])
            _mask = tmp["GTinst"][0]["Segmentation"][0]
            _cat_ids = tmp["GTinst"][0]["Categories"][0].astype(int)

            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]
            assert (n_obj == len(_cat_ids))

            for jj in range(n_obj):
                temp = np.where(_mask == jj + 1)
                obj_area = len(temp[0])
                if obj_area < self.area_thres:
                    _cat_ids[jj] = -1
                obj_counter += 1

            self.obj_dict[self.im_ids[ii]] = np.squeeze(_cat_ids, 1).tolist()

        # Save it to file for future reference
        with open(self.obj_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Pre-processing finished')

    def _download(self):
        _fpath = os.path.join(self.root, self.FILE)

        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        else:
            print('Downloading ' + self.URL + ' to ' + _fpath)

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (_fpath, float(count * block_size) /
                                  float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, _fpath, _progress)

        # extract file
        cwd = os.getcwd()
        print('Extracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def _make_img_gt_point_pair(self, index):
        _im_ii = self.obj_list[index][0]
        _obj_ii = self.obj_list[index][1]

        # Read Image
        _img = np.array(Image.open(self.images[_im_ii]).convert('RGB')).astype(np.float32)

        # Read Taret object
        _tmp = scipy.io.loadmat(self.masks[_im_ii])["GTinst"][0]["Segmentation"][0]
        _target = (_tmp == (_obj_ii + 1)).astype(np.float32)

        return _img, _target

    def __str__(self):
        return 'SBDSegmentation(split=' + str(self.split) + ', area_thres=' + str(self.area_thres) + ')'


class CombineDBs(Dataset):
    def __init__(self, dataloaders, excluded=None):
        self.dataloaders = dataloaders
        self.excluded = excluded
        self.im_ids = []

        # Combine object lists
        for dl in dataloaders:
            for elem in dl.im_ids:
                if elem not in self.im_ids:
                    self.im_ids.append(elem)

        # Exclude
        if excluded:
            for dl in excluded:
                for elem in dl.im_ids:
                    if elem in self.im_ids:
                        self.im_ids.remove(elem)

        # Get object pointers
        self.obj_list = []
        self.im_list = []
        new_im_ids = []
        obj_counter = 0
        num_images = 0
        for ii, dl in enumerate(dataloaders):
            for jj, curr_im_id in enumerate(dl.im_ids):
                if (curr_im_id in self.im_ids) and (curr_im_id not in new_im_ids):
                    flag = False
                    new_im_ids.append(curr_im_id)
                    for kk in range(len(dl.obj_dict[curr_im_id])):
                        if dl.obj_dict[curr_im_id][kk] != -1:
                            self.obj_list.append({'db_ii': ii, 'obj_ii': dl.obj_list.index([jj, kk])})
                            flag = True
                        obj_counter += 1
                    self.im_list.append({'db_ii': ii, 'im_ii': jj})
                    if flag:
                        num_images += 1

        self.im_ids = new_im_ids
        print('Combined number of images: {:d}\nCombined number of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):

        _db_ii = self.obj_list[index]["db_ii"]
        _obj_ii = self.obj_list[index]['obj_ii']
        sample = self.dataloaders[_db_ii].__getitem__(_obj_ii)

        if 'meta' in sample.keys():
            sample['meta']['db'] = str(self.dataloaders[_db_ii])

        return sample

    def __len__(self):
        return len(self.obj_list)

    def __str__(self):
        include_db = [str(db) for db in self.dataloaders]
        exclude_db = [str(db) for db in self.excluded]
        return 'Included datasets:' + str(include_db) + '\n' + 'Excluded datasets:' + str(exclude_db)


class CombineContext(Dataset):
    def __init__(self, dataloaders, excluded=[]):
        self.dataloaders = dataloaders
        self.excluded = excluded
        self.im_ids = []

        # Combine object lists
        for dl in dataloaders:
            for elem in dl.im_ids:
                if elem not in self.im_ids:
                    self.im_ids.append(elem)

        # Exclude
        if excluded:
            for dl in excluded:
                for elem in dl.im_ids:
                    if elem in self.im_ids:
                        self.im_ids.remove(elem)

        # Get object pointers
        self.obj_list = []
        self.im_list = []
        new_im_ids = []
        obj_counter = 0
        num_images = 0
        for ii, dl in enumerate(dataloaders):
            for jj, curr_im_id in enumerate(dl.im_ids):
                flag = False
                new_im_ids.append(curr_im_id)
                for kk in range(len(dl.obj_dict[curr_im_id])):
                    if dl.obj_dict[curr_im_id][kk] != -1 and dl.obj_dict[curr_im_id][kk] != (-1, -1):
                        try:
                            if isinstance(dl.obj_list[0][1], tuple) or isinstance(dl.obj_list[0][1], list):
                                obj_id = dl.obj_dict[curr_im_id][kk]
                                self.obj_list.append({'db_ii': ii, 'obj_ii': dl.obj_list.index([jj, obj_id])})
                            else:
                                self.obj_list.append({'db_ii': ii, 'obj_ii': dl.obj_list.index([jj, kk])})
                            flag = True
                        except Exception:
                            import IPython
                            IPython.embed()
                    obj_counter += 1
                    self.im_list.append({'db_ii': ii, 'im_ii': jj})
                    if flag:
                        num_images += 1

        self.im_ids = new_im_ids
        print('Combined number of images: {:d}\nCombined number of objects: {:d}'.format(len(self.im_ids),
                                                                                         len(self.obj_list)))

    def __getitem__(self, index):

        _db_ii = self.obj_list[index]["db_ii"]
        _obj_ii = self.obj_list[index]['obj_ii']
        sample = self.dataloaders[_db_ii].__getitem__(_obj_ii)

        if 'meta' in sample.keys():
            sample['meta']['db'] = str(self.dataloaders[_db_ii])

        return sample

    def __len__(self):
        return len(self.obj_list)

    def __str__(self):
        include_db = [str(db) for db in self.dataloaders]
        exclude_db = [str(db) for db in self.excluded]
        return 'Included datasets:' + str(include_db) + '\n' + 'Excluded datasets:' + str(exclude_db)


# ======== Prefetcher ========
class Prefetcher():
    """
    Modified from the data_prefetcher in https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    """

    def __init__(self, loader):
        self.orig_loader = loader
        self.stream = torch.cuda.Stream()
        self.next_sample = None

    def preload(self):
        try:
            self.next_sample = next(self.loader)
        except StopIteration:
            self.next_sample = None
            return

        with torch.cuda.stream(self.stream):
            for key, value in self.next_sample.items():
                if isinstance(value, torch.Tensor):
                    self.next_sample[key] = value.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        sample = self.next_sample
        if sample is not None:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key].record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            # throw stop exception if there is no more data to perform as a default dataloader
            raise StopIteration("No samples in loader. example: `iterator = iter(Prefetcher(loader)); "
                                "data = next(iterator)`")
        return sample

    def __iter__(self):
        self.loader = iter(self.orig_loader)
        self.preload()
        return self
