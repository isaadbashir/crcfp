import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from base import BaseDataSet, BaseDataLoader
from utils import pallete

class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image):
        image = np.asarray(image)
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        image = transforms.functional.to_pil_image(image)
        return image


class PairMoNuSegDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2

        self.datalist = kwargs.pop("datalist")
        self.stride = kwargs.pop('stride')
        self.iou_bound = kwargs.pop('iou_bound')  # default [0.3, 0.7]
        self.data_dir = kwargs['data_dir']
        self.n_split = kwargs.pop('n_labeled_examples')
        self.folder = kwargs.pop('folder')
        self.config_dir = kwargs.pop('config_dir')

        self.palette = pallete.get_voc_pallete(self.num_classes)

        super(PairMoNuSegDataset, self).__init__(**kwargs)

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            RandomGaussianBlur(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])

    def _set_files(self):

        self.root = os.path.join(self.data_dir, self.folder)

        if self.split == "val":
            file_list_path = os.path.join(self.config_dir, f'test_1by{self.n_split}_list.npy')
        elif self.split == "train_supervised":
            file_list_path = os.path.join(self.config_dir, f'supervised_1by{self.n_split}_list.npy')
        elif self.split == "train_unsupervised":
            file_list_path = os.path.join(self.config_dir, f'unsupervised_1by{self.n_split}_list.npy')

        self.file_list = np.load(file_list_path)

        self.file_list = [f'{self.root}/{x}' for x in self.file_list]

        # shuffle the list of images for randomness
        random.shuffle(self.file_list)

        self.files = self.file_list

    def _load_data(self, index):
        img_data = np.load(self.files[index])
        image, label = img_data[:, :, :3], img_data[:, :, 3]

        return image, label, self.files[index].split('/')[-1].split('_')[0]

    def __getitem__(self, index):

        img_data = np.load(self.files[index])
        image, label = img_data[:, :, :3], img_data[:, :, 3]

        h, w, _ = image.shape

        longside = random.randint(int(self.base_size * 0.8), int(self.base_size * 2.0))
        h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
        image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))

        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        crop_h, crop_w = self.crop_size, self.crop_size
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_index, **pad_kwargs)

        x1 = random.randint(0, w + pad_w - crop_w)
        y1 = random.randint(0, h + pad_h - crop_h)

        max_iters = 50
        k = 0
        while k < max_iters:
            x2 = random.randint(0, w + pad_w - crop_w)
            y2 = random.randint(0, h + pad_h - crop_h)
            # crop relative coordinates should be a multiple of 8
            x2 = (x2 - x1) // self.stride * self.stride + x1
            y2 = (y2 - y1) // self.stride * self.stride + y1
            if x2 < 0:
                x2 += self.stride
            if y2 < 0:
                y2 += self.stride

            if crop_w - abs(x2 - x1) < 0 or crop_h - abs(y2 - y1) < 0:
                k += 1
                continue

            inter = (crop_w - abs(x2 - x1)) * (crop_h - abs(y2 - y1))
            union = 2 * crop_w * crop_h - inter
            iou = inter / union
            if iou >= self.iou_bound[0] and iou <= self.iou_bound[1]:
                break
            k += 1

        if k == max_iters:
            x2 = x1
            y2 = y1

        overlap1_ul = [max(0, y2 - y1), max(0, x2 - x1)]
        overlap1_br = [min(self.crop_size, self.crop_size + y2 - y1, h // self.stride * self.stride), min(self.crop_size, self.crop_size + x2 - x1, w // self.stride * self.stride)]
        overlap2_ul = [max(0, y1 - y2), max(0, x1 - x2)]
        overlap2_br = [min(self.crop_size, self.crop_size + y1 - y2, h // self.stride * self.stride), min(self.crop_size, self.crop_size + x1 - x2, w // self.stride * self.stride)]

        try:
            assert (overlap1_br[0] - overlap1_ul[0]) * (overlap1_br[1] - overlap1_ul[1]) == (overlap2_br[0] - overlap2_ul[0]) * (overlap2_br[1] - overlap2_ul[1])
        except:
            print("h: {}, w: {}".format(h, w))
            print("image.shape: ", image.shape)
            print("x1: {}, x2: {}, y1: {}, y2: {}".format(x1, x2, y1, y2))
            print("ul1: ", overlap1_ul)
            print("br1: ", overlap1_br)
            print("ul2: ", overlap2_ul)
            print("br2: ", overlap2_br)
            print("index: ", index)
            exit()

        image1 = image[y1:y1 + self.crop_size, x1:x1 + self.crop_size].copy()
        image2 = image[y2:y2 + self.crop_size, x2:x2 + self.crop_size].copy()
        label1 = label[y1:y1 + self.crop_size, x1:x1 + self.crop_size].copy()
        label2 = label[y2:y2 + self.crop_size, x2:x2 + self.crop_size].copy()

        try:
            assert image1[overlap1_ul[0]:overlap1_br[0], overlap1_ul[1]:overlap1_br[1]].shape == image2[overlap2_ul[0]:overlap2_br[0], overlap2_ul[1]:overlap2_br[1]].shape
        except:
            print("h: {}, w: {}".format(h, w))
            print("image.shape: ", image.shape)
            print("x1: {}, x2: {}, y1: {}, y2: {}".format(x1, x2, y1, y2))
            print("ul1: ", overlap1_ul)
            print("br1: ", overlap1_br)
            print("ul2: ", overlap2_ul)
            print("br2: ", overlap2_br)
            print("index: ", index)
            exit()

        flip1 = False
        if random.random() < 0.5:
            image1 = np.fliplr(image1)
            label1 = np.fliplr(label1)
            flip1 = True

        flip2 = False
        if random.random() < 0.5:
            image2 = np.fliplr(image2)
            label2 = np.fliplr(label2)
            flip2 = True
        flip = [flip1, flip2]

        image = cv2.resize(image, (image1.shape[0], image1.shape[1]), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)

        image1 = self.train_transform(image1)
        image2 = self.train_transform(image2)
        image = self.train_transform(image)

        images = torch.stack([image1, image2])
        labels = torch.from_numpy(np.stack([label1, label2]))

        return image, label, images, labels, overlap1_ul, overlap1_br, overlap2_ul, overlap2_br, flip


class PairMoNuSeg(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')
        self.dataset = PairMoNuSegDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)

        super(PairMoNuSeg, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None, dist_sampler=dist_sampler)


class MoNuSegDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2

        self.datalist = kwargs.pop("datalist")
        self.data_dir = kwargs['data_dir']
        self.n_split = kwargs.pop('n_labeled_examples')
        self.folder = kwargs.pop('folder')
        self.config_dir = kwargs.pop('config_dir')

        self.palette = pallete.get_voc_pallete(self.num_classes)

        super(MoNuSegDataset, self).__init__(**kwargs)

    def _set_files(self):

        self.root = os.path.join(self.data_dir, self.folder)

        if self.split == "val":
            file_list_path = os.path.join(self.config_dir, f'test_1by{self.n_split}_list.npy')
        elif self.split == "train_supervised":
            file_list_path = os.path.join(self.config_dir, f'supervised_1by{self.n_split}_list.npy')
        elif self.split == "train_unsupervised":
            file_list_path = os.path.join(self.config_dir, f'unsupervised_1by{self.n_split}_list.npy')

        self.file_list = np.load(file_list_path)

        self.file_list = [f'{self.root}/{x}' for x in self.file_list]

        # shuffle the list of images for randomness
        random.shuffle(self.file_list)

        self.files = self.file_list

    def _load_data(self, index):
        img_data = np.load(self.files[index])
        image, label = img_data[:, :, :3], img_data[:, :, 3]

        return image, label, self.files[index].split('/')[-1].split('_')[0]


class MoNuSeg(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')

        self.dataset = MoNuSegDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)
        super(MoNuSeg, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None, dist_sampler=dist_sampler)
