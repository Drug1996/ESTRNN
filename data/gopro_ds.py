from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np
import torch
import os
from os.path import join
import cv2


class Crop(object):
    # Crop randomly the image in a sample.
    # Args: output_size (tuple or int): Desired output size. If int, square crop is made.
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        top, left = sample['top'], sample['left']
        new_h, new_w = self.output_size
        sample['image'] = image[top: top + new_h,
                          left: left + new_w]
        sample['label'] = label[top: top + new_h,
                          left: left + new_w]

        return sample


class Flip(object):
    # shape is (h,w,c)
    def __call__(self, sample):
        flag_lr = sample['flip_lr']
        flag_ud = sample['flip_ud']
        if flag_lr == 1:
            sample['image'] = np.fliplr(sample['image'])
            sample['label'] = np.fliplr(sample['label'])
        if flag_ud == 1:
            sample['image'] = np.flipud(sample['image'])
            sample['label'] = np.flipud(sample['label'])

        return sample


class Rotate(object):
    # shape is (h,w,c)
    def __call__(self, sample):
        flag = sample['rotate']
        if flag == 1:
            sample['image'] = sample['image'].transpose(1, 0, 2)
            sample['label'] = sample['label'].transpose(1, 0, 2)

        return sample


class Sharp2Sharp(object):
    def __call__(self, sample):
        flag = sample['s2s']
        if flag < 1:
            sample['image'] = sample['label'].copy()
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[np.newaxis, :])
        label = np.ascontiguousarray(label.transpose((2, 0, 1))[np.newaxis, :])
        sample['image'] = torch.from_numpy(image).float()
        sample['label'] = torch.from_numpy(label).float()
        return sample


class DeblurDataset(Dataset):
    def __init__(self, datapath='./lmdb_dataset/', dataset_type='train', frames=8, num_ff=2, num_fb=2, verbose=False):
        # generate files
        self.files = self.generate_files(join(datapath, dataset_type), frames)
        self.verbose = verbose
        self.frames = frames
        self.crop_size = 256
        self.W = 960
        self.H = 540
        self.C = 3
        self.down_ratio = 1
        self.num_ff = num_ff
        self.num_fb = num_fb
        # #         self.transform = transforms.Compose([Crop(256), Flip(), Rotate(), Sharp2Sharp(), ToTensor()])
        self.transform = transforms.Compose([Crop(256), Flip(), ToTensor()])

    def generate_files(self, path, frames):
        dirs = os.listdir(path) # path: .../gopro/train/, dirs: seqs dir
        files = []
        for dir in dirs:
            subpath = join(path, dir)
            sub_files = os.listdir(join(subpath, 'blur_gamma'))
            sub_files.sort()
            for i in range(len(sub_files)-frames+1):
                samples = [
                           [ join(subpath, 'blur_gamma', file) for file in sub_files[i:i+frames] ],
                           [ join(subpath, 'sharp', file) for file in sub_files[i:i+frames] ]
                          ]
                files.append(samples)
        return files

    def get_index(self):
        idx = random.randint(0, len(self.files)-1)

        return idx

    def get_img(self, input_addr, label_addr, sample):
        img_blur = cv2.imread(input_addr)
        img_gt = cv2.imread(label_addr)
        sample['image'] = img_blur
        sample['label'] = img_gt
        sample = self.transform(sample)
        if self.verbose:
            print('input_addr', input_addr, 'label_addr', label_addr, 's2s', sample['s2s'], 'top', sample['top'], 'left', sample['left'], 'flip_lr',
                  sample['flip_lr'], 'flip_ud', sample['flip_ud'], 'rotate',
                  sample['rotate'])

        return sample['image'], sample['label']

    def __getitem__(self, idx):
        top = random.randint(0, int(self.H * self.down_ratio) - self.crop_size)
        left = random.randint(0, int(self.W * self.down_ratio) - self.crop_size)
        flip_lr_flag = random.randint(0, 1)
        flip_ud_flag = random.randint(0, 1)
        rotate_flag = random.randint(0, 1)
        s2s_flag = random.randint(0, 9)
        #         reverse_flag = random.randint(0, 1)
        sample = {'s2s': s2s_flag, 'top': top, 'left': left, 'flip_lr': flip_lr_flag, 'flip_ud': flip_ud_flag,
                  'rotate': rotate_flag}
        idx = self.get_index()
        inputs_addr, labels_addr = self.files[idx]
        imgs_blur = []
        imgs_gt = []
        for i in range(self.frames):
            img_blur, img_gt = self.get_img(inputs_addr[i], labels_addr[i], sample)
            imgs_blur.append(img_blur)
            imgs_gt.append(img_gt)
        return torch.cat(imgs_blur, dim=0), torch.cat(imgs_gt[self.num_fb:self.frames - self.num_ff], dim=0)

    def __len__(self):
        return len(self.files)

class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        self.para = para
        path = join(para.data_root, para.dataset)
        self.dataset = DeblurDataset(path, ds_type, para.frames, para.future_frames, para.past_frames)
        gpus = self.para.num_gpus
        bs = self.para.batch_size
        ds_len = len(self.dataset)
        if para.trainer_mode == 'ddp':
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=para.num_gpus,
                rank=device_id
            )
            self.loader = DataLoader(
                dataset=self.dataset,
                batch_size=para.batch_size,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True,
                sampler=sampler
            )
            loader_len = np.ceil(ds_len / gpus)
            self.loader_len = int(np.ceil(loader_len / bs) * bs)

        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=self.dataset,
                batch_size=para.batch_size,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len

    def reset(self):
        pass