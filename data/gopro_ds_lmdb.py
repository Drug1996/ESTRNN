from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np
import torch
import lmdb
import pickle
from os.path import join


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
        if dataset_type == 'train':
            self.datapath_blur = join(datapath, 'gopro_ds_train')
            self.datapath_gt = join(datapath, 'gopro_ds_train_gt')
            f = open(join(datapath, 'gopro_ds_info_train.pkl'), 'rb')
            self.seqs_info = pickle.load(f)
            f.close()
            self.transform = transforms.Compose([Crop(256), Flip(), ToTensor()])
        elif dataset_type == 'valid':
            self.datapath_blur = join(datapath, 'gopro_ds_valid')
            self.datapath_gt = join(datapath, 'gopro_ds_valid_gt')
            f = open(join(datapath, 'gopro_ds_info_valid.pkl'), 'rb')
            self.seqs_info = pickle.load(f)
            f.close()
            self.transform = transforms.Compose([Crop(256), ToTensor()])
        self.verbose = verbose
        self.seq_num = self.seqs_info['num']
        self.seq_id_start = 0
        self.seq_id_end = self.seq_num - 1
        self.frames = frames
        self.crop_size = 256
        self.W = 960
        self.H = 540
        self.down_ratio = 1
        self.C = 3
        self.num_ff = num_ff
        self.num_fb = num_fb
        self.env_blur = lmdb.open(self.datapath_blur, map_size=1099511627776)
        self.env_gt = lmdb.open(self.datapath_gt, map_size=1099511627776)
        self.txn_blur = self.env_blur.begin()
        self.txn_gt = self.env_gt.begin()

    def get_index(self):
        seq_idx = random.randint(self.seq_id_start, self.seq_id_end)
        frame_idx = random.randint(0, self.seqs_info[seq_idx]['length'] - self.frames)

        return seq_idx, frame_idx

    def get_img(self, seq_idx, frame_idx, sample):
        code = '%03d_%08d' % (seq_idx, frame_idx)
        code = code.encode()
        img_blur = self.txn_blur.get(code)
        img_blur = np.frombuffer(img_blur, dtype='uint8')
        img_blur = img_blur.reshape(self.H, self.W, self.C)
        img_gt = self.txn_gt.get(code)
        img_gt = np.frombuffer(img_gt, dtype='uint8')
        img_gt = img_gt.reshape(self.H, self.W, self.C)
        sample['image'] = img_blur
        sample['label'] = img_gt
        sample = self.transform(sample)
        if self.verbose:
            print('code', code, 's2s', sample['s2s'], 'top', sample['top'], 'left', sample['left'], 'flip_lr',
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
        seq_idx, frame_idx = self.get_index()
        imgs_blur = []
        imgs_gt = []
        for i in range(self.frames):
            img_blur, img_gt = self.get_img(seq_idx, frame_idx + i, sample)
            imgs_blur.append(img_blur)
            imgs_gt.append(img_gt)
        return torch.cat(imgs_blur, dim=0), torch.cat(imgs_gt[self.num_fb:self.frames - self.num_ff], dim=0)

    def __len__(self):
        return self.seqs_info['length'] - (self.frames - 1) * self.seqs_info['num']


class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        self.para = para
        path = join(para.data_root, para.dataset+'_lmdb')
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
                num_workers=0,
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
