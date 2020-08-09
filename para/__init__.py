import argparse
import torch
import numpy as np
import random

class Parameter:
    def __init__(self):
        self.args = self.extract_args()

    def extract_args(self):
        self.parser = argparse.ArgumentParser(description='Mobile Video Deblurring')

        # experiment mark
        self.parser.add_argument('--description', type=str, default='develop', help='experiment description')

        # global parameters
        self.parser.add_argument('--seed', type=int, default=39, help='random seed')
        self.parser.add_argument('--threads', type=int, default=4, help='# of threads for dataloader')
        self.parser.add_argument('--cpu', action='store_true', help='run on CPU, not recommended')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='# of GPUs to use')
        self.parser.add_argument('--no_profile', action='store_true', help='show # of parameters and computation cost')
        self.parser.add_argument('--profile_H', type=int, default=720, help='height of image to generate profile of model')
        self.parser.add_argument('--profile_W', type=int, default=1280, help='width of image to generate profile of model')
        self.parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_file', type=str, default='', help='the path of checkpoint file')
        self.parser.add_argument('--dependency', type=str, default='lmdb tqdm thop', help='the libs that need to be installed on the cloud')

        # data parameters
        self.parser.add_argument('--data_root', type=str, default='/home/zhong/data/', help='the path of dataset')
        self.parser.add_argument('--dataset', type=str, default='gopro_ds', help='the dataset used in the project')
        self.parser.add_argument('--save_dir', type=str, default='./experiment/', help='directory to save logs of experiments')
        self.parser.add_argument('--loader_mode', type=str, default='lmdb', help='mode of dataloader')
        self.parser.add_argument('--frames', type=int, default=10, help='# of frames of subsequence')
        # self.parser.add_argument('--patch_size', type=int, default=256, help='patch size for cropping')

        # model parameters
        self.parser.add_argument('--model', type=str, default='ESTRNN', help='type of model to construct')
        self.parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
        self.parser.add_argument('--n_blocks', type=int, default=9, help='# of blocks in middle part of the model')
        self.parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
        self.parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
        self.parser.add_argument('--uncentralized', action='store_true', help='do not subtract the value of mean')
        self.parser.add_argument('--normalized', action='store_true', help='divide the range of rgb (255)')

        # loss parameters
        self.parser.add_argument('--loss', type=str, default='1*MSE', help='type of loss function, e.g. 1*MSE|1e-4*GAN')
        
        # metrics parameters
        self.parser.add_argument('--metrics', type=str, default='PSNR', help='type of evaluation metrics')

        # optimizer parameters
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='method of optimization')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        self.parser.add_argument('--milestones', type=int, nargs='*', default=[200,400])
        self.parser.add_argument('--decay_gamma', type=float, default=0.5, help='decay rate')

        # training parameters
        self.parser.add_argument('--start_epoch', type=int, default=1, help='first epoch number')
        self.parser.add_argument('--end_epoch', type=int, default=500, help='last epoch number')
        self.parser.add_argument('--trainer_mode', type=str, default='dp', help='trainer mode: distributed data parallel (ddp) or data parallel (dp)')

        args, _ = self.parser.parse_known_args()

        return args



