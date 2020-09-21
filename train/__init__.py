import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from importlib import import_module
from .optimizer import Optimizer
from model import Model
from data import Data
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from util.logger import Logger
from datetime import datetime
import pickle
import lmdb
import cv2
from os.path import join, dirname


# reduce tensor from multiple gpus
def reduce_tensor(para, ts):
    dist.reduce(ts, dst=0, op=dist.ReduceOp.SUM)
    ts /= para.num_gpus
    return ts


# computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self, para):
        self.para = para
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '6666'

    def run(self):
        # recoding parameters
        self.para.time = datetime.now()
        logger = Logger(self.para)
        logger.record_para()

        # record model profile: computation cost & # of parameters
        if not self.para.no_profile:
            self.profile(logger)

        # training
        if not self.para.test_only:
            if self.para.trainer_mode == 'ddp':
                gpus = self.para.num_gpus
                mp.spawn(dist_proc, nprocs=gpus, args=(self.para,))
            elif self.para.trainer_mode == 'dp':
                proc(self.para)

        # test
        test(self.para, logger)

    def profile(self, logger):
        model = Model(self.para)
        flops, params = model.profile()
        del model
        logger('generating profile of {} model ...'.format(self.para.model), prefix='\n')
        logger('[profile] computation cost: {:.2f} GMACs, parameters: {:.2f} M'.format(
            flops / 10 ** 9, params / 10 ** 6), timestamp=False)


# *************************************************************
# distributed data parallel training
def dist_proc(gpu, para):
    rank = gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=para.num_gpus,
        rank = rank
    )

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(rank)

    # set random seed
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    # create logger
    logger = Logger(para) if rank==0 else None

    # create model
    if logger: logger('building {} model ...'.format(para.model), prefix='\n')
    model = Model(para).model
    model = model.cuda(rank)
    if logger: logger('model structure:', model, verbose=False)

    # create criterion according to the loss function
    # loss_name = para.loss
    module = import_module('train.loss')
    criterion = getattr(module, 'Loss')(para).cuda(rank)

    # create measurement according to metrics
    metrics_name = para.metrics
    module = import_module('train.metrics')
    metrics = getattr(module, metrics_name)().cuda(rank)

    # create optimizer
    opt = Optimizer(para, model)

    # distributed data parallel
    model = DDP(model, device_ids=[rank])

    # create dataloader
    if logger: logger('loading {} dataloader ...'.format(para.dataset), prefix='\n')
    data = Data(para, rank)
    train_loader = data.dataloader_train
    valid_loader = data.dataloader_valid

    # optionally resume from a checkpoint
    if para.resume:
        if os.path.isfile(para.resume_file):
            checkpoint = torch.load(para.resume_file, map_location=lambda storage, loc: storage.cuda(rank))
            if logger:
                logger('loading checkpoint {} ...'.format(para.resume_file))
                logger.register_dict = checkpoint['register_dict']
            para.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            opt.optimizer.load_state_dict(checkpoint['optimizer'])
            opt.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            if logger:
                logger('no check point found at {}'.format(para.resume_file))


    # training and validation
    for epoch in range(para.start_epoch, para.end_epoch+1):
        dist_train(train_loader, model, criterion, metrics, opt, epoch, para, logger)
        dist_valid(valid_loader, model, criterion, metrics, epoch, para, logger)

        # save checkpoint
        if logger:
            is_best = logger.is_best(epoch)
            checkpoint = {
                'epoch': epoch+1,
                'model': para.model,
                'state_dict': model.state_dict(),
                'register_dict': logger.register_dict,
                'optimizer': opt.optimizer.state_dict(),
                'scheduler': opt.scheduler.state_dict()
            }
            logger.save(checkpoint, is_best)

        # reset DALI iterators
        train_loader.reset()
        valid_loader.reset()


def dist_train(train_loader, model, criterion, metrics, opt, epoch, para, logger):
    model.train()

    if logger:
        logger('[Epoch {} / lr {:.2e}]'.format(
            epoch, opt.get_lr()
        ), prefix='\n')
        loss_meter = AverageMeter()
        measure_meter = AverageMeter()
        batchtime_meter = AverageMeter()
        start = time.time()
        end = time.time()
        pbar = tqdm(total=len(train_loader)*para.num_gpus, ncols=80)

    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        measure = metrics(outputs.detach(), labels)

        reduced_loss = reduce_tensor(para, loss.detach())
        reduced_measure = reduce_tensor(para, measure.detach())

        # backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if logger:
            loss_meter.update(reduced_loss.item(), inputs.size(0))
            measure_meter.update(reduced_measure.item(), inputs.size(0))
            # measure elapsed time
            torch.cuda.synchronize()
            batchtime_meter.update(time.time() - end)
            end = time.time()
            pbar.update(para.num_gpus*para.batch_size)

    if logger:
        pbar.close()
        # record info
        logger.register(para.loss + '_train', epoch, loss_meter.avg)
        logger.register(para.metrics + '_train', epoch, measure_meter.avg)
        # show info
        logger('[train] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end-start, batchtime_meter.avg), timestamp=False)
        logger.report([[para.loss, 'min'], [para.metrics, 'max']], state='train', epoch=epoch)

    # adjust learning rate
    opt.lr_schedule()


def dist_valid(valid_loader, model, criterion, metrics, epoch, para, logger):
    model.eval()
    # torch.cuda.empty_cache()

    if logger:
        loss_meter = AverageMeter()
        measure_meter = AverageMeter()
        batchtime_meter = AverageMeter()
        start = time.time()
        end = time.time()
        pbar = tqdm(total=len(valid_loader)*para.num_gpus, ncols=80)

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels, valid_flag=True)
            measure = metrics(outputs.detach(), labels)

            reduced_loss = reduce_tensor(para, loss.detach())
            reduced_measure = reduce_tensor(para, measure.detach())

            if logger:
                loss_meter.update(reduced_loss.item(), inputs.size(0))
                measure_meter.update(reduced_measure.item(), inputs.size(0))
                # measure elapsed time
                torch.cuda.synchronize()
                batchtime_meter.update(time.time() - end)
                end = time.time()
                pbar.update(para.num_gpus * para.batch_size)

    if logger:
        pbar.close()
        # record info
        logger.register(para.loss + '_valid', epoch, loss_meter.avg)
        logger.register(para.metrics + '_valid', epoch, measure_meter.avg)
        # show info
        logger('[valid] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end-start, batchtime_meter.avg), timestamp=False)
        logger.report([[para.loss, 'min'], [para.metrics, 'max']], state='valid', epoch=epoch)

# *************************************************************


# *************************************************************
# data parallel training
def proc(para):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # set random seed
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    # create logger
    logger = Logger(para)

    # create model
    logger('building {} model ...'.format(para.model), prefix='\n')
    model = Model(para).model
    model.cuda()
    logger('model structure:', model, verbose=False)

    # create criterion according to the loss function
    # loss_name = para.loss
    module = import_module('train.loss')
    criterion = getattr(module, 'Loss')(para).cuda()

    # create measurement according to metrics
    metrics_name = para.metrics
    module = import_module('train.metrics')
    metrics = getattr(module, metrics_name)().cuda()

    # create optimizer
    opt = Optimizer(para, model)

    # create dataloader
    logger('loading {} dataloader ...'.format(para.dataset), prefix='\n')
    data = Data(para, device_id=0)
    train_loader = data.dataloader_train
    valid_loader = data.dataloader_valid

    # optionally resume from a checkpoint
    if para.resume:
        if os.path.isfile(para.resume_file):
            checkpoint = torch.load(para.resume_file, map_location=lambda storage, loc: storage.cuda(0))
            logger('loading checkpoint {} ...'.format(para.resume_file))
            logger.register_dict = checkpoint['register_dict']
            para.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            opt.optimizer.load_state_dict(checkpoint['optimizer'])
            opt.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            logger('no check point found at {}'.format(para.resume_file))

    # training and validation
    for epoch in range(para.start_epoch, para.end_epoch + 1):
        train(train_loader, model, criterion, metrics, opt, epoch, para, logger)
        valid(valid_loader, model, criterion, metrics, epoch, para, logger)

        # save checkpoint
        is_best = logger.is_best(epoch)
        checkpoint = {
            'epoch': epoch + 1,
            'model': para.model,
            'state_dict': model.state_dict(),
            'register_dict': logger.register_dict,
            'optimizer': opt.optimizer.state_dict(),
            'scheduler': opt.scheduler.state_dict()
        }
        logger.save(checkpoint, is_best)

        # reset DALI iterators
        train_loader.reset()
        valid_loader.reset()


def train(train_loader, model, criterion, metrics, opt, epoch, para, logger):
    model.train()
    logger('[Epoch {} / lr {:.2e}]'.format(
        epoch, opt.get_lr()
    ), prefix='\n')
    loss_meter = AverageMeter()
    measure_meter = AverageMeter()
    batchtime_meter = AverageMeter()
    start = time.time()
    end = time.time()
    pbar = tqdm(total=len(train_loader), ncols=80)

    for inputs, labels in train_loader:
        # forward
        inputs = inputs.cuda()
        labels = labels.cuda()
        # if # of gpus is larger than 1, use dp
        if para.num_gpus > 1:
            outputs = nn.parallel.data_parallel(
                model,
                inputs,
                range(para.num_gpus)
            )
        else:
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        measure = metrics(outputs.detach(), labels)
        loss_meter.update(loss.detach().item(), inputs.size(0))
        measure_meter.update(measure.detach().item(), inputs.size(0))

        # backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure elapsed time
        batchtime_meter.update(time.time() - end)
        end = time.time()
        pbar.update(para.batch_size)

    pbar.close()
    # record info
    logger.register(para.loss + '_train', epoch, loss_meter.avg)
    logger.register(para.metrics + '_train', epoch, measure_meter.avg)
    # show info
    logger('[train] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end-start, batchtime_meter.avg), timestamp=False)
    logger.report([[para.loss, 'min'], [para.metrics, 'max']], state='train', epoch=epoch)

    # adjust learning rate
    opt.lr_schedule()

def valid(valid_loader, model, criterion, metrics, epoch, para, logger):
    model.eval()

    loss_meter = AverageMeter()
    measure_meter = AverageMeter()
    batchtime_meter = AverageMeter()
    start = time.time()
    end = time.time()
    pbar = tqdm(total=len(valid_loader), ncols=80)

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            if para.num_gpus > 1:
                outputs = nn.parallel.data_parallel(
                    model,
                    inputs,
                    range(para.num_gpus)
                )
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels, valid_flag=True)
            measure = metrics(outputs.detach(), labels)
            loss_meter.update(loss.detach().item(), inputs.size(0))
            measure_meter.update(measure.detach().item(), inputs.size(0))

            # measure elapsed time
            batchtime_meter.update(time.time() - end)
            end = time.time()
            pbar.update(para.batch_size)

    pbar.close()
    # record info
    logger.register(para.loss + '_valid', epoch, loss_meter.avg)
    logger.register(para.metrics + '_valid', epoch, measure_meter.avg)
    # show info
    logger('[valid] epoch time: {:.2f}s, average batch time: {:.2f}s'.format(end-start, batchtime_meter.avg), timestamp=False)
    logger.report([[para.loss, 'min'], [para.metrics, 'max']], state='valid', epoch=epoch)
# *************************************************************

# *************************************************************
# test for lmdb dataset
def test(para, logger):
    logger('{} image results generating ...'.format(para.dataset), prefix='\n')
    if not para.test_only:
        para.test_checkpoint = join(logger.save_dir, 'model_best.path.tar')
    if para.test_save_dir == None:
        para.test_save_dir = logger.save_dir
    datasetType = para.dataset + '_lmdb'
    modelName = para.model.lower()
    model = Model(para).model.cuda()
    checkpointPath = para.test_checkpoint
    checkpoint = torch.load(checkpointPath, map_location=lambda storage, loc: storage.cuda())
    try:
        model.load_state_dict(checkpoint['state_dict'])
        model = nn.DataParallel(model)
    except:
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
    if para.dataset == 'gopro_ds':
        H, W, C = 540, 960, 3
        lmdbType = 'valid'
    else:
        H, W, C = 720, 1280, 3
        lmdbType = 'test'
    data_test_path = join(para.data_root, datasetType, datasetType[:-4] + lmdbType)
    data_test_gt_path = join(para.data_root, datasetType, datasetType[:-4] + lmdbType + '_gt')
    env_blur = lmdb.open(data_test_path, map_size=int(3e10))
    env_gt = lmdb.open(data_test_gt_path, map_size=int(3e10))
    txn_blur = env_blur.begin()
    txn_gt = env_gt.begin()
    # load dataset info
    data_test_info_path = join(para.data_root, datasetType, datasetType[:-4] + 'info_{}.pkl'.format(lmdbType))
    with open(data_test_info_path, 'rb') as f:
        seqs_info = pickle.load(f)
    for seq_idx in range(seqs_info['num']):
        # break
        logger('seq {:03d} image results generating ...'.format(seq_idx))
        torch.cuda.empty_cache()
        save_dir = join(para.test_save_dir, datasetType+'_results_test', '{:03d}'.format(seq_idx))
        os.makedirs(save_dir, exist_ok=True)  # create the dir if not exist
        start = 0
        end = para.test_frames
        while (True):
            input_seq = []
            label_seq = []
            for frame_idx in range(start, end):
                code = '%03d_%08d' % (seq_idx, frame_idx)
                code = code.encode()
                img_blur = txn_blur.get(code)
                img_blur = np.frombuffer(img_blur, dtype='uint8')
                img_blur = img_blur.reshape(H, W, C)
                img_gt = txn_gt.get(code)
                img_gt = np.frombuffer(img_gt, dtype='uint8')
                img_gt = img_gt.reshape(H, W, C)
                input_seq.append(img_blur.transpose((2, 0, 1))[np.newaxis, :])
                label_seq.append(img_gt.transpose((2, 0, 1))[np.newaxis, :])
            input_seq = np.concatenate(input_seq)[np.newaxis, :]
            label_seq = np.concatenate(label_seq)[np.newaxis, :]
            model.eval()
            with torch.no_grad():
                input_seq = torch.from_numpy(input_seq).float().cuda()
                label_seq = torch.from_numpy(label_seq).float().cuda()
                # print(seq_idx, datasetType, modelName, input_seq.shape, label_seq.shape)
                output_seq = model(input_seq).clamp(0, 255).squeeze()
            for frame_idx in range(para.past_frames, end - start - para.future_frames):
                img_blur = input_seq.squeeze()[frame_idx].squeeze()
                img_blur = img_blur.detach().cpu().numpy().transpose((1, 2, 0))
                img_gt = label_seq.squeeze()[frame_idx].squeeze()
                img_gt = img_gt.detach().cpu().numpy().transpose((1, 2, 0))
                img_deblur = output_seq[frame_idx - para.past_frames]
                img_deblur = img_deblur.detach().cpu().numpy().transpose((1, 2, 0))
                cv2.imwrite(join(save_dir, '{:08d}_blur.png'.format(frame_idx + start)), img_blur)
                cv2.imwrite(join(save_dir, '{:08d}_gt.png'.format(frame_idx + start)), img_gt)
                cv2.imwrite(join(save_dir, '{:08d}_{}.png'.format(frame_idx + start, modelName)), img_deblur)
            if end == seqs_info[seq_idx]['length']:
                break
            else:
                start = end - para.future_frames - para.past_frames
                end = start + para.test_frames
                if end > seqs_info[seq_idx]['length']:
                    end = seqs_info[seq_idx]['length']
                    start = end - para.test_frames
    if para.video:
        logger('{} video results generating ...'.format(para.dataset), prefix='\n')
        marks = ['Blur', modelName, 'GT']
        path = join(para.test_save_dir, datasetType+'_results_test')
        for i in range(seqs_info['num']):
            logger('seq {:03d} video result generating ...'.format(i))
            pic2video(path, (3 * W, 1 * H), seq_num=i, frames=seqs_info[i]['length'], save_dir=path, marks=marks, fp=para.past_frames, ff=para.future_frames)

# generate video
def pic2video(path, size, seq_num, frames, save_dir, marks, fp, ff, fps=10):
    file_path = join(save_dir, '{:03d}.avi'.format(seq_num))
    os.makedirs(dirname(save_dir), exist_ok=True)  # create the dir if not exist
    path = join(path, '{:03d}'.format(seq_num))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter(file_path, fourcc, fps, size)
    # print(frames)
    for i in range(fp, frames - ff):
        imgs = []
        for j in range(len(marks)):
            img_path = join(path, '{:08d}_{}.png'.format(i, marks[j].lower()))
            img = cv2.imread(img_path)
            img = cv2.putText(img, marks[j], (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            imgs.append(img)
        frame = np.concatenate(imgs, axis=1)
        video.write(frame)
    video.release()