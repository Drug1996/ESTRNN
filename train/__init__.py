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
        # # record model profile: computation cost & # of parameters
        # if not self.para.no_profile:
        #     self.profile(logger)
        #     del logger
        # main work of trainer
        if self.para.trainer_mode == 'ddp':
            gpus = self.para.num_gpus
            mp.spawn(dist_proc, nprocs=gpus, args=(self.para,))
        elif self.para.trainer_mode == 'dp':
            proc(self.para)

    def profile(self, logger):
        model = Model(self.para)
        flops, params = model.profile()
        del model
        frames = self.para.frames
        logger('generating profile of {} model ...'.format(self.para.model), prefix='\n')
        logger('[profile] computation cost: {:.2f} GMACs, parameters: {:.2f} M'.format(
            flops / (frames) / 10 ** 9, params / 10 ** 6), timestamp=False)


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