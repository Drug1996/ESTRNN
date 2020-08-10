from importlib import import_module


class Data(object):
    def __init__(self, para, device_id):
        dataset = para.dataset
        if para.loader_mode == 'lmdb':
            module = import_module('data.' + dataset+'_lmdb')
        elif para.loader_mode == 'torch':
            module = import_module('data.' + dataset)
        self.dataloader_train = module.Dataloader(para, device_id, ds_type='train')
        self.dataloader_valid = module.Dataloader(para, device_id, ds_type='valid')
