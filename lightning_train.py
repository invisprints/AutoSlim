import importlib
import os
import random
import math

import torch
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pytorch_lightning as pl

from utils.model_profiling import model_profiling
from utils.transforms import Lighting
from utils.distributed import init_dist, master_only, is_master
from utils.distributed import get_rank, get_world_size
from utils.distributed import master_only_print as print
from utils.loss_ops import CrossEntropyLossSoft, CrossEntropyLossSmooth
from utils.config import FLAGS


def data_transforms():
    """get transform of dataset"""
    if FLAGS.data_transforms in [
            'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']:
        if FLAGS.data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_basic':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.25
            jitter_param = 0.4
            lighting_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    elif FLAGS.data_transforms in ['cifar10_rancrop_flip']:
        input('cifar10_rancrop_flip')
        if FLAGS.data_transforms == 'cifar10_rancrop_flip':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms

    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return train_transforms, val_transforms, test_transforms

def dataset(train_transforms, val_transforms, test_transforms):
    """get dataset for classification"""
    if FLAGS.dataset == 'imagenet1k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    elif FLAGS.dataset == 'imagenet1k_val50k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
            if hasattr(FLAGS, 'random_seed'):
                seed = FLAGS.random_seed
            else:
                seed = 0
            random.seed(seed)
            val_size = 50000
            random.shuffle(train_set.samples)
            if getattr(FLAGS, 'autoslim', False):
                train_set.samples = train_set.samples[:val_size]
            else:
                train_set.samples = train_set.samples[val_size:]
        else:
            train_set = None
            val_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'val'),
                transform=val_transforms)
            test_set = None
    elif FLAGS.dataset == 'cifar10':
        if not FLAGS.test_only:
            train_set = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=val_transforms)
        test_set = None

    else:
        try:
            dataset_lib = importlib.import_module(FLAGS.dataset)
            return dataset_lib.dataset(
                train_transforms, val_transforms, test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset_dir))
    return train_set, val_set, test_set

def data_loader(train_set, val_set, test_set):
    """get data loader"""
    train_loader = None
    val_loader = None
    test_loader = None
    # infer batch size
    if getattr(FLAGS, 'batch_size', False):
        if getattr(FLAGS, 'batch_size_per_gpu', False):
            assert FLAGS.batch_size == (
                FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job)
        else:
            assert FLAGS.batch_size % FLAGS.num_gpus_per_job == 0
            FLAGS.batch_size_per_gpu = (
                FLAGS.batch_size // FLAGS.num_gpus_per_job)
    elif getattr(FLAGS, 'batch_size_per_gpu', False):
        FLAGS.batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job
    else:
        raise ValueError('batch size (per gpu) is not defined')
    # batch_size = int(FLAGS.batch_size/get_world_size())
    batch_size = int(FLAGS.batch_size)
    if FLAGS.data_loader == 'imagenet1k_basic':
        # if getattr(FLAGS, 'distributed', False):
        #     if FLAGS.test_only:
        #         train_sampler = None
        #     else:
        #         train_sampler = DistributedSampler(train_set)
        #     val_sampler = DistributedSampler(val_set)
        # else:
        #     train_sampler = None
        #     val_sampler = None
        if not FLAGS.test_only:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                # shuffle=(train_sampler is None),
                shuffle=True,
                # sampler=train_sampler,
                pin_memory=True,
                num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            # sampler=val_sampler,
            pin_memory=True,
            num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        test_loader = val_loader
    elif FLAGS.data_loader == 'cifar10_loader':

        if getattr(FLAGS, 'distributed', False):
            if FLAGS.test_only:
                train_sampler = None
            else:
                train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(val_set)
        else:
            train_sampler = None
            val_sampler = None
        if not FLAGS.test_only:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                pin_memory=True,
                num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        test_loader = val_loader

    else:
        try:
            data_loader_lib = importlib.import_module(FLAGS.data_loader)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    FLAGS.data_loader))
    if train_loader is not None:
        FLAGS.data_size_train = len(train_loader.dataset)
    if val_loader is not None:
        FLAGS.data_size_val = len(val_loader.dataset)
    if test_loader is not None:
        FLAGS.data_size_test = len(test_loader.dataset)
    return train_loader, val_loader, test_loader

train_transforms, val_transforms, test_transforms = data_transforms()
train_set, val_set, test_set = dataset(
        train_transforms, val_transforms, test_transforms)
train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set)

class slimmNetWork(pl.LightningModule):

    def __init__(self, flag):
        super().__init__()
        model_lib = importlib.import_module(FLAGS.model)
        self.model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
        self.flag = flag
        self.best_val = 1.

        if getattr(FLAGS, 'label_smoothing', 0):
            self.criterion = CrossEntropyLossSmooth(reduction='none')
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        if getattr(FLAGS, 'inplace_distill', False):
            self.soft_criterion = CrossEntropyLossSoft(reduction='none')
        else:
            self.soft_criterion = None

        if getattr(FLAGS, 'slimmable_training', False):
            self.max_width = max(FLAGS.width_mult_list)
            self.min_width = min(FLAGS.width_mult_list)

    def forward(self, x):
        # called with self(x)
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        # change learning rate if necessary
        # slimmable model (s-nets)
        logs = {}
        x, y = batch
        if getattr(FLAGS, 'slimmable_training', False):
            for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                self.model.apply(
                    lambda m: setattr(m, 'width_mult', width_mult))
                y_hat = self(x)
                loss = torch.mean(self.criterion(y_hat, y))
                loss.backward()
                logs[str(width_mult)] = loss
                if width_mult == self.max_width:
                    train_loss = loss

        return {'loss': train_loss, 'log': logs}

    def backward(self, trainer, loss, optimizer, optimizer_idx: int) -> None:
        pass


    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        logs = {}
        x, y = batch
        if getattr(FLAGS, 'slimmable_training', False):
            for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                self.model.apply(
                    lambda m: setattr(m, 'width_mult', width_mult))
                y_hat = self(x)
                loss = torch.mean(self.criterion(y_hat, y))
                logs[str(width_mult)] = loss
        return logs

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = {}
        for width_mult in sorted(FLAGS.width_mult_list):
            avg_loss[str(width_mult)] = torch.stack([x[str(width_mult)] for x in outputs]).mean()
        return {'val_loss': avg_loss, 'log': avg_loss}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        logs = {}
        x, y = batch
        if getattr(FLAGS, 'slimmable_training', False):
            for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                self.model.apply(
                    lambda m: setattr(m, 'width_mult', width_mult))
                y_hat = self(x)
                loss = torch.mean(self.criterion(y_hat, y))
                logs[str(width_mult)] = loss
        return logs


    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = {}
        for width_mult in sorted(FLAGS.width_mult_list):
            avg_loss[str(width_mult)] = torch.stack([x[str(width_mult)] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss, 'log': avg_loss, 'progress_bar': avg_loss}


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBpFGS it is automatically supported, no need for closure function)
        if FLAGS.optimizer == 'sgd':
            # all depthwise convolution (N, 1, x, x) has no weight decay
            # weight decay only on normal conv and fc
            model_params = []
            for params in self.model.parameters():
                ps = list(params.size())
                if len(ps) == 4 and ps[1] != 1:
                    weight_decay = FLAGS.weight_decay
                elif len(ps) == 2:
                    weight_decay = FLAGS.weight_decay
                else:
                    weight_decay = 0
                item = {'params': params, 'weight_decay': weight_decay,
                        'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                        'nesterov': FLAGS.nesterov}
                model_params.append(item)
            optimizer = torch.optim.SGD(model_params)
        else:
            try:
                optimizer_lib = importlib.import_module(FLAGS.optimizer)
                optimizer = optimizer_lib.get_optimizer(self.model)
            except ImportError:
                raise NotImplementedError(
                    'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))

        # lr scheduler
        warmup_epochs = getattr(FLAGS, 'lr_warmup_epochs', 0)
        if FLAGS.lr_scheduler == 'multistep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=FLAGS.multistep_lr_milestones,
                gamma=FLAGS.multistep_lr_gamma)
        elif FLAGS.lr_scheduler == 'exp_decaying':
            lr_dict = {}
            for i in range(FLAGS.num_epochs):
                if i == 0:
                    lr_dict[i] = 1
                else:
                    lr_dict[i] = lr_dict[i-1] * FLAGS.exp_decaying_lr_gamma
            lr_lambda = lambda epoch: lr_dict[epoch]
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda)
        elif FLAGS.lr_scheduler == 'linear_decaying':
            num_epochs = FLAGS.num_epochs - warmup_epochs
            lr_dict = {}
            for i in range(FLAGS.num_epochs):
                lr_dict[i] = 1. - (i - warmup_epochs) / num_epochs
            lr_lambda = lambda epoch: lr_dict[epoch]
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda)
        elif FLAGS.lr_scheduler == 'cosine_decaying':
            num_epochs = FLAGS.num_epochs - warmup_epochs
            lr_dict = {}
            for i in range(FLAGS.num_epochs):
                lr_dict[i] = (
                                     1. + math.cos(
                                 math.pi * (i - warmup_epochs) / num_epochs)) / 2.
            lr_lambda = lambda epoch: lr_dict[epoch]
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda)
        else:
            try:
                lr_scheduler_lib = importlib.import_module(FLAGS.lr_scheduler)
                lr_scheduler = lr_scheduler_lib.get_lr_scheduler(optimizer)
            except ImportError:
                raise NotImplementedError(
                    'Learning rate scheduler {} is not yet implemented.'.format(
                            FLAGS.lr_scheduler))
        # self.last_epoch = lr_scheduler.last_epoch
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        # REQUIRED
        return train_loader

    def val_dataloader(self):
        # OPTIONAL
        return val_loader

    def test_dataloader(self):
        # OPTIONAL
        return test_loader

def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = getattr(FLAGS, 'random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@master_only
def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda: {}.'.format(use_cuda))
    if getattr(FLAGS, 'autoslim', False):
        flops, params = model_profiling(
            model, FLAGS.image_size, FLAGS.image_size, use_cuda=use_cuda,
            verbose=getattr(FLAGS, 'profiling_verbose', False))
    elif getattr(FLAGS, 'slimmable_training', False):
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            model.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            print('Model profiling with width mult {}x:'.format(width_mult))
            flops, params = model_profiling(
                model, FLAGS.image_size, FLAGS.image_size, use_cuda=use_cuda,
                verbose=getattr(FLAGS, 'profiling_verbose', False))
    else:
        flops, params = model_profiling(
            model, FLAGS.image_size, FLAGS.image_size, use_cuda=use_cuda,
            verbose=getattr(FLAGS, 'profiling_verbose', True))
    return flops, params


def main():
    """train and eval model"""
    print(FLAGS)
    slim_model = slimmNetWork(FLAGS)

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(gpus=FLAGS.num_gpus_per_job,
                         benchmark=True,
                         # fast_dev_run=True,
                         max_epochs=FLAGS.num_epochs,
                         profiler=True)
    trainer.fit(slim_model)


if __name__ == "__main__":
    main()
