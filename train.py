import json
import yaml
import copy
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from pprint import pprint
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

from torch.utils.data import DataLoader
import backbone
import datasets

from utils.general_utils import *

# create auxiliary variables
loggerName = Path(__file__).stem

# create logging formatter
logFormatter = logging.Formatter(fmt='%(asctime)s %(name)15s (%(lineno)03d) :: [%(levelname)8s] :: %(message)s')

# create logger
logger = logging.getLogger(loggerName)
logger.setLevel(logging.DEBUG)

# create console handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


class Network:
    def __init__(self, cfg):
        self.cfg = cfg

        transform_train, transform_test = get_std_transforms(self.cfg.params.get('image_size', 32))
        trainset = eval(cfg.dataset.train.name)(root=cfg.dataset.train.root, transform=transform_train, **cfg.dataset.train.params)
        testset = eval(cfg.dataset.test.name)(root=cfg.dataset.test.root, transform=transform_test, **cfg.dataset.test.params)

        self.train_loader = DataLoader(trainset, shuffle=True, num_workers=cfg.params.num_workers, batch_size=cfg.params.batch_size)
        self.test_loader = DataLoader(testset, shuffle=False, num_workers=cfg.params.num_workers, batch_size=cfg.params.batch_size)

        self.model = eval(cfg.net.backbone)(**cfg.net.params)
        print(self.model)

        self.optimizer = eval(cfg.optimizer.method)(self.model.parameters(), **cfg.optimizer.params)
        self.scheduler = eval(cfg.scheduler.method)(self.optimizer, **cfg.scheduler.params)  # learning rate decay
        self.loss_function = nn.CrossEntropyLoss()

        self.board_writer = SummaryWriter(log_dir=os.path.join(cfg.paths.tensorboard, cfg.net.backbone))
        os.makedirs(cfg.paths.checkpoint, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_epoch(self, epoch=0):
        self.model.train()
        self.model.to(self.device)
        if self.cfg.params.get('freeze_k', 0) > 0:
            i = 0
            new_model = eval(self.cfg.net.backbone)(**self.cfg.net.params)
            for md, new_md in zip(self.model.modules(), new_model.modules()):
                if islayer(md):
                    if i < self.cfg.params.freeze_k:
                        for param in md.parameters():
                            param.requires_grad = False
                        md.eval()
                    i += 1

        ce_losses = AverageMeter()
        accuracies = AverageMeter()
        num_batches = 0
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            labels = labels.to(self.device)
            images = images.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)
            ce_losses.update(loss.item(), images.shape[0])
            acc = accuracy(outputs, labels)
            accuracies.update(acc[0].item(), images.shape[0])
            loss.backward()
            self.optimizer.step()

            if batch_idx > 0 and (batch_idx % self.cfg.params.log_interval == 0 or batch_idx == len(self.train_loader) - 1):
                self.board_writer.add_scalar('Train/Loss', loss.item())
                self.board_writer.add_scalar('Train/Acc', accuracies.avg)

                logger.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAcc: {:0.4f}\tLR: {:0.6f}'.format(
                    ce_losses.avg, accuracies.avg,
                    self.optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_idx * self.cfg.params.batch_size + len(images),
                    total_samples=len(self.train_loader.dataset)))

    def test_epoch(self, epoch=0):
        self.model.eval()
        self.model.to(self.device)
        ce_losses = AverageMeter()
        accuracies = AverageMeter()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                labels = labels.to(self.device)
                images = images.to(self.device)

                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                acc = accuracy(outputs, labels)
                ce_losses.update(loss.item(), images.shape[0])
                accuracies.update(acc[0].item(), images.shape[0])

                if batch_idx == len(self.test_loader) - 1:
                    logger.info(f'Testing Epoch:  {epoch} \tLoss: {ce_losses.avg:0.4f} - Accuracy: {accuracies.avg:0.4f}')
                    self.board_writer.add_scalar('Test/Loss', loss.item())
                    self.board_writer.add_scalar('Test/Acc', accuracies.avg)

    def save_checkpoints(self, epoch=None):
        checkpoint_name = f'{self.cfg.paths.checkpoint}/ckp.pth'

        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler
        }
        torch.save(checkpoint, checkpoint_name)
        if epoch is not None:
            checkpoint_name = f'{self.cfg.paths.checkpoint}/ckp_{epoch:03d}.pth'
            torch.save(checkpoint, checkpoint_name)


    def resume(self):
        resume_ckp = self.cfg.params.get('resume', '')
        if os.path.isfile(resume_ckp):
            logger.info(f'Load checkpoints: {resume_ckp}')
            ckp = torch.load(resume_ckp)
            self.model.load_state_dict(ckp['model'])

            if self.cfg.params.get('freeze_k', -1) >= 0:
                i = 0
                new_model = eval(self.cfg.net.backbone)(**self.cfg.net.params)
                for md, new_md in zip(self.model.modules(), new_model.modules()):
                    if islayer(md):
                        if i >= self.cfg.params.freeze_k:
                            if self.cfg.params.get('reset', 0) > 0:
                                for param, new_param in zip(md.parameters(), new_md.parameters()):
                                    param.data = new_param.data.clone()
                                logger.info(f'Reset: {md}')
                        i += 1
        else:
            logger.info('Train from scratch')


def main(cfg):
    net = Network(cfg)
    net.resume()
    net.test_epoch()
    for epoch in range(int(cfg.params.epochs)):
        net.train_epoch(epoch)
        net.test_epoch(epoch)
        net.scheduler.step()
        net.save_checkpoints()
        net.board_writer.add_scalar('Epoch', epoch)
        logger.info(f'------------------------- DONE EPOCH {epoch:03d} -------------------------')
    logger.info(f'DONE')


if __name__ == "__main__":
    cfg_name = sys.argv[1]
    stream = open(cfg_name, "r")
    try:
        cfg = edict(yaml.safe_load(stream))
        parse_args(cfg)
        if cfg.paths is None:
            cfg.paths = {}

        cfg.paths['log_name'] = f'{cfg_name.replace("/config.yaml", "")}' 
        if cfg.paths.log_name[-1] == '/':
            cfg.paths.log_name = cfg.paths.log_name[:-1]
        if os.path.isdir(cfg.paths.log_name) and os.path.isdir(f'{cfg.paths.log_name}/checkpoints') \
                and os.path.isfile(f'{cfg.paths.log_name}/checkpoints/ckp.pth'):
            cfg.paths.log_name += f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        cfg.paths.tensorboard = f'{cfg.paths.log_name}/tensorboard'
        cfg.paths.checkpoint = f'{cfg.paths.log_name}/checkpoints'
        os.makedirs(cfg['paths']['log_name'], exist_ok=True)
        # create file handler
        fileHandler = logging.FileHandler(f'{cfg.paths.log_name}/debug.log')
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        tmp = yaml.safe_load(json.dumps(cfg, sort_keys=False, indent=2))
        logger.info(yaml.dump(tmp, sort_keys=False))
        yaml.dump(tmp, open(f'{cfg.paths.log_name}/config.yaml', 'w'), sort_keys=False)
    except yaml.YAMLError as exc:
        print(exc)
    stream.close()

    main(cfg)
