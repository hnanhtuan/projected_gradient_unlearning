import os
import yaml
import json
import logging
from pathlib import Path
import shutil
from easydict import EasyDict as edict

import numpy as np
import torch
import copy
import time
from torch import nn
import random
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.general_utils import *
from utils.unlearn_utils import *
import datasets
import backbone


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

# loading config file
cfg_name = sys.argv[1]
stream = open(cfg_name, "r")
try:
    cfg = edict(yaml.safe_load(stream))
    load_config(cfg)
    cfg.work_dir = os.path.dirname(cfg_name)
    os.makedirs(cfg.work_dir, exist_ok=True)
    # create file handler
    fileHandler = logging.FileHandler(f'{cfg.work_dir}/debug.log')
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    tmp = yaml.safe_load(json.dumps(cfg, sort_keys=False, indent=2))
    logger.info(yaml.dump(tmp, sort_keys=False))
    yaml.dump(tmp, open(f'{cfg.work_dir}/config.yaml', 'w'), sort_keys=False)
except yaml.YAMLError as exc:
    print(exc)
stream.close()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not hasattr(cfg, 'learned_config'):
    cfg.learned_config = f'exp/learned/{cfg.model}_{cfg.dataset}/full_data/config.yaml'
learn_cfg = edict(yaml.safe_load(open(cfg.learned_config)))

transform_train, transform_test = get_std_transforms(learn_cfg.params.get('image_size', 32))
valset = eval(learn_cfg.dataset.val.name)(root=learn_cfg.dataset.val.root, transform=transform_test, 
                                            **learn_cfg.dataset.val.params)
val_loader = DataLoader(valset, shuffle=False, num_workers=6, batch_size=250)

learn_cfg.dataset.train.params.forget_range = None
full_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                      **learn_cfg.dataset.train.params)
full_train_noaug_loader = DataLoader(full_train_noaug, shuffle=False, num_workers=6, batch_size=125)

full_train_aug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_train, 
                                                    **learn_cfg.dataset.train.params)
full_train_aug_loader = DataLoader(full_train_aug, shuffle=False, num_workers=6, batch_size=125)

# Model trained on full-data
logger.info('-'*20 + ' Original model ' + '-'*20)
learned_model_dir = os.path.dirname(cfg.learned_config)
ckp = torch.load(f'{learned_model_dir}/checkpoints/ckp.pth')
logger.info(f'Load checkpoint: {learned_model_dir}/checkpoints/ckp.pth')
full_data_model = eval(learn_cfg.net.backbone)(**learn_cfg.net.params)
full_data_model.load_state_dict(ckp['model'])
full_data_model.eval()
full_data_model.cuda();
print(full_data_model)

evaluate(full_data_model, val_loader, 'Val', displayer=logger.info)


full_svd_file = f'exp/svd/{cfg.model}_{cfg.dataset}/svd_full_data.pt'
os.makedirs(f'exp/svd/{cfg.model}_{cfg.dataset}/', exist_ok=True)
if not os.path.isfile(full_svd_file):
    full_svd = compute_svd(full_data_model, full_train_aug_loader, epochs=1, printer=logger.info)
    torch.save(full_svd, full_svd_file)
else:
    full_svd = torch.load(full_svd_file)

current_model = copy.deepcopy(full_data_model)
current_svd = copy.deepcopy(full_svd)

acc_gap = None

for unlearn_step_idx, unlearn_step in enumerate(cfg.unlearn_steps):
    logger.info('-'*20 + f'{bcolors.BOLD_RED} {unlearn_step.forget_range} {bcolors.RESET}' + '-'*20)
    suffix = f'{unlearn_step.forget_range[0][0]}-{unlearn_step.forget_range[0][1]}_{len(unlearn_step.forget_range)}classes'
    all_suffix = f'0-{unlearn_step.forget_range[0][1]}_{len(unlearn_step.forget_range)}classes'

    print('forget_train_noaug')
    learn_cfg.dataset.train.params.forget_range = unlearn_step.forget_range
    learn_cfg.dataset.train.params.ignore_range = unlearn_step.get('ignore_range', [])
    learn_cfg.dataset.train.params.data_section = 'forget'
    forget_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                            **learn_cfg.dataset.train.params)
    forget_train_noaug_loader = DataLoader(forget_train_noaug, shuffle=False, num_workers=6, batch_size=128)

    forget_train_aug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_train, 
                                                            **learn_cfg.dataset.train.params)
    forget_train_aug_loader = DataLoader(forget_train_aug, shuffle=False, num_workers=6, batch_size=128)

    learn_cfg.dataset.train.params.forget_range = [[0, unlearn_step.forget_range[0][1]]]*len(unlearn_step.forget_range)
    all_forget_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                                **learn_cfg.dataset.train.params)
    all_forget_train_noaug_loader = DataLoader(all_forget_train_noaug, shuffle=False, num_workers=6, batch_size=128)

    print('retain_train_noaug')
    learn_cfg.dataset.train.params.forget_range = unlearn_step.forget_range
    learn_cfg.dataset.train.params.data_section = 'retain'
    retain_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                            **learn_cfg.dataset.train.params)
    retain_train_noaug_loader = DataLoader(retain_train_noaug, shuffle=False, num_workers=6, batch_size=128)

    # Baseline
    logger.info('-'*20 + f'{bcolors.BOLD_RED} Baseline model {suffix} {bcolors.RESET}' + '-'*20)
    ckp_file = f'exp/learned/{cfg.model}_{cfg.dataset}/forget_{all_suffix}/checkpoints/ckp.pth'
    logger.info(ckp_file)
    ckp = torch.load(ckp_file)
    baseline_model = eval(learn_cfg.net.backbone)(**learn_cfg.net.params)
    baseline_model.load_state_dict(ckp['model'])
    baseline_model.eval()
    baseline_model.cuda();

    evaluate(baseline_model, all_forget_train_noaug_loader, 'Forget train', displayer=logger.info)
    evaluate(baseline_model, retain_train_noaug_loader, 'Retain train', displayer=logger.info)
    evaluate(baseline_model, val_loader, 'Val', displayer=logger.info)

    forget_train_retrained_entropies = get_entropy(baseline_model, all_forget_train_noaug_loader)

    # Get svd for retain data
    logger.info('-'*20 + f'{bcolors.BOLD_RED} Compute Retain SVD {bcolors.RESET}' + '-'*20)
    retain_svd_file = f'exp/svd/{cfg.model}_{cfg.dataset}/retain_{suffix}.pt'
    retain_svd = compute_retain_svd(current_svd, current_model, forget_train_noaug_loader, 
                                    epochs=1, printer=logger.info)

    current_svd = retain_svd

    logger.info('-'*20 + f'{bcolors.BOLD_RED} Compute Projection Matrix {bcolors.RESET}' + '-'*20)
    
    P = {}
    for layer in retain_svd: # ,
        if isinstance(cfg.retained_var, float):
            retained_var = cfg.retained_var
        else:
            retained_var = cfg.retained_var.get(layer, cfg.retained_var.default)
        retained_var += cfg.get('retained_var_step', 0)*unlearn_step_idx
        if retained_var == 1.0:
            logger.info(f'Skip layer: {layer}')
            continue
        logger.info(f'Retained var: {retained_var:.04f}')
        k = torch.sum((torch.cumsum(retain_svd[layer]['S'], dim=0) / torch.sum(retain_svd[layer]['S'])) <= retained_var)
        logger.info(f"{k.item()}\t {retain_svd[layer]['S'].shape[0]}\t {100*k.item()/retain_svd[layer]['S'].shape[0]:.06f}")
        
        M = retain_svd[layer]['U'][:, :k]
        P[layer] = torch.mm(M, M.t()).to(device).float()
        logger.info(f' - {bcolors.BOLD_RED}Layer:{bcolors.RESET} {layer} - {P[layer].shape}')

    unlearn_model = copy.deepcopy(current_model)
    optimizer = optim.SGD(unlearn_model.parameters(), lr=0.02, momentum=0, weight_decay=0)
    global_epoch = 0
    start_lr = cfg.start_lr
    end_lr = cfg.end_lr
    num_epochs = int(cfg.get('num_epochs', 100))

    offset = float(cfg.get('offset', 0.1))
    loss1_w = float(cfg.get('loss1_w', 1))
    loss2_w = float(cfg.get('loss2_w', 1))
    wd = float(cfg.get('wd', 0))
    num_bins = cfg.get('num_bins', 100)

    alpha = np.exp(np.log(end_lr/start_lr) / num_epochs)
    base_lr = start_lr

    retain_train_res, forget_train_res, val_res = [], [], []

    exp_dir = f'{cfg.work_dir}/results/{suffix}'
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f'{exp_dir}/ckp/', exist_ok=True)
    os.makedirs(f'{exp_dir}/entropy_retrain_unlearn/', exist_ok=True)

    ce_losses = AverageMeter()
    losses1, losses2 = AverageMeter(), AverageMeter()
    prev_model = None
    prev_acc_gap = None

    name_mapping = {}
    for epoch in range(num_epochs):
        # Unlearn Training
        unlearn_model.train()
        unlearn_model.apply(freeze_norm_stats)
        for batch_idx, (images, labels) in enumerate(forget_train_noaug_loader):
            if epoch == 0:
                break
            labels = labels.to(device)
            images = images.to(device)

            optimizer.zero_grad()
            outputs = unlearn_model(images)
            
            # loss = loss_function(outputs, labels)
            logits = F.softmax(outputs, dim=1)
            loss1, loss2 = 0, 0
            for j in range(images.shape[0]):
                logit = logits[j][labels[j]]
                loss1 += -torch.log(1 - ((logit) - offset))
                
            loss2 = -torch.sum(-logits*torch.log(logits + 1e-15), dim=1).mean()
            loss1 /= images.shape[0]
            loss = loss1_w*loss1 + loss2_w*loss2

            losses1.update(loss1.item(), images.shape[0])
            losses2.update(loss2.item(), images.shape[0])
            ce_losses.update(loss.item(), images.shape[0])
            loss.backward()

            with torch.no_grad():
                for name, param in unlearn_model.named_parameters():
                    if name not in name_mapping:
                        P_name = name
                        for i in range(20):
                            P_name = P_name.replace(f'.{i}', f'[{i}]')
                        P_name = P_name.replace('.weight', '')
                        name_mapping[name] = P_name
                    else:
                        P_name = name_mapping[name]

                    if P_name not in P:
                        param.grad.data.fill_(0)
                        continue

                    # print(name, P_name)
                    sz = param.grad.data.shape[0]
                    reg_grad = param.grad.data.add(param.data, alpha=wd)
                    reg_grad = reg_grad - torch.mm(reg_grad.view(sz,-1), P[P_name]).view(param.size())
    
                    lr = base_lr
                    param.data -= lr * reg_grad
        
        # Evaluate
        res = evaluate(unlearn_model, retain_train_noaug_loader, '[Unlearned] - Retain train', displayer=logger.info)
        retain_train_res.append(res)

        loss, acc, forget_train_unlearned_entropies = \
                evaluate_entropy(unlearn_model, all_forget_train_noaug_loader, '[Unlearned] - Forget train', displayer=logger.info)
        forget_train_res.append((loss, acc))
        
        loss, acc = evaluate(unlearn_model, val_loader, '[Unlearned] - Val', displayer=logger.info)
        val_res.append((loss, acc))

        bins = np.linspace(-20, 1, 100)
        plot_histograms([forget_train_retrained_entropies, forget_train_unlearned_entropies],
                        ['Forget Train Retrained', 'Forget Train Unlearned'], num_bins=num_bins,
                        plot_name=f'{exp_dir}/entropy_retrain_unlearn/{global_epoch:03d}.jpg')
        
        logger.info(f'Epoch: {epoch:02d} - Loss: {ce_losses.avg:.04f} ({losses1.avg:.04f}+{losses2.avg:.04f}) - LR: {base_lr:.06f}')
        
        checkpoint = {'epoch': global_epoch, 'lr': base_lr}
        if cfg.get('save_model', 0) != 0:
            checkpoint.update({'model': unlearn_model.state_dict()})
        torch.save(checkpoint, f'{exp_dir}/ckp/ckp_{global_epoch:03d}.pt')
        logger.info('----------------------------------------------------------')
        ce_losses.reset()
        losses1.reset()
        losses2.reset()
        base_lr *= alpha
        global_epoch += 1

        plt.plot(np.arange(len(retain_train_res)), [x[1] for x in retain_train_res], label='Retain Train')
        plt.plot(np.arange(len(val_res)),          [x[1] for x in val_res], label='Val')
        plt.plot(np.arange(len(forget_train_res)), [x[1] for x in forget_train_res], label='Forget Train')
        plt.grid()
        plt.legend()
        plt.savefig(f'{exp_dir}/accuracies.jpg')
        plt.close()

        plt.plot(np.arange(len(retain_train_res)), [x[0] for x in retain_train_res], label='Retain Train')
        plt.plot(np.arange(len(val_res)), [x[0] for x in val_res], label='Val')
        plt.plot(np.arange(len(forget_train_res)), [x[0] for x in forget_train_res], label='Forget Train')
        plt.grid()
        plt.legend()
        plt.savefig(f'{exp_dir}/loss.jpg')
        plt.close()

        # if the accuracy of the val set is higher than the accuracy of the forget training set;
        # stop unlearning to avoid over-unlearn
        acc_gap = val_res[-1][1] - forget_train_res[-1][1]
        if acc_gap > 0:
            if prev_acc_gap is None: 
                logger.info('-------------------- Unlearning too fast --------------------')
                exit() # unlearning too fast
                
            if abs(acc_gap) < abs(prev_acc_gap):
                current_model = copy.deepcopy(unlearn_model)
                checkpoint.update({'model': current_model.state_dict()})
            else:
                current_model = copy.deepcopy(prev_model)
                checkpoint.update({'model': current_model.state_dict()})

            torch.save(checkpoint, f'{exp_dir}/ckp/ckp_{global_epoch-1:03d}.pt')
            torch.save(checkpoint, f'{exp_dir}/ckp/ckp.pt')
            break

        prev_model = copy.deepcopy(unlearn_model)
        prev_acc_gap = acc_gap
            
        
    exp_res = {'retain_train_res': retain_train_res,
               'forget_train_res': forget_train_res,
               'val_res': val_res,
              }

    np.save(f'{exp_dir}/exp_res.npy', exp_res)
    logger.info(f'{bcolors.BOLD}Epoch:{bcolors.RESET} {epoch} - {bcolors.BOLD}Accuracy gap:{bcolors.RESET} {acc_gap}')
    if epoch == num_epochs -1 and acc_gap < -0.5:
        checkpoint.update({'model': current_model.state_dict()})
        torch.save(checkpoint, f'{exp_dir}/ckp/ckp.pt')
        break

checkpoint.update({'model': current_model.state_dict()})
torch.save(checkpoint, f'{exp_dir}/ckp/ckp.pt')

