import os
import yaml
import json
import copy
import time
import logging
from pathlib import Path
import shutil
from easydict import EasyDict as edict

import numpy as np
import torch
from torch import nn
import random
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
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

transform_train, transform_test = get_std_transforms(learn_cfg.params.get('image_size', cfg.get('image_size')))
valset = eval(learn_cfg.dataset.val.name)(root=learn_cfg.dataset.val.root, transform=transform_test,
                                           **learn_cfg.dataset.val.params)
val_loader = DataLoader(valset, shuffle=False, num_workers=6, batch_size=256)

learn_cfg.dataset.train.params.forget_range = None
full_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                      **learn_cfg.dataset.train.params)
full_train_noaug_loader = DataLoader(full_train_noaug, shuffle=False, num_workers=6, batch_size=128)

full_train_aug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_train, 
                                                    **learn_cfg.dataset.train.params)
full_train_aug_loader = DataLoader(full_train_aug, shuffle=False, num_workers=6, batch_size=128)

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

logger.info('-'*20 + f'{bcolors.BOLD_RED} Compute Full-data SVD {bcolors.RESET}' + '-'*20)
full_svd_file = f'exp/svd/{cfg.model}_{cfg.dataset}/svd_full_data.pt'
os.makedirs(f'exp/svd/{cfg.model}_{cfg.dataset}/', exist_ok=True)
if not os.path.isfile(full_svd_file):
    full_svd = compute_svd(full_data_model, full_train_aug_loader, epochs=1, printer=logger.info)
    torch.save(full_svd, full_svd_file)
else:
    full_svd = torch.load(full_svd_file)

current_model = copy.deepcopy(full_data_model)
current_svd = copy.deepcopy(full_svd)

# try:
for _ in range(1):
    for unlearn_step in cfg.unlearn_steps:
        logger.info('-'*20 + f'{bcolors.BOLD_RED} {unlearn_step.forget_range} {bcolors.RESET}' + '-'*20)
        num_forget_classes = len(unlearn_step.forget_range)
        suffix = f'{num_forget_classes}classes'

        print('forget_train_noaug')
        learn_cfg.dataset.train.params.forget_range = unlearn_step.forget_range
        learn_cfg.dataset.train.params.ignore_range = unlearn_step.get('ignore_range', [])
        learn_cfg.dataset.train.params.data_section = 'forget'
        forget_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                                **learn_cfg.dataset.train.params)
        forget_train_noaug_loader = DataLoader(forget_train_noaug, shuffle=True, num_workers=6, batch_size=100)
        forget_train_aug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_train, 
                                                                **learn_cfg.dataset.train.params)
        forget_train_aug_loader = DataLoader(forget_train_aug, shuffle=True, num_workers=6, batch_size=125)

        print('retain_train_noaug')
        learn_cfg.dataset.train.params.data_section = 'retain'
        retain_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                                **learn_cfg.dataset.train.params)
        retain_train_noaug_loader = DataLoader(retain_train_noaug, shuffle=False, num_workers=6, batch_size=125)

        print('forget_val_noaug')
        learn_cfg.dataset.val.params.data_section = 'forget'
        learn_cfg.dataset.val.params.forget_range = unlearn_step.forget_range
        forget_val_noaug = eval(learn_cfg.dataset.val.name)(root=learn_cfg.dataset.val.root, transform=transform_test,
                                                              **learn_cfg.dataset.val.params)
        forget_val_noaug_loader = DataLoader(forget_val_noaug, shuffle=False, num_workers=6, batch_size=250)

        print('retain_val_noaug')
        learn_cfg.dataset.val.params.data_section = 'retain'
        retain_val_noaug = eval(learn_cfg.dataset.val.name)(root=learn_cfg.dataset.val.root, transform=transform_test, 
                                                             **learn_cfg.dataset.val.params)
        retain_val_noaug_loader = DataLoader(retain_val_noaug, shuffle=False, num_workers=6, batch_size=250)

        # Baseline
        logger.info('-'*20 + f'{bcolors.BOLD_RED} Baseline model {suffix} {bcolors.RESET}' + '-'*20)
        ckp_file = f'exp/learned/{cfg.model}_{cfg.dataset}/forget_{suffix}/checkpoints/ckp.pth'
        logger.info(ckp_file)
        ckp = torch.load(ckp_file)
        baseline_model = eval(learn_cfg.net.backbone)(**learn_cfg.net.params)
        baseline_model.load_state_dict(ckp['model'])
        baseline_model.eval()
        baseline_model.cuda()

        evaluate(baseline_model, forget_train_noaug_loader, 'Forget train', displayer=logger.info)
        evaluate(baseline_model, retain_train_noaug_loader, 'Retain train', displayer=logger.info)
        evaluate(baseline_model, val_loader, 'Val', displayer=logger.info)
        evaluate(baseline_model, forget_val_noaug_loader, 'Forget Val', displayer=logger.info)
        evaluate(baseline_model, retain_val_noaug_loader, 'Retain Val', displayer=logger.info)

        forget_train_retrained_entropies = get_entropy(baseline_model, forget_train_noaug_loader, ignore_first_k=num_forget_classes)

        # Get svd for retain data
        logger.info('-'*20 + f'{bcolors.BOLD_RED} Compute Retain SVD {bcolors.RESET}' + '-'*20)
        retain_svd_file = f'exp/svd/{cfg.model}_{cfg.dataset}/retain_{suffix}.pt'
        retain_svd = compute_retain_svd(current_svd, current_model, forget_train_aug_loader,
                                            epochs=1, printer=logger.info)

        current_svd = retain_svd

        logger.info('-'*20 + f'{bcolors.BOLD_RED} Compute Projection Matrix {bcolors.RESET}' + '-'*20)
        retained_var = cfg.retained_var
        P, Us, Ks = {}, {}, {}
        for layer in retain_svd: # ,
            if isinstance(cfg.retained_var, float):
                retained_var = cfg.retained_var
            else:
                retained_var = cfg.retained_var.get(layer, cfg.retained_var.default)

            if retained_var == 1.0:
                logger.info(f'Skip layer: {layer}')
                continue

            k = torch.sum((torch.cumsum(retain_svd[layer]['S'], dim=0) / torch.sum(retain_svd[layer]['S'])) <= retained_var).item() + 1
            min_dim_pct = cfg.get('min_dim_pct', -1)
            if min_dim_pct > 0:
                min_k = int(min_dim_pct*retain_svd[layer]['S'].shape[0] + 1)
                logger.info(f'{k}, {min_k}')
                k = max(k, min_k)
            
            M = retain_svd[layer]['U'][:, :k]
            Us[layer] = retain_svd[layer]['U'].to(device).float()
            Ks[layer] = k
            P[layer] = torch.mm(M, M.t()).to(device).float()
            logger.info(f' - {bcolors.BOLD_RED}Layer:{bcolors.RESET} {layer} - {P[layer].shape}')
            logger.info(f"{k}\t {retain_svd[layer]['S'].shape[0]}\t {100*k/retain_svd[layer]['S'].shape[0]:.06f}\t {retained_var}")

        unlearn_model = copy.deepcopy(current_model)
        optimizer = optim.SGD(unlearn_model.parameters(), lr=0.02)
        global_epoch = 0
        start_lr = cfg.start_lr
        end_lr = cfg.end_lr
        num_epochs = int(cfg.get('num_epochs', 100))

        offset = float(cfg.get('offset', 0.1))
        loss1_w = float(cfg.get('loss1_w', 1))
        loss2_w = float(cfg.get('loss2_w', 1))
        wd = float(cfg.get('wd', 0))

        alpha = np.exp(np.log(end_lr/start_lr) / num_epochs)
        base_lr = start_lr

        retain_train_res, retain_val_res = [], []
        val_res, forget_train_res, forget_val_res = [], [], []
        num_bins = cfg.get('num_bins', 100)

        exp_dir = f'{cfg.work_dir}/results/{suffix}'
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f'{exp_dir}/ckp/', exist_ok=True)
        os.makedirs(f'{exp_dir}/entropy_retrain_unlearn/', exist_ok=True)

        ce_losses = AverageMeter()
        losses1, losses2 = AverageMeter(), AverageMeter()

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

                        sz = param.grad.data.shape[0]
                        reg_grad = param.grad.data.add(param.data, alpha=wd)
                        reg_grad = reg_grad - torch.mm(reg_grad.view(sz,-1), P[P_name]).view(param.size())
        
                        lr = base_lr
                        param.data -= lr * reg_grad 

            # Evaluate
            res = evaluate(unlearn_model, retain_train_noaug_loader, '[Unlearned] - Retain train', displayer=logger.info)
            retain_train_res.append(res)

            loss, acc, forget_train_unlearned_entropies = evaluate_entropy(unlearn_model, forget_train_noaug_loader, 
                                    '[Unlearned] - Forget train', displayer=logger.info, ignore_topk=num_forget_classes)
            forget_train_res.append((loss, acc))
            
            loss, acc, forget_val_unlearned_entropies = evaluate_entropy(unlearn_model, val_loader, '[Unlearned] - Val', 
                                                                displayer=logger.info, ignore_topk=num_forget_classes)
            val_res.append((loss, acc))

            res = evaluate(unlearn_model, retain_val_noaug_loader, '[Unlearned] - Retain Val', displayer=logger.info)
            retain_val_res.append(res)

            res = evaluate(unlearn_model, forget_val_noaug_loader, '[Unlearned] - Forget Val', displayer=logger.info)
            forget_val_res.append(res)

            entropy_range=-5
            bins = np.linspace(entropy_range, 1, num_bins)
            plot_histograms([forget_train_retrained_entropies, forget_train_unlearned_entropies],
                            ['Forget Train Retrained', 'Forget Train Unlearned'],
                            num_bins=num_bins, x_range=[entropy_range, 2],
                            plot_name=f'{exp_dir}/entropy_retrain_unlearn/{global_epoch:03d}.jpg')
            
            logger.info(f'Epoch: {epoch:02d} - Loss: {ce_losses.avg:.04f} ({losses1.avg:.04f}+{losses2.avg:.04f})' \
                        f' - LR: {base_lr:.06f}')
            
            checkpoint = {'epoch': global_epoch, 'lr': base_lr}
            checkpoint.update({'model': unlearn_model.state_dict()})
            if cfg.get('save_model', 0) != 0:
                torch.save(checkpoint, f'{exp_dir}/ckp/ckp_{global_epoch:03d}.pt')
            torch.save(checkpoint, f'{exp_dir}/ckp/ckp.pt')
            logger.info('----------------------------------------------------------')
            ce_losses.reset()
            losses1.reset()
            losses2.reset()
            base_lr *= alpha
            global_epoch += 1

            plt.plot(np.arange(len(retain_train_res)), [x[1] for x in retain_train_res], label='Retain Train')
            plt.plot(np.arange(len(retain_train_res)), [x[1] for x in forget_train_res], label='Forget Train')
            plt.plot(np.arange(len(retain_train_res)), [x[1] for x in val_res], label='Val')
            plt.plot(np.arange(len(retain_train_res)), [x[1] for x in retain_val_res], label='Retain Val')
            plt.plot(np.arange(len(retain_train_res)), [x[1] for x in forget_val_res], label='Forget Val')
            plt.grid()
            plt.legend()
            plt.savefig(f'{exp_dir}/accuracies.jpg')
            plt.close()

            plt.plot(np.arange(len(retain_train_res)), [x[0] for x in retain_train_res], label='Retain Train')
            plt.plot(np.arange(len(retain_train_res)), [x[0] for x in forget_train_res], label='Forget Train')
            plt.plot(np.arange(len(retain_train_res)), [x[0] for x in val_res], label='Val')
            plt.plot(np.arange(len(retain_train_res)), [x[0] for x in retain_val_res], label='Retain Val')
            plt.plot(np.arange(len(retain_train_res)), [x[0] for x in forget_val_res], label='Forget Val')
            plt.grid()
            plt.legend()
            plt.savefig(f'{exp_dir}/loss.jpg')
            plt.close()

            exp_res = {'retain_train_res': retain_train_res,
                       'retain_val_res': retain_val_res,
                       'val_res': val_res,
                       'forget_train_res': forget_train_res,
                       'forget_val_res': forget_val_res,
                    }

            np.save(f'{exp_dir}/exp_res.npy', exp_res)

            # if the retain train accuracy drop more than the threshold, early stop the training
            early_stop_thres = cfg.get('early_stop_thres', 5.0)
            if (retain_train_res[0][1] - retain_train_res[-1][1]) > early_stop_thres:
                logger.info('Early stop')
                break
            
# except Exception as e:
#     print(e)
#     checkpoint.update({'model': unlearn_model.state_dict()})
#     torch.save(checkpoint, f'{exp_dir}/ckp/ckp.pt')