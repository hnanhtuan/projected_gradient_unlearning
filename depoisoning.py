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
import scipy
import time
from torch import nn
import random
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils.general_utils import *
from utils.unlearn_utils import *
from utils.incremental_SVD import *
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
if os.path.isdir(f'{cfg.work_dir}/tensorboard/'):
    shutil.rmtree(f'{cfg.work_dir}/tensorboard/')
board_writer = SummaryWriter(log_dir=f'{cfg.work_dir}/tensorboard/')


cfg.learned_config = f'exp/learned/{cfg.model}_{cfg.dataset}/poison_{cfg.num_poison}/full/config.yaml'
learn_cfg = edict(yaml.safe_load(open(cfg.learned_config)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
testset = eval(learn_cfg.dataset.test.name)(root=learn_cfg.dataset.test.root, transform=transform_test, **learn_cfg.dataset.test.params)
test_loader = DataLoader(testset, shuffle=False, num_workers=6, batch_size=256)

learn_cfg.dataset.train.params.data_section = 'clean'
clean_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                      **learn_cfg.dataset.train.params)
clean_train_noaug_loader = DataLoader(clean_train_noaug, shuffle=False, num_workers=6, batch_size=128)

clean_train_aug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_train, 
                                                    **learn_cfg.dataset.train.params)
clean_train_aug_loader = DataLoader(clean_train_aug, shuffle=False, num_workers=6, batch_size=128)

learn_cfg.dataset.train.params.data_section = 'poison'
poison_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                      **learn_cfg.dataset.train.params)
poison_train_noaug_loader = DataLoader(poison_train_noaug, shuffle=False, num_workers=6, batch_size=128)
poison_train_aug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_train, 
                                                      **learn_cfg.dataset.train.params)
poison_train_aug_loader = DataLoader(poison_train_aug, shuffle=False, num_workers=6, batch_size=128)

learn_cfg.dataset.train.params.data_section = 'poison_clean'
poison_clean_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
                                                      **learn_cfg.dataset.train.params)
poison_clean_train_noaug_loader = DataLoader(poison_clean_train_noaug, shuffle=False, num_workers=6, batch_size=128)

# Poisoned model
logger.info('-'*20 + ' Poisoned model ' + '-'*20)
learned_model_dir = os.path.dirname(cfg.learned_config)
ckp = torch.load(f'{learned_model_dir}/checkpoints/ckp.pth')
poisoned_model = eval(learn_cfg.net.backbone)(**learn_cfg.net.params)
poisoned_model.load_state_dict(ckp['model'])
poisoned_model.eval()
poisoned_model.cuda();
print(poisoned_model)

evaluate(poisoned_model, clean_train_noaug_loader, 'Clean train', displayer=logger.info)
evaluate(poisoned_model, poison_train_noaug_loader, 'Poison train', displayer=logger.info)
evaluate(poisoned_model, poison_clean_train_noaug_loader, 'Poison-Clean train', displayer=logger.info)
evaluate(poisoned_model, test_loader, 'Test', displayer=logger.info)

# Baseline
logger.info('-'*20 + f'{bcolors.BOLD_RED} Baseline model {bcolors.RESET}' + '-'*20)
ckp_file = f'{learned_model_dir.replace("full", "clean")}/checkpoints/ckp.pth'
logger.info(ckp_file)
ckp = torch.load(ckp_file)
baseline_model = eval(learn_cfg.net.backbone)(**learn_cfg.net.params)
baseline_model.load_state_dict(ckp['model'])
baseline_model.eval()
baseline_model.cuda();

evaluate(baseline_model, clean_train_noaug_loader, 'Clean train', displayer=logger.info)
evaluate(baseline_model, poison_train_noaug_loader, 'Poison train', displayer=logger.info)
evaluate(baseline_model, poison_clean_train_noaug_loader, 'Poison-Clean train', displayer=logger.info)
evaluate(baseline_model, test_loader, 'Test', displayer=logger.info)

conv_fea_dict, linear_fea_dict = get_feature_dict(cfg.model)
poison_clean_svd_file = f'exp/svd/{cfg.model}_{cfg.dataset}/clean_{cfg.num_poison}.pt'
os.makedirs(f'exp/svd/{cfg.model}_{cfg.dataset}/', exist_ok=True)
if not os.path.isfile(poison_clean_svd_file):
    poison_clean_svd = compute_svd(poisoned_model, clean_train_aug_loader, conv_fea_dict, linear_fea_dict, printer=logger.info)
    torch.save(poison_clean_svd, poison_clean_svd_file)
else:
    poison_clean_svd = torch.load(poison_clean_svd_file)


logger.info('-'*20 + f'{bcolors.BOLD_RED} Compute Projection Matrix {bcolors.RESET}' + '-'*20)
retained_var = cfg.retained_var
P, M = {}, {}
for layer in poison_clean_svd: # ,
    k = torch.sum((torch.cumsum(poison_clean_svd[layer]['S'], dim=0) / torch.sum(poison_clean_svd[layer]['S'])) < retained_var)
    
    M_tmp = poison_clean_svd[layer]['U'][:, :k]
    P[layer] = torch.mm(M_tmp, M_tmp.t()).to(device).float()
    M[layer] = M_tmp.to(device).float()
    logger.info(f' - {bcolors.BOLD_RED}Layer:{bcolors.RESET} {layer} - {P[layer].shape} - {retained_var}')
    logger.info(f"{k.item()}\t {poison_clean_svd[layer]['S'].shape[0]}\t {100*k.item()/poison_clean_svd[layer]['S'].shape[0]:.06f}")

unlearn_model = backbone.SmallVGG_v1(space=M, **learn_cfg.net.params)
unlearn_model.load_state_dict(poisoned_model.state_dict(), strict=False)
unlearn_model.to(device)
optimizer = optim.SGD(unlearn_model.parameters(), lr=0.02, momentum=0, weight_decay=0)
global_epoch = 0
start_lr = cfg.get('start_lr', 0.1)
end_lr = cfg.get('end_lr', 0.005)
num_epochs = int(cfg.get('num_epochs', 100))

offset = float(cfg.get('offset', 0.1))
loss1_w = float(cfg.get('loss1_w', 1))
loss2_w = float(cfg.get('loss2_w', 1))
wd = float(cfg.get('wd', 0))

alpha = np.exp(np.log(end_lr/start_lr) / num_epochs)
base_lr = start_lr

clean_res, poison_res, poison_clean_res, test_res = [], [], [], []

exp_dir = f'{cfg.work_dir}/results/'
if os.path.isdir(exp_dir):
    shutil.rmtree(exp_dir)
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(f'{exp_dir}/ckp/', exist_ok=True)

ce_losses = AverageMeter()
losses1, losses2 = AverageMeter(), AverageMeter()

# try:
for epoch in range(num_epochs):
    # Unlearn Training
    unlearn_model.train()
    unlearn_model.apply(freeze_norm_stats)
    for batch_idx, (images, labels) in enumerate(poison_train_aug_loader):
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

            tmp_outputs = torch.cat([outputs[j][:labels[j]], outputs[j][labels[j]+1:]])
            tmp_logits = F.softmax(tmp_outputs, dim=0)
            tmp_entropy = torch.sum(-tmp_logits*torch.log(tmp_logits + 1e-10))
            loss2 += tmp_entropy
            
        loss1 /= images.shape[0]
        loss2 /= images.shape[0]
        loss = loss1_w*loss1 + loss2_w*loss2

        losses1.update(loss1.item(), images.shape[0])
        losses2.update(loss2.item(), images.shape[0])
        ce_losses.update(loss.item(), images.shape[0])
        loss.backward()

        with torch.no_grad():
            for name, param in unlearn_model.named_parameters():
                if name.find('scale') >= 0:
                    param.data -= (base_lr/10) * param.grad.data
                    continue

                xs = name.split('.')
                if xs[-1] == 'bias':
                    continue

                P_name = name
                for i in range(20):
                    P_name = P_name.replace(f'.{i}.', f'[{i}].')
                P_name = P_name.replace('.bias', '').replace('.weight', '')

                if P_name not in P:
                    continue
                    
                sz = param.data.shape[0]
                reg_grd = param.grad.data.add(param.data, alpha=wd)
                reg_grd = reg_grd - torch.mm(reg_grd.view(sz,-1), P[P_name]).view(param.size())
                    
                lr = base_lr
                param.data -= lr * reg_grd 
    
    # Evaluate
    acc_dict, loss_dict = {}, {}
    res = evaluate(unlearn_model, clean_train_noaug_loader, '[Unlearned] - Clean', displayer=logger.info)
    loss_dict['Clean'], acc_dict['Clean'] = res
    clean_res.append(res)

    loss, acc = evaluate(unlearn_model, poison_train_noaug_loader, '[Unlearned] - Poison', displayer=logger.info)
    loss_dict['Poison'], acc_dict['Poison'] = loss, acc
    poison_res.append((loss, acc))

    loss, acc = evaluate(unlearn_model, poison_clean_train_noaug_loader, '[Unlearned] - Poison-Clean', displayer=logger.info)
    loss_dict['Poison-Clean'], acc_dict['Poison-Clean'] = loss, acc
    poison_clean_res.append((loss, acc))
    
    loss, acc = evaluate(unlearn_model, test_loader, '[Unlearned] - Test', displayer=logger.info)
    loss_dict['Test'], acc_dict['Test'] = loss, acc
    test_res.append((loss, acc))

    bins = np.linspace(-20, 1, 100)
    board_writer.add_scalars(f'Unlearn/Accuracy/', acc_dict, epoch)
    board_writer.add_scalars(f'Unlearn/Loss/', loss_dict, epoch)

    logger.info(f'Epoch: {epoch:02d} - Loss: {ce_losses.avg:.04f} ({losses1.avg:.04f}+{losses2.avg:.04f})' \
                f' - LR: {base_lr:.06f}')
    
    checkpoint = {
        'epoch': global_epoch,
        'lr': base_lr, 
    }
    if cfg.get('save_model', 0) != 0:
        checkpoint.update({'model': unlearn_model.state_dict()})
    torch.save(checkpoint, f'{exp_dir}/ckp/ckp_{global_epoch:03d}.pt')
    logger.info('----------------------------------------------------------')
    ce_losses.reset()
    losses1.reset()
    losses2.reset()
    base_lr *= alpha
    global_epoch += 1

    clean_acc = [x[1] for x in clean_res]
    poison_acc = [x[1] for x in poison_res]
    poison_clean_acc = [x[1] for x in poison_clean_res]
    test_acc = [x[1] for x in test_res]
    
    plt.plot(np.arange(0, len(clean_acc)), clean_acc, label='Clean')
    plt.plot(np.arange(0, len(poison_acc)), poison_acc, label='Poison')
    plt.plot(np.arange(0, len(poison_clean_acc)), poison_clean_acc, label='Poison-Clean')
    plt.plot(np.arange(0, len(test_acc)), test_acc, label='Test')
    
    plt.grid()
    plt.legend()
    plt.savefig(f'{exp_dir}/accuracies.jpg')
    plt.close()

    exp_res = {'clean_res': clean_res,
                'poison_res': poison_res,
                'test_res': test_res,
                'poison_clean_res': poison_clean_res,
                }

    np.save(f'{exp_dir}/exp_res.npy', exp_res)

    if test_acc[0] - test_acc[-1] > 5:
        break
# except Exception as e:
#     print(e)



