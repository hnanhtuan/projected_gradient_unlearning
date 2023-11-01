import torch
import sys
import numpy as np
from easydict import EasyDict as edict
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torch.optim.lr_scheduler import ExponentialLR
import backbone


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED   = "\033[1;31m"
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"
    BOLD_RED = "\033[;1m\033[1;31m"


def to_numberic(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def to_float(s):
    try:
        return float(s)
    except ValueError:
        return s


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class CustomExponentialLR(ExponentialLR):
    def __init__(self, optimizer, start_lr=1., end_lr=1, num_steps=1, last_epoch=-1, verbose=False):
        gamma = np.exp(np.log(end_lr/start_lr) / num_steps)
        super().__init__(optimizer, gamma, last_epoch, verbose)


def evaluate(model, data_loader, description='', displayer=print, num_forget_classes=None):
    ce_losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loss_function = nn.CrossEntropyLoss()
        for batch_idx, (images, labels) in enumerate(data_loader):
            labels = labels.to(device)
            images = images.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            ce_losses.update(loss.item(), images.shape[0])
            if num_forget_classes is not None:
                labels = labels - num_forget_classes
                outputs = outputs[:, num_forget_classes:]
            acc = accuracy(outputs, labels)
            accuracies.update(acc[0].item(), images.shape[0])

        displayer('[{}] Loss: {:.04f} - Acc: {:.04f}'.format(description, ce_losses.avg, accuracies.avg))
    return ce_losses.avg, accuracies.avg


def evaluate_entropy(model, data_loader, description='', displayer=print, ignore_topk=0):
    ce_losses = AverageMeter()
    accuracies = AverageMeter()
    entropies = []
    with torch.no_grad():
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loss_function = nn.CrossEntropyLoss()
        for batch_idx, (images, labels) in enumerate(data_loader):
            labels = labels.to(device)
            images = images.to(device)

            outputs = model(images)
            logits = F.softmax(outputs, dim=1)
            log_entropy = torch.log(torch.sum(-logits[:,ignore_topk:]*torch.log(logits[:,ignore_topk:]), dim=1))
            entropies.append(log_entropy)
            loss = loss_function(outputs, labels)
            ce_losses.update(loss.item(), images.shape[0])
            acc = accuracy(outputs, labels)
            accuracies.update(acc[0].item(), images.shape[0])

        displayer('[{}] Loss: {:.04f} - Acc: {:.04f}'.format(description, ce_losses.avg, accuracies.avg))
    entropies = torch.cat(entropies, dim=0).cpu().numpy().tolist()

    return ce_losses.avg, accuracies.avg, entropies


def extract_features(model, dataloader, fea_dict, epochs=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    for md in model.modules():
        name = md.__class__.__name__
        if name == 'Dropout':
            md.training = True
    labels = []
    tmp_fea_dict = {**fea_dict}
    if 'input' in fea_dict:
        features = {'input': []}
        tmp_fea_dict.pop('input')
    else:
        features = {}
    
    fea_ext = create_feature_extractor(model, tmp_fea_dict)
    with torch.no_grad():
        for _ in range(int(epochs)):
            for batch_idx, (imgs, lbls) in enumerate(dataloader):
                if 'input' in fea_dict:
                    features['input'].append(imgs)
                imgs = imgs.to(device)
                lbls = lbls.to(device)

                feats = fea_ext(imgs)
                for fea_name in feats:
                    if fea_name not in features:
                        features[fea_name] = []
                    features[fea_name].append(feats[fea_name].detach().cpu())
                labels.append(lbls)
            
    return features, torch.cat(labels, dim=0)


def _setattr(obj, attr, val):
    if attr.find('.') < 0:
        setattr(obj, attr, val)
        return

    fst_idx = attr.find('.')
    if not hasattr(obj, attr[:fst_idx]):
        setattr(obj, attr[:fst_idx], edict())
    cur_obj = getattr(obj, attr.split('.')[0])
    _setattr(cur_obj, attr[fst_idx+1:], val)


def parse_args(cfg):
    argv = sys.argv
    args = edict()
    cur_key = None
    for i in range(2, len(argv)):
        arg = argv[i]
        if arg.startswith('--'):
            cur_key = arg[2:]
            args[cur_key] = []
        else:
            if cur_key is not None:
                args[cur_key].append(to_numberic(arg))
    for k in args:
        if len(args[k]) == 0:
            args[k] = None
        elif len(args[k]) == 1:
            args[k] = args[k][0]

    for k in args:
        _setattr(cfg, k, args[k])


def load_config(cfg, root_cfg=None):
    if root_cfg is None:
        parse_args(cfg)
        root_cfg = cfg
    for key in cfg:
        if isinstance(cfg[key], str) and cfg[key].find('$') >= 0:
            value = eval(f'root_cfg.{cfg[key][1:]}')
            cfg[key] = value
        elif isinstance(cfg[key], dict):
            load_config(cfg[key], root_cfg)


def get_std_transforms(img_size):
    if img_size == 224:
        transform_train = transforms.Compose(
            [transforms.Resize(256),
             transforms.RandomCrop(224, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])
        transform_test = transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])
        return transform_train, transform_test
    elif img_size == 64:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(64, padding=8),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])
        return transform_train, transform_test
    elif img_size == 32:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])
        return transform_train, transform_test
    else:
        raise NotImplementedError('Currently only support output image size of 224, 64 or 32')

    
def plot_histograms(samples, names, num_bins=201, x_range=[-19, 1], y_range=None, title='', plot_name=None):
    plt.rcParams["figure.figsize"] = (15,10)
    plt.rcParams.update({'font.size': 22})
    bins = np.linspace(x_range[0], x_range[1], num_bins)
    assert len(samples) == len(names)
    for sample, name in zip(samples, names):
        plt.hist(sample, bins=bins, alpha=0.5, density=True, label=name)
    plt.legend()
    plt.grid()
    plt.title(title)
    if y_range is not None:
        plt.ylim(y_range)

    if plot_name is not None:
        plt.savefig(plot_name)
        plt.close()
    else:
        plt.show()


def islayer(md):
    name = md.__class__.__name__
    if name in ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d',
                 'AvgPool2d', 'Linear', 'Dropout', 'Identity', 'Flatten']:
        return True
    return False