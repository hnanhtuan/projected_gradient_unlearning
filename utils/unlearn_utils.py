import os
import copy
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from utils.general_utils import extract_features, bcolors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_entropy(model, loader, ignore_first_k=0):
    with torch.no_grad():
        model.eval()
        entropies = []
        for batch_idx, (images, labels) in enumerate(loader):
            labels = labels.to(device)
            images = images.to(device)

            outputs = model(images)
            outputs = outputs[:, ignore_first_k:]
            scores = F.softmax(outputs, dim=1)
            entropy = torch.log(torch.sum(-scores*torch.log(scores), dim=1))
            entropies.append(entropy)
            
        entropies = torch.cat(entropies, dim=0).cpu().numpy().tolist()
        return entropies

def compute_svd(model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1, printer=print):
    if conv_fea_dict is None:
        conv_fea_dict = model.conv_fea_dict
    if linear_fea_dict is None:
        linear_fea_dict = model.linear_fea_dict
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    for md in model.modules():
        name = md.__class__.__name__
        if name == 'Dropout':
            md.training = True

    fea_dict = {}
    for val in {**conv_fea_dict, **linear_fea_dict}.values():
        fea_dict[val] = val

    printer(fea_dict)
    tmp_fea_dict = {**fea_dict}
    if 'input' in fea_dict:
        features = {'input': []}
        tmp_fea_dict.pop('input')
    else:
        features = {}

    fea_ext = create_feature_extractor(model, tmp_fea_dict)

    covar, svd = {}, {}
    for key in {**conv_fea_dict, **linear_fea_dict}.keys():
        covar[key] = 0
    with torch.no_grad():
        for _ in range(int(epochs)):
            for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader)):
                imgs, lbls = imgs.to(device), lbls.to(device)
                if 'input' in fea_dict:
                    features['input'] = imgs

                feats = fea_ext(imgs)
                for fea_name in feats:
                    features[fea_name] = feats[fea_name].detach()

                for layer in conv_fea_dict: 
                    ks = eval(f'model.{layer}').kernel_size
                    padding = eval(f'model.{layer}').padding
                    f = features[conv_fea_dict[layer]]
                    patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
                    fea_dim = patch.shape[1]
                    patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
                    covar[layer] += torch.mm(patch.permute(1, 0), patch)

                for layer in linear_fea_dict:
                    f = features[linear_fea_dict[layer]].double().squeeze()
                    covar[layer] += torch.mm(f.permute(1, 0), f)
        
        for layer in covar:
            stime = time.time()
            U, S, _ = torch.svd(covar[layer]/epochs)
            svd[layer] = {'U': U, 'S': torch.sqrt(S)}
            print(f'Layer: {layer} - SVD time: {time.time() - stime:.06f}')

    process_time = time.time() - start_time
    printer(f'Processing time: {process_time:.04f}')
    return svd

def compute_retain_svd(full_svd, model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1, printer=print):
    if conv_fea_dict is None:
        conv_fea_dict = model.conv_fea_dict
    if linear_fea_dict is None:
        linear_fea_dict = model.linear_fea_dict
    start_time = time.time()
    fea_dict = {}
    for val in {**conv_fea_dict, **linear_fea_dict}.values():
        fea_dict[val] = val

    max_dim = 0
    for k in {**conv_fea_dict, **linear_fea_dict}:
        w = eval(f'model.{k}.weight')
        fea_dim = w.numel() // w.shape[0]
        max_dim = max(max_dim, fea_dim) 

    # min_epochs = max(epochs, math.ceil(max_dim/len(data_loader.dataset)))
    min_epochs = epochs

    tmp_fea_dict = {**fea_dict}
    if 'input' in fea_dict:
        features = {'input': []}
        tmp_fea_dict.pop('input')
    else:
        features = {}

    fea_ext = create_feature_extractor(model, tmp_fea_dict)

    covar, retain_svd = {}, {}
    for key in {**conv_fea_dict, **linear_fea_dict}.keys():
        covar[key] = 0
    with torch.no_grad():
        for _ in range(int(min_epochs)):
            for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader)):
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                if 'input' in fea_dict:
                    features['input'] = imgs

                feats = fea_ext(imgs)
                for fea_name in feats:
                    features[fea_name] = feats[fea_name].detach()

                for layer in conv_fea_dict: 
                    ks = eval(f'model.{layer}').kernel_size
                    padding = eval(f'model.{layer}').padding
                    f = features[conv_fea_dict[layer]]
                    patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
                    fea_dim = patch.shape[1]
                    patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
                    covar[layer] += torch.mm(patch.permute(1, 0), patch)

                for layer in linear_fea_dict:
                    f = features[linear_fea_dict[layer]].double().squeeze()
                    covar[layer] += torch.mm(f.permute(1, 0), f)
        
        process_time = time.time() - start_time
        printer(f'Processing time: {process_time:.04f}')
        for layer in covar:
            stime = time.time()
            U, S = full_svd[layer]['U'].to(device), full_svd[layer]['S'].to(device)
            M = torch.mm(torch.mm(U, torch.diag(S**2)), U.t())

            M1 = M - covar[layer]/min_epochs
            U1_, S1sq_, _ = torch.svd(M1)
            retain_svd[layer] = {'U': U1_, 'S': torch.sqrt(S1sq_)}
            printer(f'Layer: {layer} - SVD time: {time.time() - stime:.06f} - M: {M.shape}')

    process_time = time.time() - start_time
    printer(f'Processing time: {process_time:.04f}')
    return retain_svd
      
def freeze_norm_stats(net):
    try:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

    except ValueError:  
        print("error with BatchNorm")
        return

