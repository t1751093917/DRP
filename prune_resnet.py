import torch
import torch.nn as nn
import torchvision




import copy
import os
import pathlib
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchsummary
from apex import amp
from utils.core import accuracy, evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console, step_flagging
from utils.plotter import plot_results
from utils.ema import EMA
from utils.model import Model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import torch_pruning as tp
from torch_pruning_.taylor_step_pruner import TaylorStepPruner
from torch_pruning_.clsaware_pruner import ClassAwarePruner

class CLDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def adjust_lr_beta1(optimizer, lr, beta1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, 0.999)  # Only change beta1


def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def js_divergence(pred_probs, labels):
    # 将标签转换为 one-hot 编码
    N, C = pred_probs.shape
    # labels_onehot = torch.zeros((N, C), device=labels.device)
    # labels_onehot.scatter_(1, labels.view(-1, 1), 1)
    eps = 0.6 #1e-8
    labels_onehot = torch.full(size=(N, C), fill_value=eps / (C - 1)).to(labels.device)
    labels_onehot.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1 - eps)

    # M = (P + Q) / 2
    m = 0.5 * (pred_probs + labels_onehot)
    #  KL(P || M)  KL(Q || M)
    kl_p_m = kl_div(pred_probs, m)
    check_nan(kl_p_m, 'kl_p_m')
    kl_q_m = kl_div(labels_onehot, m)
    check_nan(kl_q_m, 'kl_q_m')
    # JS = 0.5 * (KL(P || M) + KL(Q || M))
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div


def check_nan(x: torch.Tensor, name=''):
    nan_tensor = torch.isnan(x)
    has_nan = nan_tensor.any()
    all_nan = nan_tensor.all()
    if has_nan:
        print(f'!!! {name} nan !!!', all_nan.item())
    return has_nan


class Sample_sort:
    def __init__(self, index, clean_prob):
        self.index = index
        self.clean_prob = clean_prob
        self.distance = 0
        self.id_sorted_clean = 0
        self.id_sorted_distance = 0

    def __str__(self):
        return ''

def build_nap_pruner(net, train_loader: DataLoader, device, pruner: TaylorStepPruner, theta, delta, result_dir, cfg):
    train_loader = DataLoader(train_loader.dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    clean_probs_all = []
    indices_all = []
    features_all = []
    net_feature = torch.nn.Sequential(*(list(net.children())[:-1]))
    # print(net_feature)
    samples = []
    pbar = tqdm(train_loader, ascii=' >', desc='backward')
    for it, sample in enumerate(pbar):
        indices = sample['index']
        input, _ = sample['data']
        input = input.to(device)
        labels = sample['label'].to(device)
        logits = net(input)
        features = net_feature(input)
        features = torch.flatten(features, 1)
        # if it == 0:
        #     print(features.shape)
        probs = F.softmax(logits, dim=1)
        js = js_divergence(probs, labels)
        indices_all.append(indices.cpu().detach())
        features_all.append(features.cpu().detach())
        if not cfg.ideal:
            clean_probs = (1 - js).tolist()
            clean_probs_all.append((1 - js).cpu().detach())
        else:
            clean_probs = [int(train_loader.dataset.targets[idx] == label) for idx, label in zip(indices.tolist(), labels.tolist())]
            clean_probs_all.extend(clean_probs)
        # for index, clean_prob in zip(indices, clean_probs):
        #     samples.append(Sample_sort(index.item(), clean_prob))
    indices_all = torch.cat(indices_all, dim=0)
    clean_probs_all = torch.cat(clean_probs_all, dim=0)
    # print(indices_all.shape, clean_probs_all.shape)
    features_all = torch.cat(features_all, dim=0)
    noisy_labels = torch.Tensor(train_loader.dataset.noisy_labels)
    noisy_labels = noisy_labels[indices_all]

    knn = KNeighborsClassifier(n_neighbors=cfg.k)
    knn.fit(features_all, noisy_labels)
    preds = torch.Tensor(knn.predict(features_all))
    # print(preds.shape, (noisy_labels[indices_all]).shape)
    # print(preds[:10], (noisy_labels[indices_all])[:10])
    cln_knn = preds == noisy_labels
    samples_ts = torch.cat([indices_all.unsqueeze(1), clean_probs_all.unsqueeze(1), cln_knn.unsqueeze(1)], dim=1)
    samples_cln = samples_ts[cln_knn]
    samples_ns = samples_ts[~cln_knn]
    # samples_ts = torch.cat([samples_ts[torch.argsort(samples_ts[:, 1], descending=True)], indices_all], dim=1)
    samples_cln = samples_cln[torch.argsort(samples_cln[:, 1], descending=True)]
    samples_ns = samples_ns[torch.argsort(samples_ns[:, 1], descending=False)]
    samples_ts = samples_ts[torch.argsort(samples_ts[:, 1], descending=True)]

    with open(f'{result_dir}/data_clean_probs.txt', 'w', encoding='utf-8') as f:
        for sample in samples_ts:
            print('\t'.join(map(str, sample.tolist())), file=f)
    indices_sorted = samples_ts[:, 0].to(torch.int)
    indices_cln = samples_cln[:, 0].to(torch.int)
    indices_ns = samples_ns[:, 0].to(torch.int)
    print(indices_cln.shape, indices_ns.shape)
    print('Taylor pruner: theta {}, delta {}'.format(theta, delta))
    batch_size = train_loader.batch_size
    train_dataset = train_loader.dataset
    num_sample = round(delta)  # batch_size *
    # [clean, noise]
    if not cfg.ideal:
        sample_inds = [indices_cln[:num_sample], indices_ns[:num_sample]]
        # print('JS only')
        # sample_inds = [indices_sorted[:num_sample], indices_sorted[-num_sample:]]
        # print('knn only')
        # sample_inds = [random.sample(list(indices_cln), num_sample), random.sample(list(indices_ns), num_sample)]
    else:
        sample_inds = [indices_cln[:num_sample], indices_ns[:num_sample]]
        # sample_inds = [random.sample(indices_sorted_js[:clean_probs_sorted.index(0)], num_sample),
        #                random.sample(indices_sorted_js[clean_probs_sorted.index(0):], num_sample)]

    ratios = [1, theta]
    pruner.set_storage()
    flag = True
    for indices, ratio in zip(sample_inds, ratios):
        print(len(indices))
        subset = torch.utils.data.Subset(train_dataset, indices)
        # batch_size = cfg.bs
        loader = DataLoader(subset, batch_size=batch_size, num_workers=8)
        for sample in loader:
            x1, x2 = sample['data']
            x = x1.to(device)
            y = sample['label'].to(device)
            logits = net(x)
            probs = F.softmax(logits, dim=1)
            js = js_divergence(probs, y)
            clean_probs = (1 - js) if flag else js

            N, C = logits.shape
            given_labels = torch.full(size=(N, C), fill_value=cfg.eps / (C - 1)).to(device)
            given_labels.scatter_(dim=1, index=torch.unsqueeze(y, dim=1), value=1 - cfg.eps)
            losses = cross_entropy(logits, given_labels, reduction='none') #+ cross_entropy(logits2, given_labels, reduction='none')
            # losses = F.cross_entropy(logits1, y)
            # loss = losses.mean()
            # loss.backward()
            losses.backward(gradient=clean_probs)
            # for param in net.parameters():
            #     if param.grad is not None:
            #         param.grad.data = torch.abs(param.grad.data)
        # calculate importance
        pruner.store_importance(add=False, ratio=ratio)
        # clear grad
        for param in net.parameters():
            if param.grad is not None:
                param.grad = None
        flag = False


def build_drp_pruner(net, train_loader: DataLoader, device, pruner: TaylorStepPruner, theta, delta, result_dir, cfg):
    train_loader = DataLoader(train_loader.dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    clean_probs_all = []
    indices_all = []
    features_all = []
    predicts_all = []
    # net_feature = torch.nn.Sequential(*(list(net.children())[:-1]))
    # print(net_feature)
    pbar = tqdm(train_loader, ascii=' >', desc='backward')
    for it, sample in enumerate(pbar):
        indices = sample['index']
        input, _ = sample['data']
        input = input.to(device)
        labels = sample['label'].to(device)
        logits = net(input)
        # features = net_feature(input)
        # features = torch.flatten(features, 1)
        probs = F.softmax(logits, dim=1)
        confs, predicts = probs.max(1)
        js = js_divergence(probs, labels)
        indices_all.append(indices.cpu().detach())
        # features_all.append(features.cpu().detach())
        predicts_all.append(predicts.cpu().detach())

        clean_probs_all.append((1 - js).cpu().detach())
        # clean_probs = [int(train_loader.dataset.targets[idx] == label) for idx, label in zip(indices.tolist(), labels.tolist())]
    indices_all = torch.cat(indices_all, dim=0)
    clean_probs_all = torch.cat(clean_probs_all, dim=0)
    predicts_all = torch.cat(predicts_all, dim=0)
    # noisy_labels = torch.Tensor(train_loader.dataset.noisy_labels)
    # noisy_labels = noisy_labels[indices_all]
    noisy_labels = train_loader.dataset.noisy_labels

    # sort
    samples_ts = torch.cat([indices_all.unsqueeze(1), clean_probs_all.unsqueeze(1), predicts_all.unsqueeze(1)], dim=1)
    # samples_ts = torch.cat([samples_ts[torch.argsort(samples_ts[:, 1], descending=True)], indices_all], dim=1)
    samples_ts = samples_ts[torch.argsort(samples_ts[:, 1], descending=True)]
    with open(f'{result_dir}/data_clean_probs.txt', 'w', encoding='utf-8') as f:
        for sample in samples_ts:
            print('\t'.join(map(str, sample.tolist())), file=f)
    indices_sorted = samples_ts[:, 0].to(torch.int)
    predicts_sorted = samples_ts[:, 2].to(torch.int)
    print('Taylor pruner: theta {}, delta {}'.format(theta, delta))
    batch_size = train_loader.batch_size
    train_dataset = train_loader.dataset
    num_sample = round(delta)  # batch_size *

    # select first [clean, noise]
    sample_inds = [indices_sorted[:num_sample], indices_sorted[-num_sample:]]
    predicts_noisy = predicts_sorted[-num_sample:]
    # classify from selected
    class_clean = [[] for _ in range(cfg.noise_nc)]
    class_noise = [[] for _ in range(cfg.noise_nc)]
    for sample_cln, sample_ns, ns_class in zip(sample_inds[0], sample_inds[1], predicts_noisy):
        class_clean[noisy_labels[sample_cln]].append(sample_cln)
        class_noise[ns_class].append(sample_ns)
    # weighted
    '''num_cls_clean = torch.Tensor([len(ls) for ls in class_clean])
    # weight_cls_clean = (torch.max(num_cls_clean) + torch.min(num_cls_clean) - num_cls_clean) / torch.sum(num_cls_clean) # F.softmax(torch.Tensor(num_cls_clean))
    weight_cls_clean = num_cls_clean / torch.sum(num_cls_clean) # F.softmax(torch.Tensor(num_cls_clean))
    num_cls_noise = torch.Tensor([len(ls) for ls in class_noise])
    weight_cls_noise = num_cls_noise / torch.sum(num_cls_noise) # F.softmax(torch.Tensor(num_cls_noise))
    print(num_cls_clean, weight_cls_clean, num_cls_noise, weight_cls_noise, sep='\n')
    # print(torch.sum(weight_cls_clean), torch.max(weight_cls_clean), torch.sum(weight_cls_noise), torch.max(weight_cls_noise))
'''
    # classify first
    # num_sample = 50
    # class_sorted = [[] for _ in range(cfg.noise_nc)]
    # for sample_idx in indices_sorted:
    #     sample_idx = sample_idx.item()
    #     class_sorted[noisy_labels[sample_idx]].append(sample_idx)

    pruner.set_storage()
    for i in range(1):  # cfg.noise_nc
        # class_inds = [class_clean[i], class_noise[i]]
        class_inds = [sample_inds[0], sample_inds[1]]
        print(i, len(class_inds[0]), len(class_inds[1]), end=';\t')
        # select per class
        # class_inds = [class_sorted[i][:num_sample], class_sorted[i][-num_sample:]]

        # ratios = [weight_cls_clean[i].item(), theta * weight_cls_noise[i].item()]
        # ratios = [num_cls_clean[i].item(), theta * num_cls_noise[i].item()]
        ratios = [1, theta]
        clean = True
        for indices, ratio in zip(class_inds, ratios):
            if len(indices)<=0:
                continue
            # print(indices)
            subset = torch.utils.data.Subset(train_dataset, indices)
            loader = DataLoader(subset, batch_size=batch_size, num_workers=8)

            imp = pruner.importance
            imp._prepare_model(net, pruner)
            for sample in loader:
                x1, x2 = sample['data']
                x = x1.to(device)
                y = sample['label'].to(device)
                logits = net(x)
                probs = F.softmax(logits, dim=1)
                js = js_divergence(probs, y)
                clean_probs = (1 - js) if clean else js

                N, C = logits.shape
                given_labels = torch.full(size=(N, C), fill_value=cfg.eps / (C - 1)).to(device)
                given_labels.scatter_(dim=1, index=torch.unsqueeze(y, dim=1), value=1 - cfg.eps)
                losses = cross_entropy(logits, given_labels, reduction='none') #+ cross_entropy(logits2, given_labels, reduction='none')
                # losses = F.cross_entropy(logits1, y)
                # loss = losses.mean()
                # loss.backward()
                losses.backward(gradient=clean_probs)
                imp.step()
            # calculate importance
            pruner.store_importance(add=clean, ratio=ratio)
            imp._rm_hooks(net)
            imp._clear_buffer()
            # clear grad
            for param in net.parameters():
                if param.grad is not None:
                    param.grad = None
            clean = False


def build_clsa_pruner(net, train_loader: DataLoader, device, pruner: ClassAwarePruner, cfg):
    train_dataset = train_loader.dataset
    clses_idxs = [[] for _ in range(cfg.noise_nc)]
    for idx, label in enumerate(train_dataset.noisy_labels):
        clses_idxs[label].append(idx)
    # num_selected = round(cfg.delta)
    num_selected = 200
    print("Selected number per class:", num_selected)
    pruner.set_clswise_storage(cfg.noise_nc)
    for i in range(cfg.noise_nc):
        selected_idxs = random.sample(clses_idxs[i], num_selected)
        subset = torch.utils.data.Subset(train_dataset, selected_idxs)
        loader = DataLoader(subset, batch_size=cfg.batch_size, num_workers=8)
        pbar = tqdm(loader, ascii=' >', desc=f'class{i}', leave=False)
        for it, sample in enumerate(pbar):
            one_backward(sample, net, device)
        pruner.store_importance(i)
        for param in net.parameters():
            param.grad = None

def one_backward(samples, net, device):
    x, _ = samples['data']
    x = x.to(device)
    y = samples['label'].to(device)
    logits = net(x)

    N, C = logits.shape
    given_labels = torch.full(size=(N, C), fill_value=params.eps / (C - 1)).to(device)
    given_labels.scatter_(dim=1, index=torch.unsqueeze(y, dim=1), value=1 - params.eps)
    losses = cross_entropy(logits, given_labels, reduction='none') 
    # losses = F.cross_entropy(logits, y)
    loss = losses.mean()
    loss.backward()


def pretrain(epochs, net, train_loader: DataLoader, test_loader: DataLoader, device, cfg):
    train_dataset = train_loader.dataset
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9, nesterov=True)
    correct_idxes = []
    # print(evaluate(test_loader, net, device))
    for epoch in range(0, epochs):
        if epoch>0:
            if len(correct_idxes) <= 0: break
            print(f"subset: {correct_idxes.max().item()}/{len(correct_idxes)}")
            subset = torch.utils.data.Subset(train_dataset, correct_idxes)
            train_loader = DataLoader(subset, batch_size=cfg.batch_size, num_workers=8)
            net.train()
        train_acc_ls = []
        correct_idxes = []
        pbar = tqdm(train_loader, ascii=' >', desc=f'pre epoch: {epoch}/{epochs}', leave=False)
        # train this epoch
        for it, sample in enumerate(pbar):
            optimizer.zero_grad()
            indices = sample['index']
            x1, _ = sample['data']
            x1 = x1.to(device)
            y = sample['label'].to(device)

            logits1 = net(x1)
            N, C = logits1.shape
            given_labels = torch.full(size=(N, C), fill_value=cfg.eps/(C - 1)).to(device)
            given_labels.scatter_(dim=1, index=torch.unsqueeze(y, dim=1), value=1-cfg.eps)

            probs = F.softmax(logits1, dim=1)
            confs, predicts = probs.max(1)
            correct_idxes.append(indices[y == predicts].cpu().detach())
            if epoch == 0: continue

            losses = cross_entropy(logits1, given_labels, reduction='none') #+ cross_entropy(logits2, given_labels, reduction='none')
            # losses = F.cross_entropy(logits1, y)
            loss = losses.mean()
            train_acc = accuracy(logits1, y, topk=(1,))
            train_acc_ls.append(train_acc[0])
            loss.backward()

            optimizer.step()
        # evaluate this epoch
        test_accuracy = evaluate(test_loader, net, device)
        train_acc_avg = 0
        if len(train_acc_ls)>0:
            train_acc_avg = sum(train_acc_ls) / len(train_acc_ls)
        print(f"epoch: {epoch}/{epochs}, train acc: {train_acc_avg}, test acc:{test_accuracy}")
        correct_idxes = torch.cat(correct_idxes, dim=0)

def prune(net, train_loader, test_loader, device, result_dir, method, rate, params):
    net.eval()
    example_data = next(iter(train_loader))
    example_inputs, _ = example_data['data']  # train_loader
    # example_inputs = example_data['data']  # test_loader
    example_inputs = example_inputs.to(device)
    print("-- Input size:", example_inputs.shape)

    if method == "random":
        imp = tp.importance.RandomImportance()
    elif method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
    elif method == "l2":
        imp = tp.importance.MagnitudeImportance(p=2)
    elif method == "fpgm":
        imp = tp.importance.FPGMImportance(p=2)
    elif method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
    elif method == "taylor":
        imp = tp.importance.TaylorImportance()
        one_backward(example_data, net, device)
    elif method == 'l2_0':
        imp = tp.importance.MagnitudeImportance(p=2, group_reduction='max')
    elif method == 'taylor_0':
        imp = tp.importance.TaylorImportance(group_reduction='max')
        one_backward(example_data, net, device)
    elif method == 'LPSR':  # SPL
        imp = tp.importance.TaylorImportance(normalizer='standarization')
        one_backward(example_data, net, device)
    elif method == 'Boosting':  # PRL
        pretrain(4, net, train_loader, test_loader, device, params)
        imp = tp.importance.MagnitudeImportance(p=2, group_reduction='max')
    elif method == "clsa":
        imp = tp.importance.TaylorImportance()
    elif method == "NAP":
        imp = tp.importance.TaylorImportance()
    elif method == "DRP":
        # imp = tp.importance.TaylorImportance(normalizer='standarization')
        # imp = tp.importance.GroupHessianImportance()
        imp = tp.importance.OBDCImportance(num_classes=params.noise_nc)
    else:
        raise NotImplementedError

    ignored_layers = [net.fc]
    iterative_steps = 1

    if method == "NAP":
        pruner = TaylorStepPruner(
            net,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=rate,
            ignored_layers=ignored_layers,
        )
        build_nap_pruner(net, train_loader, device, pruner, params.theta, params.delta, result_dir, params)
    elif method == "DRP":
        pruner = TaylorStepPruner(
            net,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=rate,
            ignored_layers=ignored_layers,
        )
        build_drp_pruner(net, train_loader, device, pruner, params.theta, params.delta, result_dir, params)
    elif method == "clsa":
        pruner = ClassAwarePruner(
            net,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=rate,
            ignored_layers=ignored_layers,
        )
        build_clsa_pruner(net, train_loader, device, pruner, params)
    else:
        pruner = tp.pruner.MagnitudePruner(
            net,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=rate,
            ignored_layers=ignored_layers,
        )
    print(f"Build {method} pruner, rate {rate}.")
    # prune
    base_macs, base_nparams = tp.utils.count_ops_and_params(net, example_inputs)
    print(f"Before prune: macs {base_macs / 1e6}M, params {base_nparams / 1e6}M")
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(net, example_inputs)
    print(f"After prune: macs {macs / 1e6}M, params {nparams / 1e6}M")
    param_rate = round(1 - nparams / base_nparams, 3)
    # print(net)
    # save and load
    torch.save(net, f'{result_dir}/prune.pth')
    # net = torch.load(f'{result_dir}/prune.pth')
    # evaluate
    test_accuracy = evaluate(test_loader, net, device)
    print("Accuracy after prune", round(test_accuracy, 3))
    return net, (method, rate, param_rate, test_accuracy)


def main(cfg, device):
    init_seeds()
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16

    # logging ----------------------------------------------------------------------------------------------------------------------------------------
    logger_root = f'Results/{cfg.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if cfg.resume is None:
        cfg.project = '' if cfg.project is None else cfg.project
        result_dir = os.path.join(logger_root, cfg.project, f'{cfg.log}-{logtime}')
        logger = Logger(logging_dir=result_dir, DEBUG=False)
        logger.set_logfile(logfile_name='log.txt')
    else:
        result_dir = os.pardir.split(cfg.resume)[0]  # TODO
        logger = Logger(logging_dir=result_dir, DEBUG=False)
        logger.set_logfile('resumed-log.txt')
    save_config(cfg, f'{result_dir}/config.cfg')
    save_params(cfg, f'{result_dir}/params.json', json_format=True)
    logger.debug(f'Result Path: {result_dir}')

    # dataset, dataloader ----------------------------------------------------------------------------------------------------------------------------
    cifar = torchvision.datasets
    transform = build_transform(rescale_size=cfg.rescale_size, crop_size=cfg.crop_size)
    dataset = build_cifar100n_dataset(os.path.join(cfg.database, cfg.dataset), CLDataTransform(transform['cifar_train']), transform['cifar_test'],
                                      noise_type=cfg.noise_type, openset_ratio=cfg.openset_ratio, closeset_ratio=cfg.closeset_ratio)
    train_loader = DataLoader(dataset['train'], batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    print('Train classes:', set(dataset['train'].noisy_labels))

    # model, optimizer, scheduler --------------------------------------------------------------------------------------------------------------------
    n_classes = int(cfg.n_classes * (1 - cfg.openset_ratio)) if cfg.dataset.startswith('cifar') else cfg.n_classes
    print_to_console(f'> number of classes: {n_classes}', color='red')
    # n_classes = 100
    cfg.noise_nc = n_classes

    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    net.to(device)

    accuracy_init = 0
    if cfg.weights is not None and len(cfg.weights) > 0:
        net.load_state_dict(torch.load(cfg.weights))
        print('Load weights {}.'.format(cfg.weights))
        accuracy_init = evaluate(test_loader, net, device)
        print(f'Initial accuracy {accuracy_init}')
    if cfg.prune_method is not None:
        net, prune_info = prune(net, train_loader, test_loader, device, result_dir, cfg.prune_method, cfg.prune_rate, cfg)

    # log network
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())
    # input_size = (3, 32, 32)
    # torchsummary.summary(net, input_size=input_size, batch_size=cfg.batch_size)

    # optimizer = build_adam_optimizer(net.parameters(), cfg.lr)
    optimizer = build_sgd_optimizer(net.parameters(), cfg.lr, cfg.weight_decay)

    # Adjust learning rate and betas for Adam Optimizer
    epoch_decay_start = 50
    mom1 = 0.9
    mom2 = 0.1
    lr_plan = [cfg.lr] * cfg.epochs
    beta1_plan = [mom1] * cfg.epochs
    for i in range(epoch_decay_start, cfg.epochs):
        lr_plan[i] = float(cfg.epochs - i) / (cfg.epochs - epoch_decay_start) * cfg.lr
        beta1_plan[i] = mom2

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    epoch_train_time = AverageMeter()

    # resume -----------------------------------------------------------------------------------------------------------------------------------------
    if cfg.resume is not None:
        assert os.path.isfile(cfg.resume), 'no checkpoint.pth exists!'
        logger.debug(f'---> loading {cfg.resume} <---')
        checkpoint = torch.load(f'{cfg.resume}')
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        best_epoch = checkpoint['best_epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        best_accuracy = 0.0
        best_epoch = None


    flag = 0
    tau_c_max = 0.95
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()

        # pre-step in this epoch
        net.train()
        adjust_lr_beta1(optimizer, lr_plan[epoch], beta1_plan[epoch])
        train_loss.reset()
        train_accuracy.reset()
        curr_lr = [group['lr'] for group in optimizer.param_groups]
        logger.debug(f'Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  Lr:[{curr_lr[0]:.5f}]')

        # train this epoch
        for it, sample in enumerate(train_loader):
            s = time.time()
            optimizer.zero_grad()
            # indices = sample['index']
            x1, x2 = sample['data']
            x1, x2 = x1.to(device), x2.to(device)
            y = sample['label'].to(device)

            logits1 = net(x1)
            logits2 = net(x2)
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)

            N, C = logits1.shape
            given_labels = torch.full(size=(N, C), fill_value=cfg.eps/(C - 1)).to(device)
            given_labels.scatter_(dim=1, index=torch.unsqueeze(y, dim=1), value=1-cfg.eps)

            losses = cross_entropy(logits1, given_labels, reduction='none') #+ cross_entropy(logits2, given_labels, reduction='none')
            # losses = F.cross_entropy(logits1, y)
            loss = losses.mean()

            train_acc = accuracy(logits1, y, topk=(1,))
            train_accuracy.update(train_acc[0], x1.size(0))
            train_loss.update(loss.item(), x1.size(0))
            loss.backward()

            optimizer.step()

            epoch_train_time.update(time.time() - s, 1)
            if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                  f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                  f"Train Accuracy:[{train_accuracy.avg:6.2f}]  " \
                                  f"Loss:[{train_loss.avg:4.4f}]  " \
                                  f"{epoch_train_time.avg:6.2f} sec/iter"
                logger.debug(console_content)

        # evaluate this epoch
        test_accuracy = evaluate(test_loader, net, device)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            torch.save(net.state_dict(), f'{result_dir}/best_epoch.pth')
        if cfg.prune_method is None and (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), f'{result_dir}/epoch_{epoch + 1}.pth')

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_epoch': best_epoch,
            'best_accuracy': best_accuracy,
            'optimizer': optimizer.state_dict(),
        }, filename=f'{result_dir}/checkpoint.pth')

        # logging this epoch
        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss.avg:>6.4f} | '
                    f'train accuracy: {train_accuracy.avg:>6.3f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')
        plot_results(result_file=f'{result_dir}/log.txt')

    if cfg.result_file is not None:
        init_acc = round(accuracy_init, 3)
        best_acc = round(best_accuracy, 3)
        if cfg.prune_method == 'DRP':
            # cfg.log.split('-')[0],
            result_ls = [init_acc, cfg.prune_rate, cfg.theta, cfg.delta, cfg.k, prune_info[-1], best_acc, best_epoch, round(best_accuracy - accuracy_init, 3)]
        else:
            result_ls = [init_acc, cfg.prune_method, cfg.prune_rate, prune_info[-1], best_acc, best_epoch]
        if cfg.tag is not None:
            result_ls.append(cfg.tag)
        with open(cfg.result_file, 'a') as f:
            print(','.join(map(str, result_ls)), file=f)
        print(f'Add result in {cfg.result_file}')

    # rename results dir -----------------------------------------------------------------------------------------------------------------------------
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--log_prefix', type=str)
    parser.add_argument('--log_freq', type=int)
    args = parser.parse_args()

    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])
    if config.dataset.startswith('cifar'):
        config.log = f'{config.net}-{config.noise_type}_closeset{config.closeset_ratio}_openset{config.openset_ratio}-{config.log_prefix}'
    else:
        config.log = f'{config.net}-{config.log_prefix}'
    if config.prune_method is not None:
        config.log = f'{config.prune_rate:.2f}_{config.prune_method}' # _{config.theta}_{config.delta}
        # if config.prune_method == 'DRP':
        #     config.log += f'-{config.theta}_{config.delta}' #_{config.k}
        # if config.ideal:
        #     config.log += '-ideal'
        # if config.weights is not None and len(config.weights) > 0:
        #     name, _ = os.path.splitext(os.path.basename(config.weights))
        #     config.log = name.split('_')[-1] + '-' + config.log
    if config.tag is not None:
        config.log = config.tag + config.log
    print(config)
    return config


if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')


