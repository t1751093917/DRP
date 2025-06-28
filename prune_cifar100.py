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

import torch_pruning as tp
from torch_pruning_.taylor_step_pruner import TaylorStepPruner

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


def build_taylor_pruner(net, train_loader: DataLoader, device, pruner, theta, delta, result_dir, cfg):
    clean_probs_all = []
    indices_all = []
    pbar = tqdm(train_loader, ascii=' >', desc='backward')
    for it, sample in enumerate(pbar):
        indices = sample['index']
        input, _ = sample['data']
        labels = sample['label'].to(device)
        logits = net(input.to(device))
        probs = F.softmax(logits, dim=1)
        check_nan(logits, 'logits')
        js = js_divergence(probs, labels)
        indices_all.extend(indices.tolist())
        clean_probs_all.extend((1 - js).tolist())
        # clean_probs_all.extend([int(train_loader.dataset.targets[idx] == label) for idx, label in zip(indices.tolist(), labels.tolist())])

    sorted_z = sorted(zip(indices_all, clean_probs_all), reverse=True, key=lambda x: x[1])
    indices_sorted, _ = zip(*sorted_z)
    indices_sorted = list(indices_sorted)
    with open(f'{result_dir}/data_clean_probs.txt', 'w', encoding='utf-8') as f:
        for tup in sorted_z:
            print(' '.join(map(str, tup)), file=f)
    print('Taylor pruner: theta {}, delta {}'.format(theta, delta))
    batch_size = train_loader.batch_size
    train_dataset = train_loader.dataset
    num_sample = round(batch_size * delta)
    # [clean, noise]
    sample_inds = [indices_sorted[:num_sample], indices_sorted[-num_sample:]]
    ratios = [1, theta]
    pruner.set_storage()
    for indices, ratio in zip(sample_inds, ratios):
        print(len(indices))
        subset = torch.utils.data.Subset(train_dataset, indices)
        # batch_size = 4
        loader = DataLoader(subset, batch_size=batch_size, num_workers=8)
        for sample in loader:
            x1, x2 = sample['data']
            x1, x2 = x1.to(device), x2.to(device)
            y = sample['label'].to(device)
            logits1 = net(x1)
            logits2 = net(x2)
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)

            N, C = logits1.shape
            given_labels = torch.full(size=(N, C), fill_value=cfg.eps / (C - 1)).to(device)
            given_labels.scatter_(dim=1, index=torch.unsqueeze(y, dim=1), value=1 - cfg.eps)
            with torch.no_grad():
                prob_clean = 1 - js_div(probs1, given_labels)
            losses = cross_entropy(logits1, given_labels, reduction='none') + cross_entropy(logits2, given_labels,
                                                                                            reduction='none')
            # loss = losses[prob_clean >= cfg.tau_clean].mean()
            loss = losses.mean()
            loss.backward()
            # for param in net.parameters():
            #     if param.grad is not None:
            #         param.grad.data = torch.abs(param.grad.data)
        # calculate importance
        pruner.store_importance(add=False, ratio=ratio)
        # clear grad
        for param in net.parameters():
            if param.grad is not None:
                param.grad = None


def prune(net, train_loader, test_loader, device, result_dir, method, rate, params):
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

        x1, x2 = example_data['data']
        x1, x2 = x1.to(device), x2.to(device)
        y = example_data['label'].to(device)
        logits1 = net(x1)
        logits2 = net(x2)
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)

        N, C = logits1.shape
        given_labels = torch.full(size=(N, C), fill_value=params.eps / (C - 1)).to(device)
        given_labels.scatter_(dim=1, index=torch.unsqueeze(y, dim=1), value=1 - params.eps)
        with torch.no_grad():
            prob_clean = 1 - js_div(probs1, given_labels)
        losses = cross_entropy(logits1, given_labels, reduction='none') + cross_entropy(logits2, given_labels,
                                                                                        reduction='none')
        # loss = losses[prob_clean >= params.tau_clean].mean()
        loss = losses.mean()
        loss.backward()
    elif method == "*Taylor":
        imp = tp.importance.TaylorImportance(multivariable=True)
    else:
        raise NotImplementedError

    ignored_layers = [net.classifier]
    iterative_steps = 1

    if method == "*Taylor":
        pruner = TaylorStepPruner(
            net,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=rate,
            ignored_layers=ignored_layers,
        )
        build_taylor_pruner(net, train_loader, device, pruner, params.theta, params.delta, result_dir, params)
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
    transform = build_transform(rescale_size=cfg.rescale_size, crop_size=cfg.crop_size)
    dataset = build_cifar100n_dataset(os.path.join(cfg.database, cfg.dataset), CLDataTransform(transform['cifar_train']), transform['cifar_test'],
                                      noise_type=cfg.noise_type, openset_ratio=cfg.openset_ratio, closeset_ratio=cfg.closeset_ratio)
    train_loader = DataLoader(dataset['train'], batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    # model, optimizer, scheduler --------------------------------------------------------------------------------------------------------------------
    n_classes = int(cfg.n_classes * (1 - cfg.openset_ratio)) if cfg.dataset.startswith('cifar') else cfg.n_classes
    print_to_console(f'> number of classes: {n_classes}', color='red')
    net = Model(arch=cfg.net, num_classes=n_classes, pretrained=True)
    net_ema = Model(arch=cfg.net, num_classes=n_classes, pretrained=True)
    net.to(device); net_ema.to(device)

    accuracy_init = 0
    if cfg.weights is not None and len(cfg.weights) > 0:
        net.load_state_dict(torch.load(cfg.weights))
        print('Load weights {}.'.format(cfg.weights))
        accuracy_init = evaluate(test_loader, net, device)
        print(f'Initial accuracy {accuracy_init}')
    net, prune_info = prune(net, train_loader, test_loader, device, result_dir, cfg.prune_method, cfg.prune_rate, cfg)
    net_ema = copy.deepcopy(net)

    # log network
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())
    # input_size = (3, 32, 32)
    # torchsummary.summary(net, input_size=input_size, batch_size=cfg.batch_size)

    # optimizer = build_adam_optimizer(net.parameters(), cfg.lr)
    optimizer = build_sgd_optimizer(net.parameters(), cfg.lr, cfg.weight_decay)
    opt_lvl = 'O1' if cfg.use_fp16 else 'O0'
    # [net, net_ema], optimizer = amp.initialize([net.to(device), net_ema.to(device)], optimizer, opt_level=opt_lvl,
    #                                            keep_batchnorm_fp32=None, loss_scale=None, verbosity=0)

    # Adjust learning rate and betas for Adam Optimizer
    epoch_decay_start = 80
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

    ema = EMA(net, alpha=0.99)
    ema.apply_shadow(net_ema)

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

        if epoch < cfg.warmup_epochs:
            threshold_clean = min(cfg.tau_clean * epoch / cfg.warmup_epochs, cfg.tau_clean)
        else:
            threshold_clean = (tau_c_max - cfg.tau_clean) * (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs) + cfg.tau_clean
        # print_to_console(f'> threshold_clean: {threshold_clean:.5f}', color='red')
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
            with torch.no_grad():
                logits1_ema = net_ema(x1)
                logits2_ema = net_ema(x2)
                soft_labels = (F.softmax(logits1_ema, dim=1) + F.softmax(logits2_ema, dim=1)) / 2

                prob_clean = 1 - js_div(probs1, given_labels)

            if epoch < cfg.warmup_epochs:
                if flag == 0:
                    step_flagging(f'start the warm-up step for {cfg.warmup_epochs} epochs.')
                    flag += 1
                losses = cross_entropy(logits1, given_labels, reduction='none') + cross_entropy(logits2, given_labels, reduction='none')
                # loss = losses[prob_clean >= threshold_clean].mean()
                loss = losses.mean()
            else:
                if flag == 1:
                    step_flagging('start the robust learning step.')
                    flag += 1

                target_labels = given_labels.clone()
                idx_clean = (prob_clean >= threshold_clean).nonzero(as_tuple=False).squeeze(dim=1)
                _, preds1 = probs1.topk(1, 1, True, True)
                _, preds2 = probs2.topk(1, 1, True, True)
                disagree = (preds1 != preds2).squeeze(dim=1)
                agree = (preds1 == preds2).squeeze(dim=1)
                unclean = (prob_clean < threshold_clean)
                idx_ood = (disagree * unclean).nonzero(as_tuple=False).squeeze(dim=1)
                idx_id = (agree * unclean).nonzero(as_tuple=False).squeeze(dim=1)
                target_labels[idx_id] = soft_labels[idx_id]
                target_labels[idx_ood] = F.softmax(soft_labels[idx_ood] / 10, dim=1)

                # classification loss
                losses = cross_entropy(logits1, target_labels, reduction='none') + cross_entropy(logits2, target_labels, reduction='none')
                loss_c = losses.mean()

                # consistency loss
                sign = torch.ones(N).to(device)
                sign[idx_ood] *= -1
                losses_o = symmetric_kl_div(probs1, probs2) * sign
                loss_o = losses_o.mean()

                # final loss
                loss = (1 - cfg.alpha) * loss_c + loss_o * cfg.alpha

            train_acc = accuracy(logits1, y, topk=(1,))
            train_accuracy.update(train_acc[0], x1.size(0))
            train_loss.update(loss.item(), x1.size(0))
            if cfg.use_fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            ema.update_params(net)
            ema.apply_shadow(net_ema)

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
        result_ls = [accuracy_init, cfg.prune_method, cfg.prune_rate, prune_info[-1], round(best_accuracy, 3), best_epoch]
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

    weight_base, _ = os.path.splitext(os.path.basename(config.weights))
    weight_epoch = int(weight_base.split('_')[-1])
    config.log = f'{weight_epoch}-{config.prune_method}-{config.prune_rate}'
    print(config)
    return config


if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
