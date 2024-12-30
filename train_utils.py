import gc
import os
import random

import numpy as np
import roma
import torch
import torch.nn as nn

from configs import ModelConfig
from model_utils import r6d2rot_mat


def to_device(data: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """
    send data to device
    :param data: data to send
    :param device: device to send
    :return: sent data
    """
    keys = data.keys()
    for k in keys:
        data[k] = data[k].to(device)

    return data


def set_seed(seed: int) -> None:
    """
    set random seed for modules :)
    :param seed: random seed
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def setup_env() -> None:
    """
    setup backend defaults :)
    """
    torch.backends.cudnn.benchmark = True
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    set_seed(0)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    get learning rate from optimizer :)
    :param optimizer: training optimizer
    :return: learning rate
    """
    return optimizer.param_groups[-1]['lr']


def count_parameters(model: torch.nn.Module) -> float:
    """
    count number of trainable parameters per million :)
    :param model: model
    :return: number of trainable parameters per million
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6


def create_log_dir(configs: ModelConfig):
    """
    create log directory :)
    :param configs: train configuration
    :return: tensorboard writer
    """
    if not configs.evaluate:
        os.makedirs(f'./train_log/{configs.exp_name}', exist_ok=True)
        os.makedirs(f'./train_log/{configs.exp_name}/checkpoint', exist_ok=True)
        os.makedirs(f'./train_log/{configs.exp_name}/predicts', exist_ok=True)
    configs.log_dir = f"./train_log/{configs.exp_name}/"


def resume(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
           scheduler: torch.optim.lr_scheduler.LRScheduler,
           configs: ModelConfig) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler,
int, float, float, dict[str, list[float]], dict[str, list[float]]]:
    """
    load parameters to continue training :)
    :param model: model :)
    :param optimizer: model optimizer :) :)
    :param scheduler: learning rate scheduler
    :param configs: training configs :)
    :return: model, optimizer, scheduler, start_epoch, train_hist, valid_hist
    """
    last_ckpt = os.path.join(configs.log_dir, f"checkpoint/last_{configs.exp_name}.pth")
    if os.path.isfile(last_ckpt) and not configs.overfit_test:
        try:
            loaded = torch.load(last_ckpt)
            model.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            scheduler.load_state_dict(loaded["scheduler"])
            start_epoch = loaded["epoch"] + 1
            train_hist = loaded["train_hist"]
            valid_hist = loaded["valid_hist"]
            acc = loaded["acc"]
            best_acc = loaded["best_acc"]
            print(f"Load all parameters from last checkpoint :)")
            print(f"Train start from epoch {start_epoch} epoch with accuracy {acc} and best accuracy {best_acc} :)")
            print()
            return model, optimizer, scheduler, start_epoch, acc, best_acc, train_hist, valid_hist
        except Exception as error:
            print(f"Something is wrong! :( ")
            print(error)
            exit()


def save(model: torch.nn.Module, acc: float, best_acc: float, optimizer: torch.optim.Optimizer,
         scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int,
         train_hist: dict, valid_hist: dict, name: str, log_dir: str) -> None:
    """
    save model and others :)
    :param model: model
    :param acc: last accuracy
    :param best_acc: best archived pose error
    :param optimizer: optimizer
    :param scheduler: learning rate scheduler
    :param epoch: last epoch
    :param train_hist: training accuracy and loss
    :param valid_hist: validation accuracy and loss
    :param name: experiment name
    :param log_dir: log directory
    """
    state = {
        'model': model.state_dict(),
        'acc': acc,
        'best_acc': best_acc,
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_hist': train_hist,
        'valid_hist': valid_hist,
    }
    torch.save(state, os.path.join(log_dir, f"checkpoint/{name}.pth"))


def angle_between(rot1: torch.Tensor, rot2: torch.Tensor):
    r"""
    Calculate the angle in radians between two rotations. (torch, batch)

    :param rot1: Rotation tensor 1 that can reshape to [batch_size, rep_dim].
    :param rot2: Rotation tensor 2 that can reshape to [batch_size, rep_dim].
    :return: Tensor in shape [batch_size] for angles in radians.
    """
    rot1 = r6d2rot_mat(rot1).float()
    rot2 = r6d2rot_mat(rot2).float()
    offsets = rot1.transpose(-1, -2).matmul(rot2)
    angles = roma.rotmat_to_rotvec(offsets).norm(dim=-1) * 180 / np.pi
    return angles.mean()


class AverageMeter:
    """
    save & calculate metric average :)
    """

    def __init__(self) -> None:
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self) -> None:
        """
        reset values :)
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        """
        update average :)
        :param val: metric value
        :param n: number of values
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
