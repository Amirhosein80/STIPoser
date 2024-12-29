import torch
import torch.optim as optim
import torch.amp as amp

from configs import ModelConfig


def get_optimizer(model: torch.nn.Module, configs: ModelConfig) \
        -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """
    load optimizer + scheduler for model :)
    :param model: model :)
    :param configs: model configuration :)

    :return: optimizer, lr_scheduler
    """
    # select optimizer
    if hasattr(model, "get_params"):
        params = model.get_params(lr=configs.lr,
                                  weight_decay=configs.weight_decay if not configs.overfit_test else 0)
    else:
        params = model.parameters()

    if configs.optimizer == "ADAMW":
        optimizer = optim.AdamW(params, lr=configs.lr,
                                weight_decay=configs.weight_decay if not configs.overfit_test else 0,
                                betas=configs.adamw_betas)

    elif configs.optimizer == "SGD":
        optimizer = optim.SGD(params, lr=configs.lr,
                              weight_decay=configs.weight_decay if not configs.overfit_test else 0,
                              momentum=configs.momentum)

    else:
        raise NotImplemented

    # add scheduler
    total_iters = configs.epochs - configs.warmup_epoch
    warm_iters = configs.warmup_epoch

    if configs.schedular == "POLY":
        main_lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters,
                                                            power=0.9)

    elif configs.schedular == "COS":
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=configs.min_lr)

    elif configs.schedular == "CONSTANT":
        main_lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, total_iters=total_iters, factor=1.0)

    else:
        raise NotImplemented

    # set warmup scheduler if you use
    if configs.warmup_epoch > 0 and not configs.overfit_test:
        warm_lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=configs.warmup_factor, total_iters=warm_iters)

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm_lr_scheduler, main_lr_scheduler], milestones=[warm_iters])

    else:
        scheduler = main_lr_scheduler

    return optimizer, scheduler
