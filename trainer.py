import os

import matplotlib.pyplot as plt
import tensorboardX as tb
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchmetrics import R2Score, MeanAbsoluteError

from configs import ModelConfig
from model_utils import rot_mat2r6d
from train_utils import save, resume, get_lr, to_device, set_seed, angle_between, AverageMeter


class Trainer:
    """
    A class to train STIPoser model :)
    """

    def __init__(self, dataloaders: dict, model: nn.Module, transforms: dict,
                 criterion: nn.Module, optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler.LRScheduler,
                 configs: ModelConfig) -> None:
        """
        :param dataloaders: train and validation dataloaders :)
        :param model: model to train :)
        :param transforms: data augmentations :)
        :param criterion: loss function :)
        :param optimizer: model optimizer :)
        :param scheduler: learning rate scheduler :)
        :param configs: model configurations :)
        """

        self.model = model.to(configs.device)

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.transforms = transforms

        self.num_epochs = configs.epochs
        self.dataloaders = dataloaders
        self.device = configs.device
        self.aux_weights = configs.aux_weights
        self.log_dir = configs.log_dir
        self.name = configs.exp_name
        self.grad_norm = configs.grad_norm

        self.metric_train_p = R2Score(num_outputs=6).to(self.device)
        self.metric_valid_p = R2Score(num_outputs=6).to(self.device)

        self.metric_trans = MeanAbsoluteError().to(self.device)
        self.metric_pose = AverageMeter()

        self.loss_train = AverageMeter()
        self.aux_loss = AverageMeter()

        self.train_hist = {
            "loss": [],
            "acc": [],
        }

        self.loss_valid = AverageMeter()
        self.valid_hist = {
            "loss": [],
            "acc": [],
        }

        if configs.resume:
            (self.model, self.optimizer, self.scheduler,
             self.start_epoch, _, self.best_acc, self.train_hist,
             self.valid_hist) = resume(model=self.model, optimizer=self.optimizer,
                                       scheduler=self.scheduler, configs=configs)
        else:
            self.start_epoch = 0
            self.best_acc = 120

        self.writer = tb.SummaryWriter(f'./train_log/{configs.exp_name}/tensorboard')

    def one_step(self, datas: dict, training: bool = True) -> None:
        """
        one training step :)
        :param datas: data dict :)
        :param training: is training :)
        """
        tar_rot = rot_mat2r6d(datas["grot"])[:, :, [1, 2, 16, 17]].reshape(-1, 6)
        tar_trn = datas["trans"].reshape(-1, 3)

        if training:

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model(datas)
                main_loss, aux_loss = self.criterion(outputs, datas)
                loss = main_loss + (self.aux_weights * aux_loss)

            per_rot = outputs["out_rot"][:, :, [0, 1, 8, 9]].reshape(-1, 6)

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

            self.optimizer.step()

            torch.cuda.synchronize()

            self.metric_train_p.update(per_rot, tar_rot)

            self.loss_train.update(main_loss)
            self.aux_loss.update(self.aux_weights * aux_loss)

        else:
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                per_rot, per_trn = self.model.forward_offline(imu_acc=datas["imu_acc"][0], imu_ori=datas["imu_ori"][0],
                                                              return_6d=True)

            per_rot = per_rot[:, [0, 1, 8, 9]].reshape(-1, 6)
            per_trn = per_trn.reshape(-1, 3)
            self.metric_valid_p.update(per_rot, tar_rot)

            self.metric_pose.update(angle_between(per_rot, tar_rot))
            self.metric_trans.update(per_trn, tar_trn)

            main_loss = torch.nn.functional.smooth_l1_loss(per_rot, tar_rot, beta=0.01)
            main_loss += torch.nn.functional.smooth_l1_loss(per_trn, tar_trn, beta=0.01)

            self.loss_valid.update(main_loss)

    def train_one_epoch(self, epoch: int) -> tuple[float, torch.Tensor]:
        """
        one training epoch :)
        :param epoch: number of epoch :)
        :return: train loss and train acc
        """
        self.model.train()

        self.loss_train.reset()
        self.aux_loss.reset()

        self.metric_train_p.reset()

        with tqdm.tqdm(self.dataloaders["train"], unit='batch') as tepoch:
            for datas in tepoch:
                datas = self.transforms["train"](datas)
                datas = to_device(data=datas, device=self.device)
                self.one_step(datas, training=True)
                tepoch.set_postfix(epoch=epoch, loss=self.loss_train.avg.item(), aux_loss=float(self.aux_loss.avg),
                                   phase="Training")
            if self.scheduler is not None:
                self.scheduler.step()

        return self.loss_train.avg.item(), self.metric_train_p.compute()

    def valid_one_epoch(self) -> tuple[float, torch.Tensor, float, float]:
        self.model.eval()
        self.loss_valid.reset()

        self.metric_valid_p.reset()

        self.metric_pose.reset()
        self.metric_trans.reset()

        with tqdm.tqdm(self.dataloaders["valid"], unit='batch') as tepoch:
            with torch.inference_mode():
                for datas in tepoch:
                    datas = to_device(data=datas, device=self.device)
                    self.one_step(datas, training=False)
                    tepoch.set_postfix(loss=self.loss_valid.avg.item(), rot_error=self.metric_pose.avg.item(),
                                       phase="Evaluation")

        return (self.loss_valid.avg.item(), self.metric_valid_p.compute(), self.metric_pose.avg.item(),
                self.metric_trans.compute().mean().item())

    def train(self) -> None:
        """
        training process :)
        """
        for epoch in range(self.start_epoch, self.num_epochs):
            set_seed(epoch)
            t_loss, t_acc = self.train_one_epoch(epoch)
            v_loss, v_acc, v_pose, v_tran = self.valid_one_epoch()

            self.train_hist["loss"].append(t_loss)
            self.valid_hist["loss"].append(v_loss)
            self.train_hist["acc"].append(t_acc.detach().cpu().item())
            self.valid_hist["acc"].append(v_acc.detach().cpu().item())

            print(
                f"Epoch: {epoch}, Train Loss: {t_loss:.4}, Train Acc: {t_acc:.4}, Valid Loss: {v_loss:.4},",
                f"Valid Acc: {v_acc:.4}, Valid Pose: {v_pose:.4}, Valid Trans: {v_tran:.4}")

            if v_pose < self.best_acc:
                self.best_acc = v_pose
                save(model=self.model, acc=v_acc, best_acc=self.best_acc, optimizer=self.optimizer,
                     scheduler=self.scheduler, epoch=epoch, train_hist=self.train_hist,
                     valid_hist=self.valid_hist, name=f"best_{self.name}", log_dir=self.log_dir)

                print('Model Saved!')

            save(model=self.model, best_acc=self.best_acc, acc=v_acc, optimizer=self.optimizer,
                 scheduler=self.scheduler, epoch=epoch, train_hist=self.train_hist,
                 valid_hist=self.valid_hist, name=f"last_{self.name}", log_dir=self.log_dir)

            self.writer.add_scalar('Loss/train', self.train_hist["loss"][-1], epoch, walltime=epoch,
                                   display_name="Training Loss", )
            self.writer.add_scalar('Metric/train', self.train_hist["acc"][-1], epoch, walltime=epoch,
                                   display_name="Training Metric", )
            self.writer.add_scalar('LR/train', get_lr(self.optimizer), epoch, walltime=epoch,
                                   display_name="Learning rate", )
            self.writer.add_scalar('Loss/valid', self.valid_hist["loss"][-1], epoch, walltime=epoch,
                                   display_name="Validation Loss", )
            self.writer.add_scalar('Metric/valid', self.valid_hist["acc"][-1], epoch, walltime=epoch,
                                   display_name="Validation Metric", )
            print()

            if epoch == 0:
                write_mode = "w"
            else:
                write_mode = "a"
            log_path = os.path.join(self.log_dir, f"log.txt")
            with open(log_path, write_mode) as f:
                f.write(f"Epoch: {epoch},"
                        f" Train acc: {self.train_hist['acc'][-1]}, Train loss: {self.train_hist['loss'][-1]},"
                        f" Valid acc: {self.valid_hist['acc'][-1]}, Valid loss: {self.valid_hist['loss'][-1]},"
                        f" Valid pose: {v_pose}\n")

            plt.clf()
            plt.plot(self.train_hist["loss"], label="Train Loss")
            plt.plot(self.valid_hist["loss"], label="Valid Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f"loss_log.png"))

            plt.clf()
            plt.plot(self.train_hist["acc"], label="Train Metric")
            plt.plot(self.valid_hist["acc"], label="Valid Metric")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f"metric_log.png"))
