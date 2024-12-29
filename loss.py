import torch
import torch.nn as nn

from configs import ModelConfig
from model_utils import rot_mat2r6d


class MotionLoss(nn.Module):
    """
    Motion loss function :)
    This loss calculates rotation error and translation error :)
    """
    def __init__(self, configs: ModelConfig) -> None:
        """
        :param configs: model configurations :)
        """
        super().__init__()
        self.configs = configs
        self.mae = nn.SmoothL1Loss(beta=0.01)

    def forward(self, predicts: dict, targets: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate losses :)
        :param predicts: model predictions :)
        :param targets: ground truth :)
        :return: main losses and auxiliary losses :
        """
        main_loss = 0
        aux_loss = 0

        for key, val in self.configs.loss_slices.items():
            for idx in range(len(val["targets"])):

                inds = val["index"][idx]
                tars = val["targets"][idx]

                if tars == "grot":
                    label = rot_mat2r6d(targets[tars])[:, :, inds]
                elif tars == "trans":
                    label = targets[tars]
                else:
                    label = targets[tars][:, :, inds]

                if key == "out_rot":
                    main_loss += self.mae(predicts[key], label)
                elif key == "out_trn":
                    main_loss += self.mae(predicts[key], label)
                else:
                    aux_loss += self.mae(predicts[key].reshape(-1, 3), label.reshape(-1, 3))

        return main_loss, aux_loss
