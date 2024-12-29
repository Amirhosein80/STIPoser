import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from dataset_utils import collate_fn
from configs import ModelConfig


class CustomDataset(Dataset):
    r"""
    Custom dataset for TSPose Model.
    """

    def __init__(self, configs: ModelConfig, phase: str):
        r"""
        :param phase: dataset for Train or Valid
        :param configs: configs
        """
        assert phase in ["Train",
                         "Valid"], "You should select phase between Train and Valid"

        if phase == "Train":
            self.files = glob.glob(
                os.path.join(configs.data_dir, f"{phase}_seq", f"seq_{configs.time_window}", "*.npz"))
        elif phase == "Valid":
            self.files = glob.glob(os.path.join(configs.data_dir, f"{phase}", f"*/*", "*.npz"))

        self.phase = phase
        self.dir = configs.data_dir
        self.data_keys = ["imu_acc", "imu_ori", "grot", "jvel", "trans",
                          "last_jvel", "last_trans"]

        self.batch_size = configs.batch_size
        self.num_worker = configs.num_worker

    def __len__(self):
        r"""
        :return number of data in dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        r"""
        load data
        :param idx: data index
        :return: data
        """
        file = self.files[idx]
        data = dict(np.load(file))

        x = {}

        for key, value in data.items():
            if key in self.data_keys:
                x[key] = torch.from_numpy(value).float()

        if self.phase == "Valid":
            if "imu_acc" not in x.keys():
                x['imu_acc'] = torch.from_numpy(data['vacc']).float()
                x['imu_ori'] = torch.from_numpy(data['vrot']).float()

        return x

    def get_data_loader(self):
        r"""
        get data loader for dataset
        :return: data loader
        """
        if self.phase in ["Train", "All"]:
            sampler = RandomSampler(self)
            bs = self.batch_size
        else:
            sampler = SequentialSampler(self)
            bs = 1
        return DataLoader(self, batch_size=bs, sampler=sampler,
                          num_workers=self.num_worker, collate_fn=collate_fn,
                          pin_memory=True)


if __name__ == '__main__':
    ds = CustomDataset(configs=ModelConfig(), phase="Train")
