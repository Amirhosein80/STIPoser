import random
import numpy as np
import torch

class Compose:
    """
    Sequential transforms for both image & mask :)
    """

    def __init__(self, transforms):
        r"""
        :param transforms: List of transforms
        """
        self.transforms = transforms

    def __call__(self, samples):
        r"""
        :param samples: a dict of data.
        """
        for transform in self.transforms:
            samples = transform(samples)
        return samples

class StandardScaler:
    r"""
    Normalize by mean and std.
    """
    def __init__(self, scales, list_keys):
        r"""
        :param scales: a dict of mean and std for each data keys.
        """
        self.scales = scales
        self.list_keys = list_keys

    def __call__(self, samples):
        r"""
        :param samples: a dict of data.
        :param list_keys: list of keys for transform.
        """
        for key in self.list_keys:
            if key == "last_ljoint":
                samples[key] = (samples[key] - self.scales["ljoint"]["mean"].unsqueeze(0)) / self.scales["ljoint"]["std"].unsqueeze(0)
            else:
                samples[key] = (samples[key] - self.scales[key]["mean"].unsqueeze(0).unsqueeze(0)) / self.scales[key]["std"].unsqueeze(0).unsqueeze(0)
        return samples
    

class RandomResample:
    r"""
    Convert data from 60Hz to another frequency.
    """
    def __init__(self, target_fps_list, list_keys, p=0.5):
        r"""
        :param target_fps_list: a list of target frequncy.
        :param p: probablity to convert.
        """
        self.p = p
        self.target_fps_list = target_fps_list
        self.list_keys = list_keys

    def __call__(self, samples):
        r"""
        :param samples: a dict of data.
        :param list_keys: list of keys for transform.
        """
        if random.random() < self.p:
            target_fps = np.random.choice(self.target_fps_list)
            for key in self.list_keys:
                indices = torch.arange(0, samples[key].shape[1], 60 // target_fps)

                start_indices = torch.floor(indices).long()
                end_indices = torch.ceil(indices).long()
                end_indices[end_indices >= samples[key].shape[1]] = samples[key].shape[1] - 1  # handling edge cases

                start = samples[key][:, start_indices]
                end = samples[key][:, end_indices]

                floats = indices - start_indices
                floats = floats.unsqueeze(0)
                for shape_index in range(len(samples[key].shape) - 2):
                    floats = floats.unsqueeze(-1)
                weights = torch.ones_like(start) * floats
                samples[key] = torch.lerp(start, end, weights)

        return samples


class UniformNoise:
    r"""
    Add unifrom noise to data.
    """
    def __init__(self, noise=0.1, p=0.5):
        r"""
        :param noise: noise domain.
        :param p: probablity to convert.
        """
        self.p = p
        self.noise = noise

    def __call__(self, samples):
        r"""
        :param samples: a dict of data.
        """
        if random.random() < self.p:
            noise = torch.zeros_like(samples["imu_acc"]).uniform_(-self.noise, self.noise)
            samples["imu_acc"] += noise
        return samples

