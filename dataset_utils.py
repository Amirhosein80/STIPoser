import os
import tqdm
import torch
import gc


def collate_fn(batch) -> dict[str, torch.Tensor]:
    r"""
    stack data to create batch.
    :param batch: list of data.
    :return: dict contain batch of datas.
    """
    keys = batch[0].keys()
    out = {}
    for i, k in enumerate(keys):
        out[k] = torch.cat([b[k].unsqueeze(0) for b in batch], dim=0)
    return out


