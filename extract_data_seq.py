import glob
import math
import os

import numpy as np
import tqdm

from configs import ModelConfig


def extract_seq(phase, data_path, seq_length, overlap=0):
    """
    Extract sequences from data of length seq_length from all data files in phase directory.

    :param phase: train or valid
    :param data_path: path to data
    :param seq_length: length of sequence
    :param overlap: overlap between sequences
    """
    data_dir = os.path.join(data_path, phase, "*/*/*.npz")
    data_dirs = glob.glob(data_dir)
    loop = tqdm.tqdm(data_dirs)
    for data_dir in loop:

        data = np.load(data_dir)
        length = data['pose'].shape[0]
        if length < seq_length:
            continue

        loop.set_description(f"Extracting sequences from {data_dir} with seq length {seq_length}")

        target_folder = os.path.join(data_path, f"{phase}_seq", f"seq_{seq_length}")
        os.makedirs(target_folder, exist_ok=True)
        num_sequences = int(math.ceil((length - overlap) / (seq_length - overlap)))
        for idx in range(num_sequences):
            start_index = idx * (seq_length - overlap) + 1
            end_index = start_index + seq_length
            len_data = data['pose'][start_index:end_index].shape[0]
            if len_data == seq_length:
                targets = {
                    'grot': data['grot'][start_index: end_index],
                    'trans': data['trans'][start_index: end_index],
                    'jvel': data['jvel'][start_index: end_index],
                    'javel': data['javel'][start_index: end_index],
                    'joint': data['joint'][start_index: end_index],
                    'ljoint': data['ljoint'][start_index: end_index],
                    'last_trans': data['trans'][start_index - 1],
                    'last_jvel': data['jvel'][start_index - 1],
                    'last_javel': data['javel'][start_index - 1],
                    'last_grot': data['grot'][start_index - 1],
                    'last_joint': data['joint'][start_index - 1],
                    'last_ljoint': data['ljoint'][start_index - 1],
                }

                if "imu_acc" and "imu_ori" in data.keys():
                    targets['imu_acc'] = data['imu_acc'][start_index: end_index]
                    targets['imu_ori'] = data['imu_ori'][start_index: end_index]

                else:
                    targets['imu_acc'] = data['vacc'][start_index: end_index]
                    targets['imu_ori'] = data['vrot'][start_index: end_index]

                name = [f"Seq_{idx + 1}"] + data_dir.split("\\")[-3:-1] + [
                    os.path.basename(data_dir).replace(".pt", "")]
                name = "_".join(name)

                np.savez(os.path.join(target_folder, name), **targets)


if __name__ == '__main__':
    extract_seq(data_path=ModelConfig.data_dir, phase="Train",
                seq_length=ModelConfig.time_window, overlap=0)
