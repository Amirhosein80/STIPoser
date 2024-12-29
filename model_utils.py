import torch
import torch.nn as nn
import torch.nn.functional as f
import time


def rot_mat2r6d(rot_mat: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrix to 6D format :)
    :param rot_mat: rotation matrix :)
    :return: 6D rotation
    """
    r = rot_mat[..., :2].transpose(-1, -2)
    return r.reshape(*rot_mat.shape[:-2], 6)


def r6d2rot_mat(r6d: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation to rotation matrix :)
    :param r6d: 6D rotation :)
    :return: rotation matrix
    """
    x_raw = r6d[..., 0:3]
    y_raw = r6d[..., 3:6]

    x = f.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = f.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack((x, y, z), dim=-1)
    return matrix


def low_pass_filter(x: torch.Tensor, cutoff_freq: int = 5, sampling_rate: float = 60.) -> torch.Tensor:
    """
    low pass filter for noise :)
    :param x: input signal :)
    :param cutoff_freq: cutoff frequency for low pass filter :)
    :param sampling_rate: data sampling rate for low pass filter :)
    :return: filtered signal
    """
    fft_tensor = torch.fft.fft(x.float(), dim=1)
    freqs = torch.fft.fftfreq(x.size(1), d=1 / sampling_rate).to(x.device)

    low_pass_mask = (torch.abs(freqs) <= cutoff_freq).float()
    low_pass_mask = low_pass_mask.view(1, -1, 1, 1)

    filtered_fft = fft_tensor * low_pass_mask

    filtered_tensor = torch.fft.ifft(filtered_fft, dim=1).real

    return filtered_tensor.to(x.dtype).to(x.device)


def run_benchmark(model: nn.Module, datas: list[torch.Tensor]) -> None:
    """
    calculate model latency and FPS :)
    :param model: Pose estimation model :)
    :param datas: input datas [imu_acc, imu_ori] :)
    """
    elapsed = 0
    model.eval()
    num_batches = 100
    print("Start benchmarking...")
    with torch.inference_mode():
        for _ in range(10):
            model(*datas)
        for i in range(100):
            if i < num_batches:
                start = time.time()
                _ = model(*datas)
                end = time.time()
                elapsed = elapsed + (end - start)
            else:
                break
    num_images = 100
    latency = elapsed / num_images * 1000
    fps = 1000 / latency
    print(f'Elapsed time: {latency:.3} ms, FPS: {fps:.3}')


if __name__ == '__main__':
    pass
