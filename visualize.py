import gc
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer

import smpl_model
from configs import ModelConfig
from model import STIPoser


def resume(model: torch.nn.Module, path):
    loaded = torch.load(path)
    model.load_state_dict(loaded["model"])

    acc = loaded["acc"]
    best_acc = loaded["best_acc"]
    print(f"Load all parameters from last checkpoint :)")
    print(f" accuracy {acc} and best accuracy {best_acc} :)")
    print()
    return model


def load_data(path, device):
    data = dict(np.load(path))
    for key in data.keys():
        data[key] = torch.from_numpy(data[key]).to(device)
    return data


def plot_trans_grid(predict, target):
    fig, subplots = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    subplots.plot(predict[:, 0] * 100, predict[:, 2] * 100, color="red", label="predict")
    subplots.plot(target[:, 0] * 100, target[:, 2] * 100, color="blue", label="real")
    subplots.set_title(f"Translation X/Z")
    subplots.set_xlabel('Translation X (cm)')
    subplots.set_ylabel('Translation Z (cm)')
    subplots.grid(True)
    subplots.legend(loc=0)
    plt.show()

    gc.collect()


def plot_trans(predict, target):
    xaxis = np.linspace(0, predict.shape[0] / 60, predict.shape[0])

    fig, subplots = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))
    subplots[0].plot(xaxis, predict[:, 0], color="red", label="predict")
    subplots[0].plot(xaxis, target[:, 0], color="blue", label="real")
    subplots[0].set_title(f"Translation X")
    subplots[0].set_xlabel('Time (s)')
    subplots[0].set_ylabel('Rotation (rad)')
    subplots[0].grid(True)
    subplots[0].legend(loc=2)

    subplots[1].plot(xaxis, predict[:, 1], color="red", label="predict")
    subplots[1].plot(xaxis, target[:, 1], color="blue", label="real")
    subplots[1].set_title(f"Translation Y")
    subplots[1].set_xlabel('Time (s)')
    subplots[1].set_ylabel('Rotation (rad)')
    subplots[1].grid(True)
    subplots[1].legend(loc=2)

    subplots[2].plot(xaxis, predict[:, 2], color="red", label="predict")
    subplots[2].plot(xaxis, target[:, 2], color="blue", label="real")
    subplots[2].set_title(f"Translation Z")
    subplots[2].set_xlabel('Time (s)')
    subplots[2].set_ylabel('Rotation (rad)')
    subplots[2].grid(True)
    subplots[2].legend(loc=2)
    plt.show()

    gc.collect()


def visualize(model: torch.nn.Module, data_file: list[str]):
    imu_acc = load_data(data_file, device=model_config.device)["imu_acc"][:-10]
    imu_ori = load_data(data_file, device=model_config.device)["imu_ori"][:-10]

    target = load_data(data_file, device=model_config.device)["pose"].to(model_config.device)[:-10]
    t_trans = load_data(data_file, device=model_config.device)["trans"].to(model_config.device)[:-10]

    model.eval()
    with torch.inference_mode():
        preds = model.forward_offline(imu_acc=imu_acc, imu_ori=imu_ori)
        p_ori = preds[0].detach()
        p_trn = preds[1].detach()
        del preds
    torch.cuda.empty_cache()
    gc.collect()

    plot_trans_grid(p_trn, t_trans)
    plot_trans(p_trn, t_trans)
    import roma
    offsets = p_ori.transpose(-1, -2).matmul(target)
    angles = roma.rotmat_to_rotvec(offsets).norm(dim=-1) * 180 / np.pi
    print(angles.mean(), angles.std())

    vp, fp, jp = smpl.view_motion(pose_list=[p_ori], tran_list=[p_trn])
    vt, ft, jt = smpl.view_motion(pose_list=[target], tran_list=[t_trans])

    body_mesh = Meshes(
        vp,
        fp,
        is_selectable=False,
        gui_affine=False,
        name="Predicted Body Mesh",
        color=(240 / 255, 120 / 255, 120 / 255, 0.8)
    )

    gt_mesh = Meshes(
        vt,
        ft,
        is_selectable=False,
        gui_affine=False,
        name="Ground Truth Body Mesh",
        color=(120 / 255, 120 / 255, 240 / 255, 0.8)
    )

    v = Viewer()
    v.scene.add(body_mesh)
    v.scene.add(gt_mesh)
    v.run()

if __name__ == "__main__":
    VALID_DIR = r"./data/Valid"
    tc_files = glob.glob(os.path.join(VALID_DIR, "TC*/*/*.npz"))
    model_config = ModelConfig()
    model_config.device = "cpu"

    smpl = smpl_model.ParametricModel(official_model_file=model_config.smpl_dir,
                                      device=model_config.device)

    model = STIPoser(configs=model_config, smpl_model=smpl)
    model = resume(model=model, path=r"train_log/model_final_SA2/checkpoint/best_model_final_SA2.pth")
    visualize(model, data_file=tc_files[5])
