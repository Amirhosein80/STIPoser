import os
import glob
import numpy as np
import gc
import tqdm

import torch
import roma

from model import STIPoser
from configs import ModelConfig
from smpl_model import ParametricModel


def resume(model: torch.nn.Module, path) -> torch.nn.Module:
    """
    Load model weights :)
    :param model: model :)
    :param path: weights path :)
    :return: model
    """
    loaded = torch.load(path)
    model.load_state_dict(loaded["model"])

    acc = loaded["acc"]
    best_acc = loaded["best_acc"]
    print(f"Load all parameters from last checkpoint :)")
    print(f" accuracy {acc} and best accuracy {best_acc} :)")
    print()
    return model


def load_data(path, device) -> dict[str, torch.Tensor]:
    """
    Load data to device :)
    :param path: data path :)
    :param device: device :)
    :return: model
    """
    data = dict(np.load(path))
    for key in data.keys():
        data[key] = torch.from_numpy(data[key]).float().to(device)
    return data


def joint_distance(joint1: torch.Tensor, joint2: torch.Tensor) -> torch.Tensor:
    """
    calculate distance between prediction joints and ground truth :)
    :param joint1: joint positions:)
    :param joint2: joint positions :)
    :return: distance (cm)
    """
    offset_from_p_to_t = (joint2[:, 0] - joint1[:, 0]).unsqueeze(1)
    je = (joint1 + offset_from_p_to_t - joint2).norm(dim=2)
    return je.mean() * 100


def jitter_error(joint1: torch.Tensor, fps: float = 60.) -> torch.Tensor:
    """
    calculate jitter error for prediction joints : )
    :param joint1: prediction joints :)
    :param fps: data frequency :)
    :return: jitter error (km/s^3)
    """
    je = joint1[3:] - 3 * joint1[2:-1] + 3 * joint1[1:-2] - joint1[:-3]
    jitter = (je * (fps ** 3)).norm(dim=2)
    return jitter.mean() / 1000


def angle_between(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """
    calculate angle between prediction joints and ground truth :)
    :param rot1: prediction joints :)
    :param rot2: ground truth :)
    :return: angular error (deg)
    """
    offsets = rot1.transpose(-1, -2).matmul(rot2)
    angles = roma.rotmat_to_rotvec(offsets).norm(dim=-1) * 180 / np.pi
    return angles.mean()


def trans_error(trans1, trans2, end_index) -> torch.Tensor:
    """
    calculate translation error for prediction and  ground truth :)
    :param trans1: prediction translation :)
    :param trans2: ground truth translation :)
    :param end_index: end index :)
    :return: translation error (cm)
    """
    te = (trans1[:end_index] - trans2[:end_index]).norm(dim=1)
    return te.mean() * 100


class PoseEvaluator:
    """
    Pose estimation evaluator class :)
    This class calculates pose estimation metrics (SIP, Angular error, Joint Distance,
                                                   Jitter Error, Translation Errors) :)
    """
    def __init__(self, model: torch.nn.Module, data_files, configs: ModelConfig) -> None:
        """
        :param model: Pose estimation model :)
        :param data_files: data paths :)
        :param configs: model config :)
        """
        self.model = model
        self.device = configs.device
        self.sip_idx = [1, 2, 16, 17]
        self.ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])

        assert len(data_files) != 0

        self.data_files = data_files

        self.pose_errors = []

    def run(self) -> None:
        """
        calculate pose estimation metrics for each data in dataset :)
        """
        loop = tqdm.tqdm(self.data_files)
        for data_file in loop:
            self.eval(data_file)

        self.pose_errors = torch.stack(self.pose_errors, dim=0)
        print(torch.argsort(self.pose_errors[:, 0]))

        print()
        print("=" * 100)
        print()
        print("Model Result\n")
        mean, std = self.pose_errors.mean(dim=0), self.pose_errors.std(dim=0)
        print(f"SIP Error (deg): {(mean[0].item(), std[0].item())}, Ang Error (deg): {(mean[1].item(), std[1].item())}")
        print(f"Joint Error (cm): {(mean[2].item(), std[2].item())}, Jitter Error (km/s^3): {(mean[3].item(), std[3].item())}")
        print(
            f"Translation 2s Error (cm): {(mean[4].item(), std[4].item())}, "
            f"Translation 5s Error (cm): {(mean[5].item(), std[5].item())}, "
            f"Translation 10s Error (cm): {(mean[6].item(), std[6].item())} "
            f"Translation All Error (cm): {(mean[7].item(), std[7].item())}"
        )

        print()
        print("=" * 100)
        print()

    def eval(self, file) -> None:
        """
        calculate pose estimation metrics for dataset :)
        :param file: data paths :)
        """
        self.model.eval()
        datas = load_data(file, device=self.device)

        t_rotation = datas["pose"]
        t_trans = datas["trans"]

        t_rotation[:, self.ignored_joint_mask] = torch.eye(3, device=t_rotation.device)
        global_t, joint_t, _, _ = self.model.smpl_model.forward_kinematics(pose=t_rotation, tran=t_trans)

        if "imu_acc" in datas.keys():
            imu_acc = datas["imu_acc"]
            imu_ori = datas["imu_ori"]

        else:
            imu_acc = datas["vacc"]
            imu_ori = datas["vrot"]

        with torch.inference_mode():
            preds1 = self.model.forward_offline(imu_acc=imu_acc, imu_ori=imu_ori)
            p_rotation = preds1[0].detach()
            p_trans = preds1[1].detach()

            p_rotation[:, self.ignored_joint_mask] = torch.eye(3, device=p_rotation.device)
            global_p, joint_p, _, _ = self.model.smpl_model.forward_kinematics(p_rotation, tran=p_trans)

            metrics = [
                angle_between(global_p.detach()[10:, self.sip_idx], global_t[10:, self.sip_idx]),
                angle_between(global_p.detach()[10:], global_t[10:]),
                joint_distance(joint_p[10:], joint_t[10:]),
                jitter_error(joint_p[10:]),

                trans_error(p_trans.detach(), t_trans, end_index=60 * 2),
                trans_error(p_trans.detach(), t_trans, end_index=60 * 5),
                trans_error(p_trans.detach(), t_trans, end_index=60 * 10),
                trans_error(p_trans.detach(), t_trans, end_index=-1),
            ]
            self.pose_errors.append(torch.tensor(metrics))

            del preds1, p_rotation, joint_p, global_p, metrics
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    VALID_DIR = r"./data/Test"
    tc_files = glob.glob(os.path.join(VALID_DIR, "DIP*/*/*.npz"))
    model_config = ModelConfig()
    model_config.device = "cuda"

    smpl = ParametricModel(official_model_file=model_config.smpl_dir, device=model_config.device)

    model = STIPoser(configs=model_config, smpl_model=smpl)
    model = resume(model=model, path=r"train_log/model_final_SA2/checkpoint/best_model_final_SA2.pth")

    evaluator = PoseEvaluator(model=model, data_files=tc_files, configs=model_config)
    evaluator.run()
