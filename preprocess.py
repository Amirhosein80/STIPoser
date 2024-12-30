# codes from https://github.com/Xinyu-Yi/PIP.git and https://github.com/dx118/dynaip.git

import gc
import glob
import os
import pickle

import numpy as np
import roma
import torch
from tqdm import tqdm

from model_utils import low_pass_filter
from preprocess_utils import (_syn_acc, _syn_ang_vel, _syn_vel, IG_JOINTS, read_mvnx,
                              read_xlsx, J_MASK, V_MASK, AMASS_ROT,
                              AMASS_TRAIN_SPLIT, AMASS_VALID_SPLIT, AMASS_TEST_SPLIT,
                              DIP_IMU_MASK, DIP_TRAIN_SPLIT, DIP_TEST_SPLIT, DIP_VALID_SPLIT,
                              TC_IMU_MASK, TC_TRAIN_SPLIT, TC_VALID_SPLIT, TC_TEST_SPLIT,
                              XSENS_IMU_MASK, EM_TRAIN_SPLIT, EM_VALID_SPLIT, EM_TEST_SPLIT,
                              MTW_TRAIN_SPLIT, MTW_VALID_SPLIT, MTW_TEST_SPLIT)
from smpl_model import ParametricModel


def process_amass(smpl_path, amass_path) -> None:
    r"""
    Load Pose & Translation and Place Virtual IMU on AMASS datasets +
    Synthesize Joint Velocity, Joint Acceleration, Joint Angular Velocity, Joint Position

    :param smpl_path: path to smpl.pkl model.
    :param amass_path: folder containing AMASS datasets.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    amass_files = glob.glob(os.path.join(amass_path, '*/*/*_poses.npz'))
    loop = tqdm(amass_files)
    for file in loop:

        try:
            cdata = np.load(file)
            loop.set_description(f'processing {file}')
        except:
            continue

        if "mocap_framerate" not in cdata.keys():
            print(f'no mocap_framerate in {file}')
            discread_files.append(file)
            continue
        framerate = int(cdata['mocap_framerate'])
        if framerate == 120 or framerate == 100:
            step = 2
        elif framerate == 60 or framerate == 59 or framerate == 50:
            step = 1
        elif framerate == 250:
            step = 5
        elif framerate == 150:
            step = 3
        else:
            print(f'framerate {framerate} not supported in {os.path.basename(file)}')
            discread_files.append(file)
            continue
        poses = cdata['poses'][::step].astype(np.float32).reshape(-1, 52, 3)
        trans = cdata['trans'][::step].astype(np.float32)
        shape = torch.tensor(cdata['betas'][:10]).float().to("cuda")
        gender = cdata['gender']
        del cdata
        length = poses.shape[0]
        if length <= 10:
            discread_files.append(file)
            loop.set_description(
                f'discreaded file {os.path.basename(file)}, discrete num {len(discread_files)}, length {length}')
            continue
        poses[:, 23] = poses[:, 37]
        poses = poses[:, :24]

        poses = torch.tensor(poses, device="cuda")
        trans = torch.tensor(trans, device="cuda")

        poses[:, 0] = roma.rotmat_to_rotvec(AMASS_ROT.matmul(roma.rotvec_to_rotmat(poses[:, 0])))
        trans = AMASS_ROT.matmul(trans.unsqueeze(-1)).view_as(trans)
        rot_mat = roma.rotvec_to_rotmat(poses).view(-1, 24, 3, 3)
        rot_mat[:, IG_JOINTS] = torch.eye(3, device="cuda")
        trans = trans - trans[0:1]

        grot, joint, vert, ljoint = body_model.forward_kinematics(pose=rot_mat, tran=trans, calc_mesh=True, shape=shape)

        jvel = _syn_vel(joint, grot[:, 0:1])

        vacc = low_pass_filter(_syn_acc(vert[:, V_MASK]).unsqueeze(0)).squeeze(0)
        vrot = grot[:, J_MASK]
        javel = _syn_ang_vel(grot)

        folder = file.split("\\")[1]

        targets = {

            'pose': rot_mat.cpu().numpy(),
            'trans': trans.cpu().numpy(),
            'grot': grot.cpu().numpy(),

            'jvel': jvel.cpu().numpy(),

            'vacc': vacc.cpu().numpy(),
            'vrot': vrot.cpu().numpy(),

            'javel': javel.cpu().numpy(),
            "joint": joint.cpu().numpy(),
            "ljoint": ljoint.cpu().numpy(),

        }

        if folder in AMASS_TRAIN_SPLIT:
            split = "Train"
        elif folder in AMASS_VALID_SPLIT:
            split = "Valid"
        elif folder in AMASS_TEST_SPLIT:
            split = "Test"
        else:
            raise ValueError(f"Folder {folder} not recognized")

        targets_folder = file.split("AMASS")[0].replace("raw data", split)
        targets_folder = os.path.join(targets_folder, f"{folder}_process_data", file.split("\\")[2])
        os.makedirs(targets_folder, exist_ok=True)

        np.savez(os.path.join(targets_folder, os.path.basename(file).replace(".npz", "")), **targets)

        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del poses, trans, rot_mat, grot, joint, vert, targets, framerate
        torch.cuda.empty_cache()
        gc.collect()

    print("discread files: ", discread_files)
    print("AMASS process done!")


def process_dip(smpl_path, dip_path) -> None:
    r"""
    Load Pose and Place Virtual IMU on AMASS datasets +
    Synthesize Joint Velocity, Joint Acceleration, Joint Angular Velocity, Joint Position

    :param smpl_path: path to smpl.pkl model.
    :param dip_path: folder containing DIP datasets.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    dip_files = glob.glob(os.path.join(dip_path, '*/*.pkl'))
    loop = tqdm(dip_files)
    for file in loop:

        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        ori = torch.from_numpy(data['imu_ori']).float()[:, DIP_IMU_MASK].to("cuda")
        acc = torch.from_numpy(data['imu_acc']).float()[:, DIP_IMU_MASK].to("cuda")
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3).to("cuda")
        for _ in range(4):
            acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
            ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
            acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
            ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

        if torch.isnan(acc).sum() != 0 or torch.isnan(ori).sum() != 0 or torch.isnan(pose).sum() != 0:
            print(f'DIP-IMU: {file} has too much nan! Discard!')
            continue

        folder = file.split("\\")[1]

        trans = torch.zeros(pose.shape[0], 3)

        acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
        length = pose.shape[0]

        rot_mat = roma.rotvec_to_rotmat(pose).view(-1, 24, 3, 3)
        rot_mat[:, IG_JOINTS] = torch.eye(3, device="cuda")
        # trans = torch.zeros(rot_mat.shape[0], 3, device="cuda")
        trans = trans - trans[0:1]

        grot, joint, vert, ljoint = body_model.forward_kinematics(pose=rot_mat,
                                                                  tran=trans, calc_mesh=True)

        jvel = _syn_vel(joint, grot[:, 0:1])

        vacc = low_pass_filter(_syn_acc(vert[:, V_MASK]).unsqueeze(0)).squeeze(0)
        vrot = grot[:, J_MASK]
        javel = _syn_ang_vel(grot)

        joint[:, 1:] = joint[:, 1:] - joint[:, :1]

        targets = {
            'imu_acc': low_pass_filter(acc.unsqueeze(0)).squeeze(0).cpu().numpy(),
            'imu_ori': ori.cpu().numpy(),

            'pose': rot_mat.cpu().numpy(),
            'grot': grot.cpu().numpy(),
            'javel': javel.cpu().numpy(),

            "joint": joint.cpu().numpy(),
            "ljoint": ljoint.cpu().numpy(),

            "trans": trans.cpu().numpy(),
            "jvel": jvel.cpu().numpy(),
        }

        if folder in DIP_TRAIN_SPLIT:
            folder = "Train"
        elif folder in DIP_VALID_SPLIT:
            folder = "Valid"
        elif folder in DIP_TEST_SPLIT:
            folder = "Test"
        else:
            raise ValueError(f"Folder {folder} not recognized")
        target_folder = file.split(os.path.basename(file))[0].replace("DIP_IMU",
                                                                      "DIP_process_data").replace("raw data",
                                                                                                  folder)
        os.makedirs(target_folder,
                    exist_ok=True)

        np.savez(os.path.join(target_folder, os.path.basename(file).replace(".pkl", "")), **targets)

        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del pose, trans, rot_mat, grot, joint, vert, targets
        torch.cuda.empty_cache()
        gc.collect()


def process_total_capture(smpl_path, tc_path) -> None:
    r"""
    Load Pose & Translation and Place Virtual IMU on Total Capture dataset +
    Synthesize Joint Velocity, Joint Acceleration, Joint Angular Velocity, Joint Position

    :param smpl_path: path to smpl.pkl model.
    :param tc_path: folder containing TC datasets.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    tc_files = glob.glob(os.path.join(tc_path, '*/*.pkl'))
    loop = tqdm(tc_files)
    for file in loop:

        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, TC_IMU_MASK].to("cuda")
        acc = torch.from_numpy(data['acc']).float()[:, TC_IMU_MASK].to("cuda")
        # pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3).to("cuda")

        # if acc.shape[0] < pose.shape[0]:
        #     pose = pose[:acc.shape[0]]
        # elif acc.shape[0] > pose.shape[0]:
        #     acc = acc[:pose.shape[0]]
        #     ori = ori[:pose.shape[0]]
        # length = pose.shape[0]

        pose_path = file.split("TotalCapture_Real_60FPS")[0]
        pose_file = os.path.basename(file).split("_")
        subject = pose_file[0].upper()
        pose_file = os.path.join(pose_path, "TotalCapture",
                                 pose_file[0].lower(), f"{pose_file[1].split('.')[0]}_poses.npz")
        if not os.path.exists(pose_file):
            print(f"No trans file found for {file}")
            continue
        cdata = np.load(pose_file)
        framerate = int(cdata['mocap_framerate'])
        if framerate == 120 or framerate == 100:
            step = 2
        elif framerate == 60 or framerate == 59 or framerate == 50:
            step = 1
        elif framerate == 250:
            step = 5
        elif framerate == 150:
            step = 3
        else:
            print(f'framerate {framerate} not supported in {os.path.basename(file)}')
            discread_files.append(file)
            continue

        trans = cdata['trans'][::step].astype(np.float32)
        shape = torch.tensor(cdata['betas'][:10]).float().to("cuda")
        trans = torch.tensor(trans).to("cuda")
        trans = trans - trans[0:1]

        pose = cdata['poses'][::step].astype(np.float32).reshape(-1, 52, 3)
        pose[:, 23] = pose[:, 37]
        pose = pose[:, :24]
        pose = torch.tensor(pose, device="cuda")

        pose[:, 0] = roma.rotmat_to_rotvec(AMASS_ROT.matmul(roma.rotvec_to_rotmat(pose[:, 0])))

        trans = AMASS_ROT.matmul(trans.unsqueeze(-1)).view_as(trans)

        if acc.shape[0] < trans.shape[0]:
            trans = trans[:acc.shape[0]]
            pose = pose[:acc.shape[0]]

        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert trans.shape[0] == acc.shape[0]

        length = acc.shape[0]

        rot_mat = roma.rotvec_to_rotmat(pose).view(-1, 24, 3, 3)

        rot_mat[:, IG_JOINTS] = torch.eye(3, device="cuda")

        grot, joint, vert, ljoint = body_model.forward_kinematics(pose=rot_mat, tran=trans, shape=shape, calc_mesh=True)

        vacc = _syn_acc(vert[:, V_MASK])
        d = vacc.mean(dim=0, keepdim=True) - acc.mean(dim=0, keepdim=True)
        acc = acc + d

        jvel = _syn_vel(joint, grot[:, 0:1])

        vacc = low_pass_filter(_syn_acc(vert[:, V_MASK]).unsqueeze(0)).squeeze(0)
        vrot = grot[:, J_MASK]
        javel = _syn_ang_vel(grot)

        targets = {
            'imu_acc': low_pass_filter(acc.unsqueeze(0)).squeeze(0).cpu().numpy(),
            'imu_ori': ori.cpu().numpy(),

            'pose': rot_mat.cpu().numpy(),
            'trans': trans.cpu().numpy(),
            'grot': grot.cpu().numpy(),

            'jvel': jvel.cpu().numpy(),

            'vacc': vacc.cpu().numpy(),
            'vrot': vrot.cpu().numpy(),

            'javel': javel.cpu().numpy(),
            "joint": joint.cpu().numpy(),
            "ljoint": ljoint.cpu().numpy(),

        }

        if subject in TC_TRAIN_SPLIT:
            folder = "Train"
        elif subject in TC_VALID_SPLIT:
            folder = "Valid"
        elif subject in TC_TEST_SPLIT:
            folder = "Test"
        else:
            raise ValueError(f"Folder {subject} not recognized")

        target_folder = file.split("TotalCapture_Real_60FPS")[0].replace("TotalCapture", "TC_process_data").replace(
            "raw data", folder)
        os.makedirs(os.path.join(target_folder, subject), exist_ok=True)

        np.savez(os.path.join(target_folder, subject, os.path.basename(file).replace(".pkl", "")), **targets)

        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del pose, trans, rot_mat, grot, joint, vert, targets
        torch.cuda.empty_cache()
        gc.collect()


def process_emonike(smpl_path, emonike_path) -> None:
    r"""
    Load Pose & Translation and Place Virtual IMU on EMONIKE dataset +
    Synthesize Joint Velocity, Joint Acceleration, Joint Angular Velocity, Joint Position

    :param smpl_path: path to smpl.pkl model.
    :param emonike_path: folder containing EMONIKE dataset.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    em_files = glob.glob(os.path.join(emonike_path, '*.mvnx'))
    loop = tqdm(em_files)
    for file in loop:
        data = read_mvnx(file=file, smpl_file=smpl_path)
        pose = data['joint']['orientation'].to("cuda")
        trans = data['joint']['translation'].to("cuda")
        length = pose.shape[0]

        ori = data['imu']['calibrated orientation'][:, XSENS_IMU_MASK].to("cuda")
        acc = data['imu']['free acceleration'][:, XSENS_IMU_MASK].to("cuda")
        trans = trans - trans[0:1]

        grot, joint, vert, ljoint = body_model.forward_kinematics(pose=pose, tran=trans, calc_mesh=True)

        jvel = _syn_vel(joint, grot[:, 0:1])

        vacc = low_pass_filter(_syn_acc(vert[:, V_MASK]).unsqueeze(0)).squeeze(0)
        vrot = grot[:, J_MASK]
        javel = _syn_ang_vel(grot)

        targets = {
            'imu_acc': low_pass_filter(acc.unsqueeze(0)).squeeze(0).cpu().numpy(),
            'imu_ori': ori.cpu().numpy(),

            'pose': pose.cpu().numpy(),
            'trans': trans.cpu().numpy(),
            'grot': grot.cpu().numpy(),

            'jvel': jvel.cpu().numpy(),

            'vacc': vacc.cpu().numpy(),
            'vrot': vrot.cpu().numpy(),

            'javel': javel.cpu().numpy(),
            "joint": joint.cpu().numpy(),
            "ljoint": ljoint.cpu().numpy(),

        }

        seq = os.path.basename(file).split("_")[1]
        if seq in EM_TRAIN_SPLIT:
            folder = "Train"
        elif seq in EM_VALID_SPLIT:
            folder = "Valid"
        elif seq in EM_TEST_SPLIT:
            folder = "Test"
        else:
            raise ValueError(f"Folder {seq} not recognized")
        target_folder = file.split('EmokineDataset_v1.0')[0].replace("raw data", folder)
        target_folder = os.path.join(target_folder, "EM_process_data", seq)
        os.makedirs(target_folder,
                    exist_ok=True)

        np.savez(os.path.join(target_folder, os.path.basename(file).replace('.mvnx', '')), **targets)

        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del pose, trans, grot, joint, vert, targets, length
        torch.cuda.empty_cache()
        gc.collect()


def process_xsens(smpl_path, mvnx_path) -> None:
    r"""
    Load Pose & Translation and Place Virtual IMU on Xsens dataset +
    Synthesize Joint Velocity, Joint Acceleration, Joint Angular Velocity, Joint Position

    :param smpl_path: path to smpl.pkl model.
    :param mvnx_path: folder containing Xsens dataset.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    em_files = glob.glob(os.path.join(mvnx_path, "*/*/*/*/*.xlsx"))
    loop = tqdm(em_files)
    for file in loop:
        if "calibration" in file:
            loop.set_description(f"Calibration file {os.path.basename(file)}")
            continue
        if file.split("\\")[-2][0] == "_":
            loop.set_description(f"Bug file {os.path.basename(file)}")
            discread_files.append(os.path.basename(file))
            continue
        data = read_xlsx(xsens_file_path=file, smpl_file=smpl_path)
        pose = data['joint']['orientation'].to("cuda")
        trans = data['joint']['translation'].to("cuda")
        length = pose.shape[0]

        trans = trans - trans[0:1]

        ori = data['imu']['calibrated orientation'][:, XSENS_IMU_MASK].to("cuda")
        acc = data['imu']['free acceleration'][:, XSENS_IMU_MASK].to("cuda")

        grot, joint, vert, ljoint = body_model.forward_kinematics(pose=pose, tran=trans, calc_mesh=True)

        jvel = _syn_vel(joint, grot[:, 0:1])

        vacc = low_pass_filter(_syn_acc(vert[:, V_MASK]).unsqueeze(0)).squeeze(0)
        vrot = grot[:, J_MASK]
        javel = _syn_ang_vel(grot)

        targets = {
            'imu_acc': low_pass_filter(acc.unsqueeze(0)).squeeze(0).cpu().numpy(),
            'imu_ori': ori.cpu().numpy(),

            'pose': pose.cpu().numpy(),
            'trans': trans.cpu().numpy(),
            'grot': grot.cpu().numpy(),

            'jvel': jvel.cpu().numpy(),

            'vacc': vacc.cpu().numpy(),
            'vrot': vrot.cpu().numpy(),

            'javel': javel.cpu().numpy(),
            "joint": joint.cpu().numpy(),
            "ljoint": ljoint.cpu().numpy(),

        }

        target_folder = file.split('MTwAwinda')[0]
        split_foder = file.split('\\')[1]

        if split_foder in MTW_TRAIN_SPLIT:
            folder = "Train"
        elif split_foder in MTW_VALID_SPLIT:
            folder = "Valid"
        elif split_foder in MTW_TEST_SPLIT:
            folder = "Test"
        else:
            raise ValueError(f"Folder {split_foder} not recognized")

        target_folder = os.path.join(target_folder, f"MTW_process_data/{split_foder}").replace("raw data", folder)
        os.makedirs(target_folder,
                    exist_ok=True)

        np.savez(os.path.join(target_folder, os.path.basename(file).replace('.xlsx', '')), **targets)

        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del pose, trans, grot, joint, vert, targets, length
        torch.cuda.empty_cache()
        gc.collect()


def process_mvnx(smpl_path, mvnx_path) -> None:
    r"""
    Load Pose & Translation and Place Virtual IMU on MVNX dataset +
    Synthesize Joint Velocity, Joint Acceleration, Joint Angular Velocity, Joint Position

    :param smpl_path: path to smpl.pkl model.
    :param mvnx_path: folder containing MVNX dataset.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path, device="cuda")
    em_files = glob.glob(os.path.join(mvnx_path, '*/*.mvnx'))
    loop = tqdm(em_files)
    for file in loop:
        data = read_mvnx(file=file, smpl_file=smpl_path)
        pose = data['joint']['orientation'].to("cuda")
        trans = data['joint']['translation'].to("cuda")
        length = pose.shape[0]

        ori = data['imu']['calibrated orientation'][:, XSENS_IMU_MASK].to("cuda")
        acc = data['imu']['free acceleration'][:, XSENS_IMU_MASK].to("cuda")
        trans = trans - trans[0:1]

        grot, joint, vert, ljoint = body_model.forward_kinematics(pose=pose, tran=trans, calc_mesh=True)

        jvel = _syn_vel(joint, grot[:, 0:1])

        vacc = low_pass_filter(_syn_acc(vert[:, V_MASK]).unsqueeze(0)).squeeze(0)
        vrot = grot[:, J_MASK]
        javel = _syn_ang_vel(grot)

        targets = {
            'imu_acc': acc.cpu().numpy(),
            'imu_ori': ori.cpu().numpy(),

            'pose': pose.cpu().numpy(),
            'trans': trans.cpu().numpy(),
            'grot': grot.cpu().numpy(),

            'jvel': jvel.cpu().numpy(),

            'vacc': vacc.cpu().numpy(),
            'vrot': vrot.cpu().numpy(),

            'javel': javel.cpu().numpy(),
            "joint": joint.cpu().numpy(),
            "ljoint": ljoint.cpu().numpy(),

        }

        seq = file.split('\\')[-2]
        folder = "Train"
        target_folder = file.split('xens_mnvx')[0].replace("raw data", folder)
        target_folder = os.path.join(target_folder, "MVNX_process_data", seq)
        os.makedirs(target_folder,
                    exist_ok=True)

        np.savez(os.path.join(target_folder, os.path.basename(file).replace('.mvnx', '')), **targets)

        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del pose, trans, grot, joint, vert, targets, length
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    smpl_path = "./data/smpl/SMPL_MALE.pkl"

    amass_path = "./data/raw data/AMASS"
    dip_path = "./data/raw data/DIP_IMU"
    tc_path = "./data/raw data/TotalCapture"
    emonike_path = "./data/raw data/EmokineDataset_v1.0/Data/MVNX"
    mtw_path = "./data/raw data/MTwAwinda"
    mvnx_path = "./data/raw data/xens_mnvx/"

    process_amass(smpl_path=smpl_path, amass_path=amass_path)
    process_dip(smpl_path=smpl_path, dip_path=dip_path)
    process_total_capture(smpl_path=smpl_path, tc_path=tc_path)
    process_emonike(smpl_path=smpl_path, emonike_path=emonike_path)
    process_xsens(smpl_path=smpl_path, mvnx_path=mtw_path)
    process_mvnx(smpl_path=smpl_path, mvnx_path=mvnx_path)
