# codes from https://github.com/Xinyu-Yi/PIP.git and
#            https://github.com/dx118/dynaip.git and
#            https://github.com/jyf588/transformer-inertial-poser.git


import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
import roma
import torch

from smpl_model import ParametricModel

V_MASK = [3021, 1176, 4662, 411, 1961, 5424]
J_MASK = [0, 4, 5, 15, 18, 19]
AMASS_ROT = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]], device="cuda")

AMASS_TRAIN_SPLIT = ['ACCAD', 'BMLhandball', 'BMLmovi', 'CMU',
                     'GRAB', 'HUMAN4D', 'HumanEva', 'MPI_HDM05',
                     'MPI_Limits', 'MPI_mosh', 'SFU', 'SOMA',
                     'SSM_synced', 'TCD_handMocap', 'Transitions_mocap',
                     'Eyes_Japan_Dataset', 'DFaust_67', 'EKUT', 'DanceDB', "TotalCapture"]
# AMASS_TRAIN_SPLIT = ['CMU', 'BMLmovi', 'Eyes_Japan_Dataset']
AMASS_VALID_SPLIT = []
AMASS_TEST_SPLIT = []

DIP_IMU_MASK = [2, 11, 12, 0, 7, 8]
DIP_TRAIN_SPLIT = []
DIP_VALID_SPLIT = []
DIP_TEST_SPLIT = ["s_01", "s_02", "s_03", "s_04", "s_05", "s_06", "s_07", "s_08", "s_09", "s_10"]

# TC_IMU_MASK = torch.tensor([5, 0, 1, 4, 2, 3])
TC_IMU_MASK = [5, 2, 3, 4, 0, 1]
TC_TRAIN_SPLIT = []
TC_VALID_SPLIT = ["S1", "S2", "S3", "S4", "S5"]
TC_TEST_SPLIT = []

XSENS_IMU_MASK = [0, 15, 12, 2, 9, 5]
EM_TRAIN_SPLIT = ["seq1", "seq2", "seq3", "seq4", "seq5", "seq6", "seq7", "seq8", "seq9"]
EM_VALID_SPLIT = []
EM_TEST_SPLIT = []

MTW_TRAIN_SPLIT = ["trials", "trials_extra_indoors", "trials_extra_outdoors"]
MTW_VALID_SPLIT = []
MTW_TEST_SPLIT = []

IG_JOINTS = [7, 8, 10, 11, 20, 21, 22, 23]


def _syn_acc(trans: torch.Tensor) -> torch.Tensor:
    r"""
    Synthesize acceleration from positions
    
    :param trans: translation in [T, N, 3]
    :return: acceleration in [T, N, 3]
    """
    acc = torch.zeros_like(trans)
    acc[1:-1] = (trans[0:-2] + trans[2:] - 2 * trans[1:-1]) * 3600
    acc[4:-4] = (trans[0:-8] + trans[8:] - 2 * trans[4:-4]) * 3600 / 16
    return acc


def _syn_ang_vel(pose: torch.Tensor) -> torch.Tensor:
    r"""
    Synthesize angular velocity from pose
    
    :param pose: pose in [T, N, 3, 3]
    :return: angular velocity in [T, N, 3]
    """
    ang_v = torch.zeros(pose.shape[:-1])
    quat = roma.rotmat_to_unitquat(pose)

    cond = torch.linalg.norm(quat[2:] - quat[:-2], dim=-1, keepdim=True) < torch.linalg.norm(quat[2:] + quat[:-2],
                                                                                             dim=-1, keepdim=True)
    sub = torch.where(cond, quat[2:] - quat[:-2], quat[2:] + quat[:-2])

    dori = 2 * roma.quat_product(sub, roma.quat_conjugation(quat[2:]))
    ang_v[1:-1] = (dori * 30)[..., :3]
    return ang_v


def _syn_vel(trans: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    r"""
    Synthesize velocity from positions
    
    :param trans: positions in [T, N, 3]
    :return: velocity in [T, N, 3]
    """
    vel = torch.zeros_like(trans)
    vel[1:] = (trans[1:] - trans[:-1])
    # vel[:, 1:] = vel[:, 1:] - vel[:, :1]
    vel = root.transpose(-1, -2) @ vel.unsqueeze(-1)
    return vel[..., 0] * 60


def gen_trans_dip(pose, body_model):
    r"""
    Synthesize trans for y-axis
    :param pose: body pose in [T, 24, 3]
    :param body_model: SMPL body model
    :return: translation vector in [T, 3]
    """
    j, _ = body_model.get_zero_pose_joint_and_vertex()
    floor_y = j[10:12, 1].min().item()

    j = body_model.forward_kinematics(pose=pose, calc_mesh=False)[1]

    trans = torch.zeros(j.shape[0], 3)

    for i in range(j.shape[0]):
        current_foot_y = j[i, [10, 11], 1].min().item()
        if current_foot_y > floor_y:
            trans[i, 1] = floor_y - current_foot_y

    return trans


def imu_calibration(imu_data, joints_data, n_frame, joint_mask):
    r""" 
    Sensor Calibration for Xesns datasets
    
    :param imu_data: imu orientation data in [T, N, 3, 3]
    :param joint_data: joint orientation data in [T, N, 3, 3]
    :param n_frame: number of frame for calibraion 
    :param joint_mask: imu mask
    :return: calibrated imu orientation data in [T, N, 3, 3]
    """
    quat1 = imu_data[:n_frame]
    quat2 = joints_data[:n_frame, joint_mask]
    quat_off = roma.quat_product(roma.quat_inverse(quat1), quat2)
    ds = quat_off.abs().mean(dim=0).max(dim=-1)[1]
    for i, d in enumerate(ds):
        quat_off[:, i] = quat_off[:, i] * quat_off[:, i, d:d + 1].sign()

    quat_off = roma.quat_normalize(roma.quat_normalize(quat_off).mean(dim=0))
    return roma.quat_product(imu_data,
                             quat_off.repeat(imu_data.shape[0], 1, 1))


def convert_quaternion_xsens(quat):
    r""" inplace convert
        R = [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        smpl_pose = R mvnx_pose R^T
    """
    old_quat = quat.reshape(-1, 4).clone()
    quat.view(-1, 4)[:, 1] = old_quat[:, 2]
    quat.view(-1, 4)[:, 2] = old_quat[:, 3]
    quat.view(-1, 4)[:, 3] = old_quat[:, 1]


def convert_point_xsens(point):
    r""" inplace convert
        R = [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        smpl_pose = R mvnx_pose R^T
    """
    old_point = point.reshape(-1, 3).clone()
    point.view(-1, 3)[:, 0] = old_point[:, 1]
    point.view(-1, 3)[:, 1] = old_point[:, 2]
    point.view(-1, 3)[:, 2] = old_point[:, 0]


def convert_quaternion_excel(quat):
    r""" inplace convert
        R = [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        smpl_pose = R mvnx_pose R^T
    """
    quat[:, :, [1, 2, 3]] = quat[:, :, [2, 3, 1]]
    return quat


def convert_point_excel(point):
    r""" inplace convert
        R = [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        smpl_point = R mvnx_point
    """
    point[:, :, [0, 1, 2]] = point[:, :, [1, 2, 0]]
    return point


def read_mvnx(file: str, smpl_file: str):
    r"""
    Load IMU & Joint data from MVNX file.

    :param file: MVNX file path.
    :param smpl_file: path to smpl.pkl model.
    :return: data dictionary containing IMU and Joint data.
    """
    model = ParametricModel(smpl_file)
    tree = ET.parse(file)

    # read framerate
    frameRate = int(tree.getroot()[2].attrib['frameRate'])

    segments = tree.getroot()[2][1]
    n_joints = len(segments)
    joints = []
    for i in range(n_joints):
        assert int(segments[i].attrib['id']) == i + 1
        joints.append(segments[i].attrib['label'])

    sensors = tree.getroot()[2][2]
    n_imus = len(sensors)
    imus = []
    for i in range(n_imus):
        imus.append(sensors[i].attrib['label'])

    # read frames
    frames = tree.getroot()[2][-1]
    data = {'framerate': frameRate,
            'timestamp ms': [],
            'joint': {'orientation': [], 'position': []},
            'imu': {'free acceleration': [], 'orientation': []},
            }

    if frameRate != 60:
        step = int(frameRate // 60)
    else:
        step = 1

    for i in range(len(frames)):
        if frames[i].attrib['type'] in ['identity', 'tpose', 'tpose-isb']:  # virginia
            continue

        elif ('index' in frames[i].attrib) and (frames[i].attrib['index'] == ''):  # unipd
            continue

        orientation = torch.tensor([float(_) for _ in frames[i][0].text.split(' ')]).view(n_joints, 4)
        position = torch.tensor([float(_) for _ in frames[i][1].text.split(' ')]).view(n_joints, 3)
        sensorFreeAcceleration = torch.tensor([float(_) for _ in frames[i][7].text.split(' ')]).view(n_imus, 3)
        try:
            sensorOrientation = torch.tensor([float(_) for _ in frames[i][9].text.split(' ')]).view(n_imus, 4)
        except:
            sensorOrientation = torch.tensor([float(_) for _ in frames[i][8].text.split(' ')]).view(n_imus, 4)

        data['timestamp ms'].append(int(frames[i].attrib['time']))
        data['joint']['orientation'].append(orientation)
        data['joint']['position'].append(position)
        data['imu']['free acceleration'].append(sensorFreeAcceleration)
        data['imu']['orientation'].append(sensorOrientation)

    data['timestamp ms'] = torch.tensor(data['timestamp ms'])
    for k, v in data['joint'].items():
        data['joint'][k] = torch.stack(v)
    for k, v in data['imu'].items():
        data['imu'][k] = torch.stack(v)

    data['joint']['name'] = joints
    data['imu']['name'] = imus

    # to smpl coordinate frame

    convert_quaternion_xsens(data['joint']['orientation'])
    convert_point_xsens(data['joint']['position'])
    convert_quaternion_xsens(data['imu']['orientation'])
    convert_point_xsens(data['imu']['free acceleration'])

    if step != 1:
        data['joint']['orientation'] = data['joint']['orientation'][::step].clone()
        data['joint']['position'] = data['joint']['position'][::step].clone()
        data['imu']['free acceleration'] = data['imu']['free acceleration'][::step].clone()
        data['imu']['orientation'] = data['imu']['orientation'][::step].clone()

    # use first 150 frames for calibration
    n_frames_for_calibration = 150
    imu_idx = [data['joint']['name'].index(_) for _ in data['imu']['name']]

    imu_ori = imu_calibration(roma.quat_wxyz_to_xyzw(data['imu']['orientation']),
                              roma.quat_wxyz_to_xyzw(data['joint']['orientation']),
                              n_frames_for_calibration,
                              imu_idx)
    data['imu']['calibrated orientation'] = imu_ori

    data['joint']['orientation'] = roma.quat_normalize(roma.quat_wxyz_to_xyzw(data['joint']['orientation']))
    data['joint']['orientation'] = roma.unitquat_to_rotmat(data['joint']['orientation'])

    data['imu']['calibrated orientation'] = roma.quat_normalize(data['imu']['calibrated orientation'])
    data['imu']['calibrated orientation'] = roma.unitquat_to_rotmat(data['imu']['calibrated orientation'])

    data['imu']['orientation'] = roma.quat_normalize(data['imu']['orientation'])
    data['imu']['orientation'] = roma.unitquat_to_rotmat(data['imu']['orientation'])

    indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
    data['joint']['orientation'] = data['joint']['orientation'][:, indices]
    data['joint']['orientation'] = model.inverse_kinematics_R(data['joint']['orientation'])
    data['joint']['position'] = data['joint']['position'][:, indices]
    data['joint']['translation'] = data['joint']['position'][:, 0]

    return data


def read_xlsx(xsens_file_path: str, smpl_file: str):
    r"""
    Load IMU & Joint data from xlsx file.

    :param file: xlsx file path.
    :param smpl_file: path to smpl.pkl model.
    :return: data dictionary containing IMU and Joint data.
    """
    model = ParametricModel(smpl_file)
    pos3s_com, segments_pos3d, segments_quat, \
        imus_ori, imus_free_acc = pd.read_excel(
        xsens_file_path,
        sheet_name=["Center of Mass",
                    "Segment Position",  # positions of joints in 3d space
                    "Segment Orientation - Quat",  # segment global orientation
                    "Sensor Orientation - Quat",  # sensor orientation
                    "Sensor Free Acceleration",  # sensor free acceleration (accelerometer data without gravity vector)
                    ],
        index_col=0
    ).values()

    data = {'framerate': 60.,
            'joint': {'orientation': [], 'position': []},
            'imu': {'free acceleration': [], 'orientation': []},
            }

    # add dim (S, [1], 3)  +  ignore com_vel / com_accel
    pos3s_com = np.expand_dims(pos3s_com.values, axis=1)[..., [0, 1, 2]]
    n_samples = len(pos3s_com)

    # assumes a perfect sampling freq of 60hz
    timestamps = np.arange(1, n_samples + 1) * (1 / 60.)

    segments_pos3d = segments_pos3d.values.reshape(n_samples, -1, 3)
    segments_quat = segments_quat.values.reshape(n_samples, -1, 4)
    imus_free_acc = imus_free_acc.values.reshape(n_samples, -1, 3)
    imus_ori = imus_ori.values.reshape(n_samples, -1, 4)
    mask = torch.tensor([0, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21])
    imus_ori = imus_ori[:, mask, :]
    imus_free_acc = imus_free_acc[:, mask, :]

    data['joint']['orientation'] = torch.tensor(segments_quat.astype(np.float32)).clone()
    data['joint']['position'] = torch.tensor(segments_pos3d.astype(np.float32)).clone()
    data['imu']['orientation'] = torch.tensor(imus_ori.astype(np.float32)).clone()
    data['imu']['free acceleration'] = torch.tensor(imus_free_acc.astype(np.float32)).clone()

    data['joint']['orientation'] = convert_quaternion_excel(data['joint']['orientation'])
    data['joint']['position'] = convert_point_excel(data['joint']['position'])
    data['imu']['orientation'] = convert_quaternion_excel(data['imu']['orientation'])
    data['imu']['free acceleration'] = convert_point_excel(data['imu']['free acceleration'])

    data['joint']['name'] = ['Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head', 'RightShoulder', 'RightUpperArm',
                             'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
                             'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToe', 'LeftUpperLeg', 'LeftLowerLeg',
                             'LeftFoot', 'LeftToe']
    data['imu']['name'] = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
                           'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg',
                           'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']

    # use first 150 frames for calibration
    n_frames_for_calibration = 150
    imu_idx = [data['joint']['name'].index(_) for _ in data['imu']['name']]

    imu_ori = imu_calibration(roma.quat_wxyz_to_xyzw(data['imu']['orientation']),
                              roma.quat_wxyz_to_xyzw(data['joint']['orientation']),
                              n_frames_for_calibration,
                              imu_idx)
    data['imu']['calibrated orientation'] = imu_ori

    data['joint']['orientation'] = roma.quat_normalize(roma.quat_wxyz_to_xyzw(data['joint']['orientation']))
    data['joint']['orientation'] = roma.unitquat_to_rotmat(data['joint']['orientation'])

    data['imu']['calibrated orientation'] = roma.quat_normalize(data['imu']['calibrated orientation'])
    data['imu']['calibrated orientation'] = roma.unitquat_to_rotmat(data['imu']['calibrated orientation'])

    data['imu']['orientation'] = roma.quat_normalize(data['imu']['orientation'])
    data['imu']['orientation'] = roma.unitquat_to_rotmat(data['imu']['orientation'])

    indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
    data['joint']['orientation'] = data['joint']['orientation'][:, indices]
    data['joint']['orientation'] = model.inverse_kinematics_R(data['joint']['orientation'])
    data['joint']['position'] = data['joint']['position'][:, indices]
    data['joint']['translation'] = data['joint']['position'][:, 0]

    return data
