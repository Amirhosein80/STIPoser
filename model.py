import time
from collections import OrderedDict

import torch
import torch.nn as nn

from blocks import ConvLSTMBlock, TransformerBlock, CausalAvgPool1dWithState, TransformerBlockDecoder, MixerBlock, Mixer
from configs import ModelConfig
from model_utils import r6d2rot_mat
from smpl_model import ParametricModel


class BaseModel(nn.Module):
    """
    A base class for all the models. It contains the common members and methods used by all the models.

    The BaseModel class is an abstract class that provides a common interface for all the models.
    It contains the SMPL model, the shape vector, the reduced joints, the ignored joints, the IMU index,
    the leaf index, and the device.
    
    The BaseModel class also contains a method to get the model parameters, and a method to convert the global 6D pose
    to the full local matrix pose.

    The BaseModel class should be inherited by all the models.
    """

    def __init__(self, configs: ModelConfig, smpl_model: ParametricModel) -> None:
        """
        :param configs: The configurations for the model
        :param smpl_model: The SMPL model
        :return: None
        """
        super().__init__()

        self.smpl_model = smpl_model

        self.reduced = configs.reduced_joints
        self.ignored = configs.ignored_joints

        self.leaf_joints = configs.leaf_idx
        self.imu_idx = configs.imu_idx
        self.joint_pos_idx = configs.joint_pos_idx

        self.device = configs.device

        adj_dict = torch.load("./adj_matrix.pt")
        self.adj_joint = adj_dict["joints"].float().to(self.device)
        self.adj_imu = adj_dict["imu"].float().to(self.device)
        self.adj_imu2joint = adj_dict["imu2joints"].float().to(self.device)

    def glb_6d_to_full_local_mat(self, glb_pose: torch.Tensor, sensor_rot: torch.Tensor = None) -> torch.Tensor:
        """
        Convert the global 6D pose to the full local matrix pose.

        This method takes the global 6D pose, and converts it to the full local matrix pose using the SMPL model.

        The input global 6D pose is in the shape of (T, N, 6), where T is the sequence length, N is the number of joints,
         and 6 is the dimension of the 6D pose.

        The output full local matrix pose is in the shape of (T, 24, 3, 3), where 24 is the number of joints in
         the SMPL model, and 3 is the dimension of the 3D rotation matrix.

        :param glb_pose: The global 6D pose in the shape of (T, N, 6)
        :return: The full local matrix pose in the shape of (T, 24, 3, 3)
        """
        T, N, _ = glb_pose.shape

        glb_pose = r6d2rot_mat(glb_pose).view(T, N, 3, 3)
        ign_pose = torch.eye(3, device=glb_pose.device).reshape(1, 1, 3, 3).repeat(1, len(self.ignored), 1, 1)

        global_full_pose = torch.eye(3, device=glb_pose.device).repeat(T, 24, 1, 1)
        global_full_pose[:, self.reduced[1:]] = glb_pose

        if sensor_rot is not None:
            global_full_pose[:, [0, 4, 5, 15, 18, 19]] = sensor_rot

        pose = self.smpl_model.inverse_kinematics_R(global_full_pose)
        pose[:, self.ignored] = ign_pose

        return pose

    def get_params(self, lr: float, weight_decay: float) -> list[dict]:
        """
        Get the model parameters with learning rate and weight decay.

        This method is used to get the model parameters with learning rate and weight decay for the optimizer.

        The method returns a list of dictionaries, where each dictionary contains the parameters, the learning rate, and the weight decay.

        :param lr: The learning rate
        :param weight_decay: The weight decay
        :return: A list of dictionaries, where each dictionary contains the parameters, the learning rate, and the weight decay.
        """
        params_wd = []
        params_nwd = []

        for p in self.parameters():
            if p.dim == 1:
                params_nwd.append(p)
            else:
                params_wd.append(p)

        params = [
            {"params": params_wd, "lr": lr, "weight_decay": weight_decay},
            {"params": params_nwd, "lr": lr, "weight_decay": 0},
        ]

        return params


class STIPoser(BaseModel):
    """
    The STIPoser class is a model  predict the SMPL pose from the input data.
    """

    def __init__(self, configs: ModelConfig, smpl_model: ParametricModel) -> None:
        """
        :param configs: The configurations for the model
        :param smpl_model: The SMPL model
        :return: None
        """
        super().__init__(smpl_model=smpl_model, configs=configs)

        self.embed_dim = configs.embed_dim
        self.n_lstm_layer = configs.n_lstm_layer
        self.mlp_coef = configs.mlp_coef
        self.kernel_size = configs.kernel_size
        self.n_head = configs.n_head

        self.imu_temporal = ConvLSTMBlock(self.embed_dim, n_layer=self.n_lstm_layer,
                                          kernel_size=self.kernel_size, coef=self.mlp_coef)
        self.trn_temporal = ConvLSTMBlock(self.embed_dim, n_layer=1,
                                          kernel_size=self.kernel_size, coef=self.mlp_coef)

        self.imu_spatial_modules = nn.Sequential(
            TransformerBlock(self.embed_dim, adj_mat=self.adj_imu, coef=self.mlp_coef, n_head=self.n_head),
            TransformerBlock(self.embed_dim, adj_mat=self.adj_imu, coef=self.mlp_coef, n_head=self.n_head),
            TransformerBlock(self.embed_dim, adj_mat=self.adj_imu, coef=self.mlp_coef, n_head=self.n_head),
            TransformerBlock(self.embed_dim, adj_mat=self.adj_imu, coef=self.mlp_coef, n_head=self.n_head),
        )

        self.joint_spatial_modules = nn.Sequential(
            TransformerBlockDecoder(self.embed_dim, adj_mat_query=self.adj_joint, adj_mat_query_key=self.adj_imu2joint,
                                    coef=self.mlp_coef, n_head=self.n_head),
            TransformerBlockDecoder(self.embed_dim, adj_mat_query=self.adj_joint, adj_mat_query_key=self.adj_imu2joint,
                                    coef=self.mlp_coef, n_head=self.n_head),
        )

        self.mixer = nn.Sequential(
            MixerBlock(self.embed_dim, d_input=6, d_output=24, coef=self.mlp_coef),
            MixerBlock(self.embed_dim, d_input=24, d_output=11, coef=self.mlp_coef),
        )

        self.cat_layer = nn.Sequential(
            nn.RMSNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )

        self.imu_embedding = nn.Sequential(
            nn.RMSNorm(12),
            nn.Linear(12, self.embed_dim),
        )

        self.pose_head = nn.Sequential(
            nn.RMSNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 6),
        )

        self.tran_head = nn.Sequential(
            nn.RMSNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 3),
        )

        self.imuv_head = nn.Sequential(
            nn.RMSNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 3),
        )

        self.init_jvel_state = nn.Sequential(
            nn.RMSNorm(3),
            nn.Linear(3, 2 * self.n_lstm_layer * self.embed_dim),
            Mixer(24, 6, coef=self.mlp_coef),
            nn.Linear(2 * self.n_lstm_layer * self.embed_dim, 2 * self.n_lstm_layer * self.embed_dim),
            nn.RMSNorm(2 * self.n_lstm_layer * self.embed_dim),
        )

        self.init_tran_state = nn.Sequential(
            nn.RMSNorm(3),
            nn.Linear(3, self.embed_dim * 4),
            nn.SiLU(),
            nn.Linear(self.embed_dim * 4, 2 * 1 * self.embed_dim),
            nn.RMSNorm(2 * 1 * self.embed_dim),
        )

        self.trans_avg_pool = CausalAvgPool1dWithState(kernel_size=self.kernel_size)

        self.state_inited = False
        self.states = [None, None, None, None]

        self.imu_pos_emb = nn.Parameter(torch.Tensor(1, 6, self.embed_dim))
        self.joint_pos_emb = nn.Parameter(torch.Tensor(1, len(self.reduced), self.embed_dim))

        nn.init.xavier_normal_(self.imu_pos_emb)
        nn.init.xavier_normal_(self.joint_pos_emb)

        self.to(self.device)

    def forward(self, datas: dict) -> OrderedDict:
        """
        Forward pass of the model. The model takes in a batch of data, including:

        - imu_acc: (B, T, 6, 3) tensor of imu acceleration data
        - imu_ori: (B, T, 6, 3, 3) tensor of imu orientation data
        - last_jvel: (B, 24, 3) tensor of the last joint velocity
        - last_trans: (B, 3) tensor of the last translation

        The model outputs a dictionary with the following keys:

        - pose: (B, T, 16, 3) tensor of the predicted pose
        - tran: (B, T, 3) tensor of the predicted translation

        :param datas: The data to be processed
        :return: A dictionary with the predicted pose and translation
        """
        outputs = OrderedDict()

        imu_acc = datas["imu_acc"]
        imu_ori = datas["imu_ori"]

        batch_size, seq_len, num_imus, _ = imu_acc.size()

        # Normalize imu_acc and imu_ori
        imu_acc[:, :, 1:] = imu_acc[:, :, 1:] - imu_acc[:, :, 0:1]
        imu_acc = imu_acc.matmul(imu_ori[:, :, 0])
        imu_ori[:, :, 1:] = imu_ori[:, :, :1].transpose(-1, -2).matmul(imu_ori[:, :, 1:])

        x = torch.cat((imu_acc, imu_ori.flatten(-2)), dim=-1)

        last_vimu = datas["last_jvel"].to(x.dtype)
        last_tran = datas["last_trans"].to(x.dtype)

        hs0, cs0 = self.init_jvel_state(last_vimu).reshape(batch_size * 6, 2, self.n_lstm_layer,
                                                           self.embed_dim).permute(1, 2, 0, 3)
        hs1, cs1 = self.init_tran_state(last_tran).reshape(batch_size, 2, 1, self.embed_dim).permute(1, 2, 0, 3)

        hs0, cs0 = hs0.contiguous(), cs0.contiguous()
        hs1, cs1 = hs1.contiguous(), cs1.contiguous()

        x = self.imu_embedding(x.reshape(batch_size * seq_len, num_imus, 12)) + self.imu_pos_emb

        y = self.imu_spatial_modules(x)
        y = y.reshape(batch_size, seq_len, num_imus, self.embed_dim).transpose(1, 2).reshape(batch_size * num_imus,
                                                                                             seq_len, self.embed_dim)
        y, _ = self.imu_temporal(y, ((hs0, cs0), None))
        y = y.reshape(batch_size, num_imus, seq_len, self.embed_dim).transpose(1, 2).reshape(batch_size * seq_len,
                                                                                             num_imus, self.embed_dim)

        outputs["imus_velocity"] = self.imuv_head(y).reshape(batch_size, seq_len, num_imus, 3)

        x = self.cat_layer(torch.cat((x, y), dim=-1))

        q = self.mixer(y) + self.joint_pos_emb

        q, x = self.joint_spatial_modules((q, x))
        t = q[:, 0].reshape(batch_size, seq_len, self.embed_dim)
        x = q[:, 1:]

        t, _ = self.trn_temporal(t, ((hs1, cs1), None))

        outputs["out_rot"] = self.pose_head(x).reshape(batch_size, seq_len, len(self.reduced) - 1, 6)
        outputs["out_trn"] = self.tran_head(t).reshape(batch_size, seq_len, 3)

        return outputs

    def init_state(self):
        """
        Initial state of the model.
        :return:
        """
        self.states = [None, None, None, None]

        last_vimu = torch.zeros((1, 24, 3), device=self.device)
        last_tran = torch.zeros((1, 3), device=self.device)

        hs0, cs0 = self.init_jvel_state(last_vimu).reshape(1 * 6, 2, self.n_lstm_layer, self.embed_dim).permute(1, 2, 0,
                                                                                                                3)
        hs1, cs1 = self.init_tran_state(last_tran).reshape(1, 2, 1, self.embed_dim).permute(1, 2, 0, 3)

        hs0, cs0 = hs0.contiguous().detach(), cs0.contiguous().detach()
        hs1, cs1 = hs1.contiguous().detach(), cs1.contiguous().detach()

        self.states[0] = ((hs0, cs0), None)
        self.states[2] = ((hs1, cs1), None)

        self.state_inited = True

    @torch.no_grad()
    def forward_online(self, imu_acc, imu_ori):
        """
        Forward method for online inference.

        :param imu_acc: The acceleration data of the IMU [6, 3]
        :param imu_ori: The orientation data of the IMU [6, 3, 3]
        :return: The predicted pose and translation with shape (1, 24, 3) and (1, 3)
        """
        # assert imu_acc.size() == (6, 3) and imu_ori.size() == (6, 3, 3)

        sensors = imu_ori.clone().unsqueeze(0)

        if not self.state_inited:
            self.init_state()

        # Normalize imu_acc and imu_ori
        imu_acc[1:] = imu_acc[1:] - imu_acc[0:1]
        imu_acc = imu_acc.matmul(imu_ori[0])
        imu_ori[1:] = imu_ori[:1].transpose(-1, -2).matmul(imu_ori[1:])

        x = torch.cat((imu_acc, imu_ori.flatten(-2)), dim=-1).unsqueeze(0)

        x = self.imu_embedding(x) + self.imu_pos_emb

        y = self.imu_spatial_modules(x)
        y, self.states[0] = self.imu_temporal(y.transpose(0, 1), self.states[0])
        y = y.transpose(0, 1)

        x = self.cat_layer(torch.cat((x, y), dim=-1))

        q = self.mixer(y) + self.joint_pos_emb
        # q, self.states[1] = self.pos_temporal(q.transpose(0, 1),  self.states[1])

        q, x = self.joint_spatial_modules((q, x))

        t = q[:, :1]
        x = q[:, 1:]

        t, self.states[2] = self.trn_temporal(t, self.states[2])
        x = self.pose_head(x)
        t = self.tran_head(t)

        t, self.states[3] = self.trans_avg_pool(t, self.states[3])

        return self.glb_6d_to_full_local_mat(x, sensor_rot=sensors), t[0]

    @torch.no_grad()
    def forward_offline(self, imu_acc, imu_ori, return_6d=False):
        """
        Forward method for ofline inference.

        :param imu_acc: The acceleration data of the IMU [T, 6, 3]
        :param imu_ori: The orientation data of the IMU [T, 6, 3, 3]
        :return: The predicted pose and translation with shape (T, 24, 3) and (T, 3)
        """

        sensors = imu_ori.clone()

        T = imu_acc.size(0)
        # assert imu_acc.size() == (T, 6, 3) and imu_ori.size() == (T, 6, 3, 3)

        imu_acc[:, 1:] = imu_acc[:, 1:] - imu_acc[:, 0:1]
        imu_acc = imu_acc.matmul(imu_ori[:, 0])

        imu_ori[:, 1:] = imu_ori[:, :1].transpose(-1, -2).matmul(imu_ori[:, 1:])

        last_vimu = torch.zeros((1, 24, 3), device=self.device)
        last_tran = torch.zeros((1, 3), device=self.device)

        hs0, cs0 = self.init_jvel_state(last_vimu).reshape(1 * 6, 2, self.n_lstm_layer, self.embed_dim).permute(1, 2, 0,
                                                                                                                3)
        hs1, cs1 = self.init_tran_state(last_tran).reshape(1, 2, 1, self.embed_dim).permute(1, 2, 0, 3)

        hs0, cs0 = hs0.contiguous().detach(), cs0.contiguous().detach()
        hs1, cs1 = hs1.contiguous().detach(), cs1.contiguous().detach()

        x = torch.cat((imu_acc, imu_ori.flatten(-2)), dim=-1)

        x = self.imu_embedding(x) + self.imu_pos_emb

        y = self.imu_spatial_modules(x)
        y, _ = self.imu_temporal(y.transpose(0, 1), ((hs0, cs0), None))
        y = y.transpose(0, 1)

        x = self.cat_layer(torch.cat((x, y), dim=-1))

        q = self.mixer(y) + self.joint_pos_emb
        # q, _ = self.pos_temporal(q.transpose(0, 1),  None)

        q, x = self.joint_spatial_modules((q, x))

        t = q[:, :1]
        x = q[:, 1:]

        t, _ = self.trn_temporal(t.transpose(0, 1), ((hs1, cs1), None))

        x = self.pose_head(x)
        t = self.tran_head(t)
        t, _ = self.trans_avg_pool(t, None)

        if return_6d:
            return x, t[0]
        else:
            return self.glb_6d_to_full_local_mat(x, sensor_rot=sensors), t[0]





if __name__ == '__main__':
    import torchinfo

    torch.set_float32_matmul_precision('high')

    conf = ModelConfig()
    conf.exp_name = "test_speed"
    conf.device = "cpu"
    conf.time_window = 1
    conf.evaluate = True

    smpl = ParametricModel("./data/smpl/SMPL_MALE.pkl", device=conf.device)

    m = STIPoser(configs=conf, smpl_model=smpl)
    m.to(conf.device)
    m.forward = m.forward_offline
    torchinfo.summary(model=m,
                      input_data=[torch.randn(24, 6, 3).to(conf.device), torch.randn(24, 6, 3, 3).to(conf.device)],
                      device=conf.device, verbose=1)
    # print(m.adj_joint.shape)
    # print(m.states[0][0][0].shape, m.states[0][0][1].shape, m.states[0][1].shape)
    # # print(torch.load("./data/adj_matrix.pt").keys)
    m.forward = m.forward_online
    torchinfo.summary(model=m,
                      input_data=[torch.randn(6, 3).to(conf.device), torch.randn(6, 3, 3).to(conf.device)],
                      device=conf.device, verbose=1)
    run_benchmark(m, datas=(torch.randn(6, 3).to(conf.device),
                            torch.randn(6, 3, 3).to(conf.device)))

    # print(m.states[0][0][0].shape, m.states[0][0][1].shape, m.states[0][1].shape)
