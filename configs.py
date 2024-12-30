from dataclasses import dataclass
import inspect
import torch
from operator import itemgetter

@dataclass
class ModelConfig:
    """ 
    Configuration for the TSPoser model.
    Any field that is not set when the class is instantiated will get its default value from the class definition.
    """
    exp_name = "model_final_SA2"
    evaluate = False
    log_dir = None

    device = "cuda"
    
    embed_dim = 128
    mlp_coef = 4
    n_lstm_layer = 2
    kernel_size = 7
    n_head = 4
    aux_weights = 1.0

    reduced_joints = torch.tensor([0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17])
    ignored_joints = torch.tensor([7, 8, 10, 11, 20, 21, 22, 23])
    imu_idx = torch.tensor([0, 4, 5, 11, 14, 15])
    leaf_idx = torch.tensor([0, 7, 8, 15, 20, 21])
    joint_pos_idx = torch.tensor([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15, 16, 18, 20, 22, 17, 19, 21, 23])
    all_joints = list(range(1, 24))
    shape = None

    adj_matrix = None
    use_adj_mask = True
    time_window = 200

    target_fps_list = [30]

    data_dir = "./data"
    smpl_dir = "./data/smpl/SMPL_MALE.pkl"
    pose_model_path = r"train_log/model_final_TemporalGraphXCTX/checkpoint/best_model_final_TemporalGraphXCTX.pth"
    tran_model_path = r"train_log/model_final_TemporalGraphXCT2/checkpoint/best_model_final_TemporalGraphXCT2.pth"
    batch_size = 32
    num_worker = 4

    optimizer = "ADAMW"
    lr = 1e-3
    min_lr = lr /100
    momentum = 0.9
    adamw_betas = (0.9, 0.999)
    weight_decay = 1e-5
    grad_norm = 5.0
    sd_prob = 0.1

    epochs = 100
    
    accum_intervals = [1, 4, 8, 32]
    rot_loss_coef = 1.0

    early_stop_epoch = 10
    early_stop_delta = 0.5
    warmup_epoch = 0
    warmup_factor = 0.1

    schedular = "COS"

    resume = True
    overfit_test = False

    loss_slices = {

        "out_rot": {
            "targets": ["grot"],
            "index": [reduced_joints[1:]],
        },
        
        "imus_velocity": {
            "targets": ["jvel"],
            "index": [leaf_idx],
        },
        "out_trn": {
                "targets": ["trans"],
                "index": [None],
            },
    }
