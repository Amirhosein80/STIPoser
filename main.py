import os
os.environ["USE_FLASH_ATTENTION"] = "1"

import dataset
import loss
import optimizers
import transformes
import train_utils
import smpl_model
import torchinfo

from trainer import Trainer

from configs import ModelConfig
from model import STIPoser


def test_loss(batch, model, loss, device):
    train_utils.to_device(batch, device)
    pred = model(batch)
    print([(k , v.shape) for k, v in pred.items()])
    print(loss(pred, batch))


def main():

    model_config = ModelConfig()

    train_utils.setup_env()

    train_utils.create_log_dir(configs=model_config)

    train_ds = dataset.CustomDataset(configs=model_config, phase="Train")
    valid_ds = dataset.CustomDataset(configs=model_config, phase="Valid")

    train_transform = transformes.Compose([
        transformes.UniformNoise(p=0.5),
    ])

    train_dl = train_ds.get_data_loader()
    valid_dl = valid_ds.get_data_loader()

    dataloaders = {
        "train": train_dl,
        "valid": valid_dl,
    }

    trans = {
        "train": train_transform,
    }

    smpl = smpl_model.ParametricModel(official_model_file=model_config.smpl_dir,
                                      device=model_config.device)

    model = STIPoser(configs=model_config, smpl_model=smpl)
    model = model.to(model_config.device)
    print(f"\nModel have {train_utils.count_parameters(model)} parameters \n")
    
    batch = next(iter(train_dl))
    torchinfo.summary(model=model, input_data=[batch], device=model_config.device, verbose=1)

    criterion = loss.MotionLoss(configs=model_config).to(model_config.device)
    
    # batch = next(iter(train_dl))
    # print(batch["trans"].shape, batch["last_trans"].shape)
    # test_loss(batch, model, criterion, model_config.device)
    
    # del batch

    optimizer, scheduler, scaler = optimizers.get_optimizer(model, configs=model_config)

    trainer = Trainer(dataloaders=dataloaders, model=model, criterion=criterion, transforms=trans,
                      optimizer=optimizer, scheduler=scheduler, configs=model_config, scaler=scaler)

    trainer.train()


if __name__ == "__main__":
    main()
