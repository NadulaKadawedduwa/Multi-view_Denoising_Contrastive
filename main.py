from pprint import pprint
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import argparse

from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore

from config.TrainConfig import TrainConfig
from dataset import CustomDataset
from smaller_model import UNet

cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)

def train_cpu():
    print("Not implemented")


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train(rank, world_size, cfg: DictConfig):
    sub_set = int(cfg.subset_size)
    batch_size = int(cfg.batch_size)
    n_epochs = int(cfg.n_epochs)
    log_every = int(cfg.log_every)
    img_list_path = cfg.img_list_path

    gpu_id = rank
    ddp_setup(gpu_id, world_size)

    dataset = CustomDataset(img_list_file=img_list_path, device=gpu_id, sub_set=sub_set)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )

    lr = 1e-3
    model = UNet(input_channels=2).to(device=gpu_id)
    optim = Adam(model.parameters(), lr=lr)
    model = DDP(model, device_ids=[gpu_id])

    loss_fn = MSELoss()

    for epoch in range(n_epochs):
        for noisy, clean in tqdm(
            dataloader, disable=gpu_id != 0, desc=f"Epoch {epoch}/{n_epochs}"
        ):
            optim.zero_grad()
            pred = model(noisy)

            loss = loss_fn(pred, clean)
            loss.backward()
            optim.step()

        if epoch > 0 and epoch % log_every == 0:
            if gpu_id == 0:
                print(f"Loss: {loss.item()}")
                torch.save(model.module.state_dict(), f"./checkpoints/model_{epoch}.pth")
    
    dist.destroy_process_group()

@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg: TrainConfig):
    print("Beginning training with config:")
    pprint(cfg)
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"World size: {world_size}")
        mp.spawn(
            train,
            args=(
                world_size,
                cfg,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        train_cpu()



if __name__ == "__main__":
    main()
