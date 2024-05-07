import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as  dist
import torch.multiprocessing as mp
import os
import argparse

from dataset import CustomDataset
from smaller_model import UNet


def train_cpu():
    print('Not implemented')


def train(rank, world_size, args):
    sub_set = int(args.subset_size)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    log_every = int(args.log_every)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    dataset = CustomDataset(device=rank, sub_set=sub_set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device=rank)

    lr = 1e-3
    optim = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    for epoch in range(n_epochs):
        for noisy, clean in tqdm(dataloader, disable=rank != 0, desc=f'Epoch {epoch}/{n_epochs}'):
            optim.zero_grad()
            pred = model(noisy.to(torch.float))

            loss = loss_fn(pred, clean)
            loss.backward()
            optim.step()

        if epoch > 0 and epoch % log_every == 0:
            if rank == 0:
                print(f'Loss: {loss.item()}')
                torch.save(model.state_dict(), f"./checkpoints/model_{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", default=None)
    parser.add_argument("--batch-size", default=32)
    parser.add_argument("--n-epochs", default=50)
    parser.add_argument("--log-every", default=5)
    args = parser.parse_args()


    if torch.cuda.is_available():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size, args, ), nprocs=world_size, join=True)
    else:
        train_cpu()