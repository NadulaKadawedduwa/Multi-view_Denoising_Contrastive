import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataset
from smaller_model import UNet


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    dataset = CustomDataset(device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = UNet().to(device=device)

    lr = 1e-3
    optim = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    log_every = 10

    n_epochs = 10

    for epoch in range(n_epochs):
        for noisy, clean in tqdm(dataloader):
            optim.zero_grad()
            pred = model(noisy.to(torch.float))

            loss = loss_fn(pred, clean)
            loss.backward()
            optim.step()

            if epoch > 0 and epoch % log_every == 0:
                print(f'Loss: {loss.item()}')
    
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    train()