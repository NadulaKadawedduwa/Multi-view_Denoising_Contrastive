from torch.utils.data import Dataset
import torchvision.transforms as T
import PIL.Image as Image
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, img_list_file='/mnt/data/ILSVRC/Data/train_img_paths.txt', transforms=None, std=None, crop_size=224, device='cpu') -> None:
        super().__init__()
        with open(img_list_file, 'r') as f:
            self.img_paths = f.readlines()
        self.crop_size = crop_size
        self.transform = transforms or  T.Compose([
                T.ToTensor(),
                T.CenterCrop(size=224)
            ])
        self.std = std
        self.device = device
    
    def __add_noise__(self, img):
        std = self.std or np.random.uniform(0, 1)
        img = np.array(img)

        noise = np.random.normal(0, std, img.shape)
        noisy_img = np.clip(img + noise, 0, 255)
        return noisy_img
    
    def __get_img__(self, idx):
        img_path = self.img_paths[idx].strip()
        img = Image.open(img_path).convert("L")
        noisy_img = self.__add_noise__(img)

        img = self.transform(img)
        noisy_img = self.transform(noisy_img)

        return img, noisy_img
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_1, noisy_1 = self.__get_img__(idx)

        # get a random image
        target_idx = np.random.randint(0, len(self.img_paths))
        img_2, noisy_2 = self.__get_img__(target_idx)

        noisy = torch.cat((noisy_1, noisy_2)).to(torch.float)
        clean = torch.cat((img_1, img_2), dim=0).to(torch.float)
        return noisy.to(device=self.device), clean.to(device=self.device)