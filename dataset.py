from torch.utils.data import Dataset
import torchvision.transforms as T
import PIL.Image as Image
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, img_list_file, transforms=None, std=None, crop_size=224, device='cpu', sub_set=None) -> None:
        super().__init__()
        with open(img_list_file, 'r') as f:
            self.img_paths = f.readlines()
        self.img_paths = [img_path.strip() for img_path in  self.img_paths]
        if sub_set:
            self.img_paths = self.img_paths[:sub_set]
        self.crop_size = crop_size
        self.transform = transforms or  T.Compose([
                T.ToTensor(),
                T.CenterCrop(size=224)
            ])
        self.std = std
        self.device = device
        self.perspective = T.RandomPerspective(distortion_scale=0.4, p=1)
    
    def __add_noise__(self, img):
        std = self.std or np.random.uniform(0, 1)
        img = np.array(img)

        noise = np.random.normal(0, std, img.shape)
        noisy_img = np.clip(img + noise, 0, 255)
        return noisy_img
    
    def __get_img_pair__(self, img_path, add_noise=False):        
        img = Image.open(img_path).convert("L")
        warped_img = self.perspective(img)

        if add_noise:
            img = self.__add_noise__(img)
            warped_img = self.__add_noise__(warped_img)

        img = self.transform(img).to(dtype=torch.float, device=self.device)
        warped_img = self.transform(warped_img).to(dtype=torch.float, device=self.device)
        
        return img, warped_img
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx].strip()

        clean, clean_warped = self.__get_img_pair__(img_path)
        noisy, noisy_warped = self.__get_img_pair__(img_path, add_noise=True)

        clean = torch.cat((clean, clean_warped), dim=0)
        noisy = torch.cat((noisy, noisy_warped), dim=0)

        return clean, noisy