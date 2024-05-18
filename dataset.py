from torch.utils.data import Dataset
import torchvision.transforms as T
import PIL.Image as Image
import numpy as np
import torch
import random

class CustomDataset(Dataset):
    def __init__(self, img_list_file, transforms=None, std=None, crop_size=224, device='cpu', sub_set=None, is_val=False, rand_sub=True, stereo=False) -> None:
        super().__init__()
        with open(img_list_file, 'r') as f:
            self.img_paths = f.readlines()
        self.img_paths = [img_path.strip() for img_path in  self.img_paths]
        if sub_set:
            sub_set = min(sub_set, len(self.img_paths))
            if rand_sub:
                self.img_paths = random.sample(self.img_paths, sub_set)
            else:
                self.img_paths = self.img_paths[:sub_set]
        self.crop_size = crop_size
        self.transform = transforms or  T.Compose([
                T.ToTensor(),
                T.Resize((256, 256)),
                T.CenterCrop(size=self.crop_size)
            ])
        self.std = std
        self.device = device
        self.perspective = T.RandomPerspective(distortion_scale=0.4, p=1)
        self.is_val = is_val
        self.stereo = stereo
    
    def __add_noise__(self, img):
        std = self.std or np.random.uniform(0, 1)
        img = np.array(img)

        noise = np.random.normal(0, std, img.shape)
        noisy_img = np.clip(img + noise, 0, 255)
        return noisy_img
    
    def __get_img_pair__(self, img):        
        warped_img = self.perspective(img)

        scr = self.transform(self.__add_noise__(img))
        warped_img = self.transform(self.__add_noise__(warped_img))

        scr = scr.to(dtype=torch.float, device=self.device)
        warped_img = warped_img.to(dtype=torch.float, device=self.device)
        
        return scr, warped_img
    
    def __len__(self):
        return len(self.img_paths)
    
    def __get_noisy_pair__(self, img, warped_img):
        noisy_img = self.transform(self.__add_noise__(img))
        noisy_warped_img = self.transform(self.__add_noise__(warped_img))

        noisy_img = noisy_img.to(dtype=torch.float, device=self.device)
        noisy_warped_img = noisy_warped_img.to(dtype=torch.float, device=self.device)

        return noisy_img, noisy_warped_img
    
    def __sterio_pair__(self, idx):
        img_path = self.img_paths[idx].strip()
        if idx + 1 < len(self.img_paths):
            img_path2 = self.img_paths[idx+1].strip()
        else:
            img_path2 = self.img_paths[0].strip()
        
        img1 = Image.open(img_path).convert("L")
        img2 = Image.open(img_path2).convert("L")

        img1_n = self.transform(self.__add_noise__(img1)).to(dtype=torch.float, device=self.device)
        img2_n = self.transform(self.__add_noise__(img2)).to(dtype=torch.float, device=self.device)

        img1 = self.transform(img1).to(dtype=torch.float, device=self.device)
        img2 = self.transform(img2).to(dtype=torch.float, device=self.device)
        
        source = torch.cat((img1_n, img2_n), dim=0)
        target = torch.cat((img1, img2), dim=0)

        return source, target
    
    def __getitem__(self, idx):
        if self.stereo:
            return self.__sterio_pair__(idx)
        
        img_path = self.img_paths[idx].strip()

        clean = Image.open(img_path).convert("L")
        clean_warped = self.perspective(clean)

        src, src_warped = self.__get_noisy_pair__(clean, clean_warped)
        target, target_warped = self.__get_noisy_pair__(clean, clean_warped)

        src = torch.cat((src, src_warped), dim=0)
        
        if self.is_val:
            clean = self.transform(clean).to(dtype=torch.float, device=self.device)
            clean_warped = self.transform(clean_warped).to(dtype=torch.float, device=self.device)
            target = torch.cat((clean, clean_warped), dim=0)
        else:
            target = torch.cat((target, target_warped), dim=0)
        
        return src, target