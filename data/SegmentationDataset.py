import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os

class SegmentationDataset(data.Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.names = os.listdir(img_dir)

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        X = Image.open(self.img_dir + self.names[index]).convert('RGB')
        y = Image.open(self.mask_dir + self.names[index])
        
        return self.transform(X), self.transform(y)

    def __len__(self):
        return len(self.names)
