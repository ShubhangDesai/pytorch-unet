import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .SegmentationDataset import *

def get_loader(img_dir, mask_dir, batch_size):
    seg_dataset = SegmentationDataset(img_dir, mask_dir)
    loader = torch.utils.data.DataLoader(seg_dataset, 
                                         batch_size=batch_size,
                                         num_workers=1,
                                         shuffle=True)

    return loader

def get_test_loader(img_dir, batch_size):
    dataset = datasets.ImageFolder(root=img_dir, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=1,
                                         shuffle=True
                                        )

    return loader
