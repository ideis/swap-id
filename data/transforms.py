import torch
from torchvision.transforms import *

def transform_train(img):
    transform = Compose([
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        ToTensor(),
        Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    img_tensor = transform(img)
    return img_tensor

def transform_val(img):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    img_tensor = transform(img)
    return img_tensor

def transform_inv(img_tensor):
    transform = Compose([
        Normalize(mean=[-1.] * 3,
                  std=[2.] * 3),
        ToPILImage()
    ])
    img = transform(img_tensor)
    return img

