import cv2
import glob
import os
import random

from PIL import Image
from torch.utils.data import TensorDataset

class FFHQ(TensorDataset):
    def __init__(self, data_path, transform, same_person_prob=0.8):
        self.images = list()
        self.same_person_prob = same_person_prob
        self.images = glob.glob(f'{data_path}/*.*g')
        self.transform = transform

    def __getitem__(self, item):
        image_path = self.images[item]
        Xs = cv2.imread(image_path)
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_person_prob:
            image_path = self.images[random.randint(0, len(self.images) - 1)]

            Xt = cv2.imread(image_path)
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transform(Xs), self.transform(Xt), same_person

    def __len__(self):
        return len(self.images)
