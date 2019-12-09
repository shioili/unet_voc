import os
import numpy as np
from PIL import Image
import torch
import torchvision

VOC_PATH = "./VOCdevkit/VOC2012"

def get_dataloader(batch_size, crop_size=(256, 256), shuffle=True):
    dataset = VOCDataset(crop_size=crop_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, crop_size=(256, 256), path=VOC_PATH):
        self.crop_size = crop_size
        with open("./ImageSets/train.txt") as f:
            lines = f.readlines()
            self.img_files = map(
                    lambda l: os.path.join(path, "JPEGImages", l.replace('\n', '.jpg')),
                    lines)
            self.seg_files = map(
                    lambda l: os.path.join(path, "SegmentationClass", l.replace('\n', '.png')),
                    lines)

            def size_checker(f):
                img = Image.open(f)
                w, h = img.size
                return w >= crop_size[0] & h >= crop_size[1]

            self.img_files = filter(size_checker, self.img_files)
            self.seg_files = filter(size_checker, self.seg_files)
            self.img_files = list(self.img_files)
            self.seg_files = list(self.seg_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        seg = Image.open(self.seg_files[idx])

        t, l, h, w = torchvision.transforms.RandomCrop.get_params(img, output_size=self.crop_size)
        img = torchvision.transforms.functional.crop(img, t, l, h, w)
        seg = torchvision.transforms.functional.crop(seg, t, l, h, w)

        toTensor = torchvision.transforms.ToTensor()
        img = toTensor(img)
        #normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #img = normalize(toTensor(img))

        seg = torch.tensor(np.array(seg), dtype=torch.long)
        seg[seg==255] = 0   # Make 'void' class to 'background'

        return img, seg
