from torch.utils.data import Dataset
from torchvision import transforms
import os
from utils.mytransforms import *
import json


class PolypDataSet(Dataset):
    def __init__(self, root, data_dir, mode='train', ratio=10, sign='label', transform=None):
        super(PolypDataSet, self).__init__()
        self.mode = mode
        self.sign = sign
        root_path = os.path.join(root, data_dir)
        imgfile = os.path.join(root_path, mode + '_polyp.json')
        with open(imgfile, 'r') as f:
            imglist = json.load(f)

        imglist = [os.path.join(root_path, mode, img) for img in imglist]

        if mode == 'train' and ratio < 10:
            split_file_path = os.path.join(root_path, "train_split_10.json")
            with open(split_file_path, 'r') as f:
                split_10_list = json.load(f)

            if sign == 'label':
                Limglist = []
                for i in range(ratio):
                    Limglist += split_10_list[str(i)]
                self.imglist = [imglist[index] for index in Limglist]
            if sign == 'unlabel':
                Uimglist = []
                for i in range(ratio, 10):
                    Uimglist += split_10_list[str(i)]
                self.imglist = [imglist[index] for index in Uimglist]
        else:
            self.imglist = imglist

        if transform is None:
            if mode == 'train' and sign == 'label':
               transform = transforms.Compose([
                   Resize((320, 320)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   RandomCrop((256, 256)),
                   ToTensor()
               ])
            elif mode == 'train' and sign == 'unlabel':
                transform = transforms.Compose([
                    transforms.Resize((320, 320)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.RandomCrop((256, 256)),
                    transforms.ToTensor()
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((320, 320)),
                   ToTensor()
                ])
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train' and self.sign == 'unlabel':
            img_path = self.imglist[index]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                return self.transform(img)
        else:
            img_path = self.imglist[index]
            gt_path = img_path.replace('images', 'masks')
            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')
            data = {'image': img, 'label': gt}
            if self.transform:
                data = self.transform(data)

            return data

    def __len__(self):
        return len(self.imglist)
