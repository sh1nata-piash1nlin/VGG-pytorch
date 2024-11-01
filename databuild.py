# -*- coding: utf-8 -*-
"""
    @author: Nguyen "sh1nata" Duc Tri <tri14102004@gmail.com>
"""
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VNCurrencyDataset(Dataset):
    def __init__(self, root="", train=True, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.currencyList = []
        self.labels = []
        self.image_paths = []
        self.transform = transform

        if train:
            datafile_path = os.path.join(root, "currency_train")
        else:
            datafile_path = os.path.join(root, "currency_test")

        for filename in os.listdir(datafile_path):
            if os.path.isdir(os.path.join(datafile_path, filename)):
                self.currencyList.append(filename)


        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")  # List of valid image extensions
        for category in self.currencyList:
            category_path = os.path.join(datafile_path, category)
            for image_name in os.listdir(category_path):
                if image_name.endswith(valid_extensions):
                    image_path = os.path.join(category_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(self.currencyList.index(category))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        label = self.labels[idx]
        return image, label

if __name__ == "__main__":
    train_size = 224
    train_transform = A.Compose([
        A.Resize(width=train_size, height=train_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(),
        A.Sharpen(),
        A.RGBShift(),
        A.Cutout(num_holes=5, max_h_size=25, max_w_size=25, fill_value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=55.0),
        # mean and std of ImageNet
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(width=train_size, height=train_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=55.0),
        ToTensorV2(),
    ])

    dataset  = VNCurrencyDataset(root="../../data/VNCurrency", train=True, transform=train_transform)
    # image, label = dataset.__getitem__(123)
    # print(image.shape)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=4)
    for images, labels in dataloader:
        print(images.shape)
        print(labels)



