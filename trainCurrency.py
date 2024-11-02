"""
    @author: Nguyen "sh1nata" Duc Tri <tri14102004@gmail.com>
"""
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.databuild import VNCurrencyDataset
from src.vgg import *
from src.utils import *
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix





def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = A.Compose([
        A.Resize(width=args.image_size, height=args.image_size),
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
        A.Resize(width=args.image_size, height=args.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=55.0),
        ToTensorV2(),
    ])


    train_set = VNCurrencyDataset(root=args.data_pathm, train=True, transform = train_transform)
    valid_set = VNCurrencyDataset(root=args.data_pathm, train=False, transform = test_transform)

    training_params = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "drop_last": True,
        "shuffle": True,
    }

    valid_params = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "drop_last": False,
        "shuffle": False,
    }

    training_dataloader = DataLoader(train_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **training_params)

    model = VGG16_ConfigC(num_classes=len(train_set.currencyList).to(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    scheduler = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
    else:
        start_epoch = 0
        best_acc = 0

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.tensorboard_path)

    total_iters = len(training_dataloader)
    for epoch in range(start_epoch, args.epochs):
        # Training Phase
        model.train()
        losses = []
        progress_bar = tqdm(training_dataloader, colour='cyan')
        for iter, (images, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            loss = criterion(prediction, labels)
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            progress_bar.set_description("Epoch {}/{}. Loss value: {:.4f}".format(epoch + 1, args.epochs, loss_val))
            losses.append(loss_val)
            writer.add_scalar("Train/Loss", np.mean(losses), epoch * total_iters + iter)

        model.eval()
        losss = []
        all_predictions = []
        all_ground_truth = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                prediction = model(images)
                maxVal_ofIdx = torch.argmax(prediction, dim=1)
                loss = criterion(prediction, labels)
                losss.append(loss.item())
                all_ground_truth.extend(labels.tolist())
                all_predictions.extend(maxVal_ofIdx.tolist())

        writer.add_scalar("Validation/Loss", np.mean(losss), epoch)
        acc = accuracy_score(all_ground_truth, all_predictions)
        writer.add_scalar("Validation/Accuracy", acc, epoch)
        conf_matrix = confusion_matrix(all_ground_truth, all_predictions)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(len(train_set.currencyList))], epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "batch_size": args.batch_size,
        }

        torch.save(checkpoint, os.path.join(args.VNCur_trained_models, "lastVNCur.pth"))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.VNCur_trained_models, "bestVNCur.pth"))
            best_acc = acc
        scheduler.step()

if __name__ == '__main__':
    args = get_args()
    train(args)