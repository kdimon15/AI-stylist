import math
import os
import random
import time
import cv2
import numpy as np
import torch
import pandas as pd
from Config import CFG
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from contextlib import contextmanager
from torch.utils.data import Dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, clothes_model, face_model, criterion, optimizer, device):
    losses = AverageMeter()
    face_model.train()
    clothes_model.train()
    for step, (clothes, faces_true, faces_false) in enumerate(train_loader):
        batch_size = clothes.size(0)

        clothes = clothes.to(device)
        clothes_preds = clothes_model(clothes)

        faces_true = faces_true.to(device)
        face_true_preds = face_model(faces_true)

        faces_false = faces_false.to(device)
        face_false_preds = face_model(faces_false)

        loss = criterion(clothes_preds, face_true_preds, face_false_preds)

        losses.update(loss.item(), batch_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return losses.avg


def valid_fn(valid_loader, clothes_model, face_model, criterion, device):
    losses = AverageMeter()
    face_model.eval()
    clothes_model.eval()
    for step, (clothes, faces_true, faces_false) in enumerate(valid_loader):
        batch_size = clothes.size(0)

        clothes = clothes.to(device)
        clothes_preds = clothes_model(clothes)

        faces_true = faces_true.to(device)
        face_true_preds = face_model(faces_true)

        faces_false = faces_false.to(device)
        face_false_preds = face_model(faces_false)

        loss = criterion(clothes_preds, face_true_preds, face_false_preds)
        losses.update(loss.item(), batch_size)
    return losses.avg


def get_transforms(data='', purpose='clothes'):
    if purpose == 'clothes':
        size = CFG.clothes_size
    else:
        size = CFG.faces_size
    if data == 'train':
        return albu.Compose([
            albu.RandomResizedCrop(size, size, scale=(0.9, 1.0)),
            albu.HorizontalFlip(),
            albu.Rotate(p=0.5),
            albu.Blur(blur_limit=5, p=0.15),
            albu.RandomBrightnessContrast(p=0.15),
            albu.HueSaturationValue(p=0.15),
            albu.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif data == 'valid':
        return albu.Compose([
            albu.Resize(size, size),
            albu.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def init_logger(log_file="train.log"):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


class TrainDataset(Dataset):
    def __init__(self, df, faces_transform=None, clothes_transform=None):
        self.face_true_paths = df.face_true.values
        self.clothes_paths = df.clothes.values
        self.faces_transform = faces_transform
        self.clothes_transform = clothes_transform

    def __len__(self):
        return len(self.face_true_paths)

    def __getitem__(self, idx):
        face_true_path = self.face_true_paths[idx]
        face_true_image = cv2.imread(face_true_path)
        face_true_image = cv2.cvtColor(face_true_image, cv2.COLOR_BGR2RGB)

        face_false_path = f'preprocessed_dataset/{random.choice(os.listdir("preprocessed_dataset"))}/face.jpg'
        face_false_image = cv2.imread(face_false_path)
        face_false_image = cv2.cvtColor(face_false_image, cv2.COLOR_BGR2RGB)

        clothes_path = self.clothes_paths[idx]
        clothes_image = cv2.imread(clothes_path)
        clothes_image = cv2.cvtColor(clothes_image, cv2.COLOR_BGR2RGB)

        if self.faces_transform:
            face_true_image = self.faces_transform(image=face_true_image)['image']
            face_false_image = self.faces_transform(image=face_false_image)['image']
            clothes_image = self.clothes_transform(image=clothes_image)['image']
        return clothes_image, face_true_image, face_false_image


def make_train_df():
    face_true, clothes = [], []
    for path1 in os.listdir('preprocessed_dataset'):
        for path2 in os.listdir(f'preprocessed_dataset/{path1}/'):
            if path2 != 'face.jpg':
                clothes.append(f'data/{path1}/{path2}')
                face_true.append(f'data/{path1}/face.jpg')

    df = pd.DataFrame({
        'clothes': clothes,
        'face_true': face_true,
    })
    df.to_csv('train.csv', index=False)


class ClothesDataset(Dataset):
    def __init__(self, clothes_paths, transform=None):
        self.clothes_paths = clothes_paths
        self.transform = transform

    def __len__(self):
        return len(self.clothes_paths)

    def __getitem__(self, idx):
        clothes_path = self.clothes_paths[idx]
        clothes_image = cv2.imread(clothes_path)
        clothes_image = cv2.cvtColor(clothes_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            clothes_image = self.transform(image=clothes_image)['image']
        return clothes_image


class FaceDataset(Dataset):
    def __init__(self, face_paths, transform=None):
        self.face_paths = face_paths
        self.transform = transform

    def __len__(self):
        return len(self.face_paths)

    def __getitem__(self, idx):
        face_path = self.face_paths[idx]
        face_image = cv2.imread(face_path)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            face_image = self.transform(image=face_image)['image']
        return face_image