import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from Config import CFG
from Model import CustomModel
from utils import make_train_df, TrainDataset, get_transforms, device, LOGGER, train_fn, valid_fn
import pandas as pd
from sklearn.model_selection import train_test_split

make_train_df()
df = pd.read_csv('train.csv')
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=CFG.seed)

train_dataset = TrainDataset(train_df,
                             clothes_transform=get_transforms(data='train', purpose='clothes'),
                             faces_transform=get_transforms(data='train', purpose='faces'))
train_loader = DataLoader(train_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers,
                          pin_memory=False)

valid_dataset = TrainDataset(valid_df,
                             clothes_transform=get_transforms(data='valid', purpose='clothes'),
                             faces_transform=get_transforms(data='valid', purpose='faces'))
valid_loader = DataLoader(valid_dataset,
                          CFG.batch_size // 2,
                          shuffle=True,
                          num_workers=CFG.num_workers,
                          pin_memory=False)

face_model = CustomModel()
face_model.to(device)

clothes_model = CustomModel()
clothes_model.to(device)

optimizer = optim.Adam(list(face_model.parameters()) + list(clothes_model.parameters()), lr=CFG.lr,
                       weight_decay=CFG.weight_decay, amsgrad=False)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
criterion = nn.TripletMarginLoss(margin=1.0, p=2)

best_loss = np.inf
for epoch in range(CFG.epochs):
    print(f'Epoch: {epoch + 1}')
    start_time = time.time()
    avg_loss = train_fn(train_loader, clothes_model, face_model, criterion, optimizer, device)
    avg_val_loss = valid_fn(valid_loader, clothes_model, face_model, criterion, device)

    scheduler.step(avg_loss)

    elapsed = time.time() - start_time
    LOGGER.info(
        f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f} avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s")
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        LOGGER.info(f"Epoch {epoch + 1} - Save Best loss: {best_loss:.4f} Model")

        torch.save(face_model.state_dict(), "Weights/face_model.pth")
        torch.save(clothes_model.state_dict(), "Weights/clothes_model.pth")
