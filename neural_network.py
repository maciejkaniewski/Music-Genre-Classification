import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pytorch_lightning as pl


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(58, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def shared_step(self, batch, prefix):
        x_, y_ = batch
        y_pred = self(x_)
        loss = nn.CrossEntropyLoss()(y_pred, y_)
        pred = torch.argmax(y_pred, dim=1)
        acc = torch.sum(pred == y_).item() / len(pred)
        self.log(f'{prefix}_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer


final_data = pd.read_csv("data/music_data.csv")
final_data = final_data.drop(labels='filename', axis=1)

class_list = final_data.iloc[:, -1]
convertor = LabelEncoder()
y = convertor.fit_transform(class_list)

fit = StandardScaler()
x = fit.fit_transform(np.array(final_data.iloc[:, :-1], dtype=float))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=12)

model = MyModel()

trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=500, log_every_n_steps=25)
trainer.fit(model, train_loader, test_loader)
