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


# comparison of hidden layers sizes
class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = nn.Linear(58, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.dense4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.2)
        self.dense5 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = self.dropout1(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout2(x)
        x = torch.relu(self.dense3(x))
        x = self.dropout3(x)
        x = torch.relu(self.dense4(x))
        x = self.dropout4(x)
        x = self.dense5(x)
        return torch.softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.CrossEntropyLoss()(y_pred, y)

        pred = torch.argmax(y_pred, dim=1)
        acc = torch.sum(pred == y).item() / len(pred)

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.CrossEntropyLoss()(y_pred, y)

        pred = torch.argmax(y_pred, dim=1)
        acc = torch.sum(pred == y).item() / len(pred)

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, on_step=False, prog_bar=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer


final_data = pd.read_csv("data/music_data.csv")
final_data = final_data.drop(labels='filename', axis=1)

class_list = final_data.iloc[:, -1]
convertor = LabelEncoder()
y = convertor.fit_transform(class_list)

fit = StandardScaler()
X = fit.fit_transform(np.array(final_data.iloc[:, :-1], dtype=float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

model = MyModel()

trainer = pl.Trainer(max_epochs=500, enable_model_summary=True)
trainer.fit(model, train_loader, test_loader)
