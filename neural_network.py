import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import pickle


class MyDataset(Dataset):
    def __init__(self, x__, y__):
        self.x = torch.from_numpy(x__).float()
        self.y = torch.from_numpy(y__).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(58, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 10),
        )
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        x_, y_ = batch
        y_pred = self(x_)
        loss = nn.CrossEntropyLoss()(y_pred, y_)
        pred = torch.argmax(y_pred, dim=1)
        acc = torch.sum(pred == y_).item() / len(pred)
        self.log(f'val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f'val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)
        self.val_preds.extend(pred.detach().cpu().numpy().tolist())
        self.val_labels.extend(y_.detach().cpu().numpy().tolist())
        return loss

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []

    def on_validation_epoch_end(self):
        # Convert lists to tensors
        val_preds_tensor = torch.tensor(self.val_preds)
        val_labels_tensor = torch.tensor(self.val_labels)
        # Calculate confusion matrix
        cm = confusion_matrix(val_labels_tensor.cpu().numpy(), val_preds_tensor.cpu().numpy())
        # Store the confusion matrix in an instance variable
        self.confusion_matrix = cm

    def shared_step(self, batch, prefix):
        x_, y_ = batch
        y_pred = self(x_)
        loss = nn.CrossEntropyLoss()(y_pred, y_)
        pred = torch.argmax(y_pred, dim=1)
        acc = torch.sum(pred == y_).item() / len(pred)
        self.log(f'{prefix}_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f'{prefix}_acc', acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.00055, weight_decay=1e-5)


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, filepath, batch_size=32, num_workers=2):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.filepath = filepath
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        data = pd.read_csv(self.filepath)
        data = data.drop(labels='filename', axis=1)

        encoder = LabelEncoder()
        y = encoder.fit_transform(data.iloc[:, -1])
        scaler = StandardScaler().fit(np.array(data.iloc[:, :-1], dtype=float))

        # Save the encoder
        with open('model/encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)

        # Save the scaler
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        x = scaler.transform(np.array(data.iloc[:, :-1], dtype=float))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        self.train_dataset = MyDataset(x_train, y_train)
        self.test_dataset = MyDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=25,
    verbose=True,
    mode='min'
)


class ConfusionMatrixCallback(pl.Callback):
    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            cm = pl_module.confusion_matrix
            # Plot confusion matrix
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion matrix")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()


model = MyModel()
data_module = MusicDataModule("data/music_data.csv", num_workers=4)

trainer = pl.Trainer(callbacks=[early_stop_callback, ConfusionMatrixCallback()], accelerator="gpu", devices=1,
                     max_epochs=500, log_every_n_steps=4)
trainer.fit(model, data_module)

# Save the model
torch.save(model.state_dict(), "model/model.pth")
