import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split

music_genres = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.labels = [music_genres[genre] for genre in self.labels]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features.iloc[idx].values)
        y = torch.tensor(self.labels[idx])
        return x, y


# LightningDataModule
class MyDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size=32, test_size=0.2, random_state=42):
        super(MyDataModule, self).__init__()
        self.df = None
        self.val_dataset = None
        self.train_dataset = None
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(self):
        # Load the CSV file into a pandas DataFrame
        self.df = pd.read_csv(self.csv_file)

    def setup(self, stage=None):
        # Split the DataFrame into features and labels
        features = self.df.iloc[:, 1:-1]  # Exclude the first column (filename) and the last column (label)
        labels = self.df.iloc[:, -1]  # Get the last column as labels

        # Split the features and labels into training and validation sets
        features_train, features_val, labels_train, labels_val = train_test_split(
            features, labels, test_size=self.test_size, random_state=self.random_state)

        # Create custom datasets for training and validation
        self.train_dataset = MyDataset(features_train, labels_train)
        self.val_dataset = MyDataset(features_val, labels_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


# Lightning Module
class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model architecture here
        self.fc = nn.Sequential(
            nn.Linear(58, 128, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(128, 64, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(64, 10, dtype=torch.float64),
        )

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)

        # Calculate accuracy
        _, predicted_labels = torch.max(y_hat, dim=1)
        accuracy = (predicted_labels == y).sum().item() / y.size(0)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', accuracy, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)

        # Calculate accuracy
        _, predicted_labels = torch.max(y_hat, dim=1)
        accuracy = (predicted_labels == y).sum().item() / y.size(0)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


# Initialize the data module
data_module = MyDataModule(csv_file='data/music_data.csv', batch_size=32)

# Initialize the model
model = MyModel()

# Initialize a trainer
trainer = pl.Trainer(max_epochs=1000)

# Train the model
trainer.fit(model, data_module)
