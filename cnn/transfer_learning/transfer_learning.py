import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
from basic_cnn import BasicCNN  # Import your BasicCNN model

class TransferLearning(pl.LightningModule):
    def __init__(self, data_path, batch_size, lr):
        super().__init__()
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        
        # Data preparation
        dataset = CIFAR10(data_path, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]), download=True)

        dataset_size = len(dataset)
        train_size = int(dataset_size * .95)
        val_size = dataset_size - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Load the pre-trained model
        self.model = BasicCNN(num_classes=10)  # Ensure the model has 10 output classes for CIFAR-10
        
        # Initialize weights with the pre-trained model
        self.model.load_state_dict(torch.load("basic_cnn_reg_mdl.pth"))

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Accuracy metric
        self.accuracy = torchmetrics.Accuracy(num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          shuffle=False)
