import os
import numpy as np
import cv2
import pandas as pd
from time import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from tqdm import tqdm

from model import ConvNetModel
from const import MODEL_PATH


class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform = None):
        self.dataframe = dataframe.values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_name, label = self.dataframe[index]
        img_path = os.path.join(self.img_dir, img_name + ".png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def main():
    # .csv file directory
    train_ds_path = "./APTOS Dataset/"
    
    # .csv file
    train_file = "train.csv"

    # Image directory
    train_img_dir = "./APTOS Dataset/train_images"

    # Change data type
    train_df = pd.read_csv(train_ds_path + train_file)
    train_df["id_code"] = train_df["id_code"].astype(str)
    train_df["diagnosis"] = train_df["diagnosis"].astype(int)

    train_batch_size = 64   # Training data batch size
    valid_batch_size = 64   # Training data batch size
    img_size = 224          # Image size

    # Transform input data
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_data = CustomDataset(dataframe = train_df, img_dir = train_img_dir, transform = train_transforms)
    train_dataloader = DataLoader(train_data, batch_size = train_batch_size)
    
    # Compute mean and standard deviation for transforming input data to model
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0

    for images in tqdm(train_dataloader):
        images = images[0]
        batch_samples = images.size(0)
        
        mean += images.mean([0, 2, 3]) * batch_samples
        std += images.std([0, 2, 3]) * batch_samples
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples

    mean = tuple([t.item() for t in mean])
    std = tuple([t.item() for t in std])


    valid_size = 0.2
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split : ], indices[ : split]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std),
        transforms.RandomHorizontalFlip()
    ])

    train_data = CustomDataset(dataframe = train_df, img_dir = train_img_dir, transform = train_transforms)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataloader = DataLoader(train_data, batch_size = train_batch_size, sampler = train_sampler)
    valid_dataloader = DataLoader(train_data, batch_size = valid_batch_size, sampler = valid_sampler)

    # Run on GPU if CUDA is available else run in CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNetModel().to(device)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)

    epochs = 100                    # Train the model for 100 epochs
    history = []
    valid_loss_min = float("inf")   # Monitor validation loss

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}]", end = "\r")
        start_time = time()
        
        model.train()
        train_loss, val_loss, val_acc = [], [], []

        result = {}
        
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            train_loss.append(loss)

            print(f"Epoch [{epoch + 1}] Train Loss: {torch.stack(train_loss).mean():.4f}", end = "\r")
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        result["train_loss"] = torch.stack(train_loss).mean()

        with torch.no_grad():
            model.eval()

            for images, labels in valid_dataloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                val_loss.append(loss)
                
                _, predicted = torch.max(outputs, dim = 1)
                acc = torch.tensor(torch.sum(predicted == labels).item() / len(predicted))
                val_acc.append(acc)

                print(f"Epoch [{epoch + 1}] Train Loss: {result["train_loss"]:.4f} Val Loss: {torch.stack(val_loss).mean():.4f} Val Accuracy: {torch.stack(val_acc).mean():.4f}", end = "\r")
        
        end_time = time()

        result["val_loss"] = torch.stack(val_loss).mean()
        result["val_acc"] = torch.stack(val_acc).mean()
        
        history.append(result)
    
        print(f"Epoch [{epoch + 1}] Train Loss: {result["train_loss"]:.4f} Val Loss: {result["val_loss"]:.4f} Val Accuracy: {result["val_acc"]:.4f} Time: {(end_time - start_time):.4f}s", end = "\n")
        
        if result["val_loss"] < valid_loss_min:
            print("Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(valid_loss_min, result["val_loss"]), end = "\n")
            torch.save(model.state_dict(), MODEL_PATH)
            valid_loss_min = result["val_loss"]



if __name__ == "__main__":
    main()