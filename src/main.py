
import os

from dataset import ClassificationDataset
from model import initialize_resnet_model
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

import torch
import torch.nn as nn


DATASET_PATH = "resources/dataset"
CLASSES = ["Arts, Crafts & Sewing", "Electronics"]
EPOCHS = 10


if __name__ == "__main__":
    
    train_set = ClassificationDataset(os.path.join(DATASET_PATH, "train.csv"),
                                      os.path.join(DATASET_PATH, "train/train"),
                                      CLASSES)
    
    train_set, validation_set = random_split(train_set, [0.8, 0.2])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=1)

    model = initialize_resnet_model(len(CLASSES))

    optimizer = Adam(model.parameters(), lr=0.01)

    loss_function = nn.CrossEntropyLoss()

    model.cuda()
    for i in EPOCHS:
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in val_loader:
                output = model(x)
                prediction = torch.max(output, 1)
                total += y.size(0)
                correct += (prediction == y).sum().item()

            print(f"accuracy on epoch {i}: {(correct/total)*100}")

            

