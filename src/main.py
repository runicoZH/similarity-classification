
import os

from dataset import ClassificationDataset

from torch.utils.data import DataLoader
from PIL import Image


DATASET_PATH = "resources/dataset"
CLASSES = ["Arts, Crafts & Sewing", "Electronics"]
if __name__ == "__main__":
    print(os.getcwd())
    train_set = ClassificationDataset(os.path.join(DATASET_PATH, "train.csv"),
                                      os.path.join(DATASET_PATH, "train/train"),
                                      CLASSES)
    print(len(train_set))
    im, label = train_set.__getitem__(21)
    im.show()