
import os

import pandas as pd

from typing import Optional, List, Tuple
from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """Dataset for basic classification

    Attributes
    ----------
    class_map : dict[str, int]
        maps class name to int label
    images : List[Image]
        list of all images
    labels : List[int]
        list of all labels, index matches images
    """
    def __init__(self, annotation_file_path : str, image_path : str, classes : List[str]):
        """initializes dataset

        Parameters
        ----------
        annotation_file_path : str
            path to annotation_file
        image_path : str
            path to image folder
        classes : List[str]
            list of class names used
        """
        df = pd.read_csv(annotation_file_path)

        self.class_map = {class_name : i for i, class_name in enumerate(classes)}
        n_error = 0
        self.images = []
        self.labels = []
        for idx, row in df.iterrows():
            class_name = row["categories"]
            img_name = row["ImgId"]
            if class_name in self.class_map:
                try:
                    image = Image.open(os.path.join(image_path, f"{img_name}.jpg"))
                except:
                    n_error += 1
                    print(f"error loading image: {img_name}, this happened {n_error} times")
                    continue
                self.images.append(image)
                self.labels.append(self.class_map[class_name])

    def __len__(self) -> int:
        return(len(self.labels))
    
    def __getitem__(self, index : int):
        return self.images[index], self.labels[index]