import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd 
import numpy as np


class ChestXRayDataset(Dataset):
    def __init__(self, root: str, train=False, transform=None, convert='RGB'):
        super().__init__()
        if train:
            self.img_dir =os.path.join(root, "train_val")
        else:
            self.img_dir =os.path.join(root, "test")

        # keep only first two columns: 
        # 'Image Index' contains image name
        # 'Finding Labels' contains classification label
        ann_data = pd.read_csv(os.path.join(root, "annotations_file.csv"), usecols=[0, 1])

        # remove annotation data of files not present in image subdirectory
        img_list = os.listdir(self.img_dir)
        self.ann_data = ann_data[ann_data['Image Index'].isin(pd.Series(img_list))]
        assert self.ann_data.shape[0] == len(img_list)
        self.transform = transform
        self.convert = convert
        self.classes = np.array(['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'])
    

    def __len__(self):
        return self.ann_data.shape[0]
    
    def __getitem__(self, idx):
        img_dir = os.path.join(self.img_dir, self.ann_data.iloc[idx, 0]) # type: ignore
        img = Image.open(img_dir).convert(self.convert)

        # create multi-hot tensor of 
        labels = self.ann_data.iloc[idx, 1].split('|') # type: ignore
        enc_label = np.zeros(len(self.classes))
        for label in labels:
            enc_label[np.where(self.classes == label)] = 1

        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(enc_label).float()
    