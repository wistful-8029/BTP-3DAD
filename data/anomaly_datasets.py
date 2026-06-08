import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    PRESETS = {
        "Real3D": [
            'airplane', 'candybar', 'car', 'chicken', 'diamond', 'duck',
            'fish', 'gemstone', 'seahorse', 'shell', 'starfish', 'toffees'
        ],
        "AnomalyShapeNet": [
            'ashtray0', 'bag0', 'bottle0', 'bottle1',
            'bottle3', 'bowl0', 'bowl1', 'bowl2',
            'bowl3', 'bowl4', 'bowl5', 'bucket0',
            'bucket1', 'cap0', 'cap3', 'cap4',
            'cap5', 'cup0', 'cup1', 'eraser0',
            'headset0', 'headset1', 'helmet0', 'helmet1',
            'helmet2', 'helmet3', 'jar0', 'microphone0',
            'shelf0', 'tap0', 'tap1', 'vase0',
            'vase1', 'vase2', 'vase3', 'vase4',
            'vase5', 'vase7', 'vase8', 'vase9'
        ]
    }

    def __init__(self, root_dir, split='train', transform=None, classes=None, dataset_name="Real3D"):
        """
        Args:
            root_dir (str): Dataset root containing one folder per class.
            split (str): 'train' or 'test'.
            transform (callable, optional): Point-cloud transform.
            classes (list, optional): Class names to load, e.g. ['airplane', 'car'].
            dataset_name (str): Dataset name, one of ['Real3D', 'AnomalyShapeNet'].
        """
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        assert dataset_name in self.PRESETS, f"dataset_name must be one of {list(self.PRESETS.keys())}"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.dataset_name = dataset_name

        all_classes = self.PRESETS[dataset_name]
        if classes is not None:
            self.classes = [cls for cls in classes if cls in all_classes]
        else:
            self.classes = all_classes

        self.CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(all_classes)}

        self.samples = []                   
        self.categories = []                 

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls, split)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.endswith(('.npy', '.npz')):
                    continue
                path = os.path.join(cls_dir, fname)
                self.samples.append(path)
                self.categories.append(self.CLASS_TO_IDX[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        category = self.categories[idx]

        loaded = np.load(path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if 'points' in loaded and 'labels' in loaded:
                points = loaded['points'].astype(np.float32)
                labels = loaded['labels'].astype(np.int64)
            elif 'points' in loaded and 'gt' in loaded:
                points = loaded['points'].astype(np.float32)
                labels = loaded['gt'].astype(np.int64)
            else:
                first_key = loaded.files[0]
                arr = loaded[first_key]
                points = arr[:, :3].astype(np.float32)
                labels = arr[:, -1].astype(np.int64)
        else:
            arr = loaded
            points = arr[:, :3].astype(np.float32)         
            labels = arr[:, -1].astype(np.int64)          

        points = torch.from_numpy(points)
        labels = torch.from_numpy(labels)

        if self.transform:
            points = self.transform(points)

        return {
            'points': points,          
            'labels': labels,         
            'category': category,
            'path': path
        }
