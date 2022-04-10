import copy
import os, glob

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations
from albumentations.core.composition import Compose


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_dir, classes, transform, device):
        super().__init__()
        self.device = device
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.classes = classes
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = img_path.split("/")[-1]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = []
        for class_name in self.classes:
            mask_path = os.path.join(self.mask_dir, class_name, img_name)
            if not os.path.exists(mask_path):
                mask.append(np.zeros((img.shape[0], img.shape[1])))
            else:
                mask.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        # TODO mask>0 SOR
        mask[mask<50] = 0
        mask[mask>=50] = 1
        img = img.astype('float32') / 255.
        img = torch.from_numpy(img.transpose(2, 0, 1)).to(self.device)
        mask = torch.from_numpy(mask.transpose(2, 0, 1).astype("float32")).to(self.device)
        return {'input': {"path":img_path, "image":img}, # goes to model
                'target': {"path":mask_path, "mask":mask}} # goes to loss
    
    
def get_dataloader(config, device, overfit_mode):
    # todo random crop other aug?
    train_transform = Compose([
        albumentations.augmentations.geometric.rotate.RandomRotate90(),
        albumentations.augmentations.Flip(),
        albumentations.augmentations.RandomBrightnessContrast(),
        #albumentations.augmentations.RandomCrop(config['input_h'], config['input_w']),
        albumentations.Normalize()
    ])
    val_transform = Compose([
        #albumentations.augmentations.RandomCrop(config['input_h'], config['input_w']),
        albumentations.Normalize()
    ])    
    train_paths = glob.glob(os.path.join(config["train_root"], "images/*"))
    val_paths = glob.glob(os.path.join(config["val_root"], "images/*"))
    if overfit_mode:
        train_paths = train_paths[:1]
    train_mask_dir = os.path.join(config["train_root"], "masks")
    val_mask_dir = os.path.join(config["val_root"], "masks")
    classes = sorted(glob.glob(os.path.join("/".join(train_paths[0].split("/")[:-2]), "masks/*")))
    classes = [c.split("/")[-1] for c in classes]
    print("\nCLASSES")
    for i,c in enumerate(classes):
        print(i,c)
    print()
    if overfit_mode:
        val_paths = train_paths
        val_mask_dir = train_mask_dir
        train_transform = val_transform
    train_data = SegmentationDataset(train_paths, train_mask_dir, classes, train_transform, device)
    val_data = SegmentationDataset(val_paths, val_mask_dir, classes, val_transform, device)
    train_loader = DataLoader(train_data, config["batch_size"], num_workers=config["num_workers"],
                              shuffle=True)
    val_loader = DataLoader(val_data, config["batch_size"], num_workers=config["num_workers"],
                            shuffle=False)
        
    return train_loader, val_loader
