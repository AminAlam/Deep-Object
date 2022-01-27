import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import mat73
import numpy as np
import os
import random
import utils

import torchvision.transforms.functional as TF

class ODD_Dataset(Dataset):

    '''
    This class loads images, depths, and labels from .mat file.
    
    init inputs = path to .mat file, torch vision transfomr(optional).
    '''

    def __init__(self, root_folder, num_datas, transform=None):
        
        self.root_folder = root_folder
        self.transforms = transform
        self.num_datas = num_datas

        self.image_ids = ['imgNo{0}.png'.format(i) for i in range(self.num_datas)]
        self.label_ids = ['labelNo{0}.png'.format(i) for i in range(self.num_datas)]
        self.depth_ids = ['depthNo{0}.png'.format(i) for i in range(self.num_datas)]

    def __len__(self):
        return self.num_datas

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        label_id = self.label_ids[index]
        depth_id = self.depth_ids[index]

        images = Image.open(os.path.join(self.root_folder, img_id)).convert("RGB")
        mask = Image.open(os.path.join(self.root_folder, label_id)).convert("RGB")
        depths = Image.open(os.path.join(self.root_folder, depth_id)).convert("RGB")

        mask = np.array(mask)[:,:,1]
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        obj_ids = torch.as_tensor(obj_ids, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = obj_ids
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        if self.transforms is not None:
            images, target = self.transforms(images, target)

        return images, target



def get_loader(
    root_folder, transform=None, batch_size=32, num_datas=50, 
    num_workers=0, shuffle=True, pin_memory=True, train_test_ratio=1.0):
    '''
    This function returns train and test data loaders.
    inputs = path to .mat file, torch vision transfomr(optional), batch_size, num_workers.
    outputs = returns train dataloader, dataset train, test dataloader, dataset test
    '''
    dataset = ODD_Dataset(root_folder, num_datas=num_datas, transform=transform)

    dataset_train, dataset_test = random_split(dataset, 
                                  [int(len(dataset)*train_test_ratio), 
                                  len(dataset)-int(len(dataset)*train_test_ratio)],
                                  generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        dataset = dataset_train,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn=utils.collate_fn,
    )

    test_loader = DataLoader(
        dataset = dataset_test,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn=utils.collate_fn,
    ) 

    return train_loader , test_loader