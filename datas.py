import torch
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import mat73
import os


class ODD_Dataset(Dataset):

    '''
    This class loads images, depths, and labels from .mat file.
    
    init inputs = path to .mat file, torch vision transfomr(optional).
    '''

    def __init__(self, root_folder, num_datas, transform=None):
        
        self.root_folder = root_folder
        self.transform = transform
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
        labels = Image.open(os.path.join(self.root_folder, label_id)).convert("RGB")
        depths = Image.open(os.path.join(self.root_folder, depth_id)).convert("RGB")

        if self.transform is not None:
            images = self.transform(images)
            depths = self.transform(depths)
            labels = self.transform(labels)

        return images, depths, labels



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
    )

    test_loader = DataLoader(
        dataset = dataset_test,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
    ) 

    return train_loader, dataset_train, test_loader, dataset_test