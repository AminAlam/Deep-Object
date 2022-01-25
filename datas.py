import torch
from torch.utils.data import DataLoader, Dataset, random_split
import mat73


class ODD_Dataset(Dataset):

    '''
    This class loads images, depths, and labels from .mat file.
    
    init inputs = path to .mat file, torch vision transfomr(optional).
    '''

    def __init__(self, path2dataset, transform=None):
        
        self.path2dataset = path2dataset
        self.transform = transform
        self.load_mat()


    def load_mat(self):
        datas = mat73.loadmat(self.path2dataset)
        self.images = datas['images']
        self.depths = datas['depths']
        self.labels = datas['labels']
        del datas


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img = self.images[:,:,:index]
        depth = self.depths[:,:,index]
        label = self.labels[:,:,index]

        if self.transform is not None:
            img = self.transform(img)

        return img, depth, label

def get_loader(
    path2dataset, transform=None, batch_size=32, 
    num_workers=0, shuffle=True, pin_memory=True, train_test_ratio=0.0):
    '''
    This function returns train and test data loaders.
    inputs = path to .mat file, torch vision transfomr(optional), batch_size, num_workers.
    outputs = returns train dataloader, dataset train, test dataloader, dataset test
    '''
    dataset = ODD_Dataset(path2dataset)

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