import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np

def default_loader(root, path_imgs, path_map):
    imgs = [os.path.join(root, path) for path in path_imgs]
    map = os.path.join(root,path_map)
    return [imread(img).astype(np.float32) for img in imgs], imread(map)


class ListDataset(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, target = self.path_list[index]

        inputs, target = self.loader(self.root, inputs, target)

        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
            inputs[2] = self.transform(inputs[2])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)

