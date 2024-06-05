from torch.utils.data import Dataset

import os

import numpy as np

class BouncingBallDataLoader(Dataset):

    def __init__(self, root_dir):
        if isinstance(root_dir,str):
            self.root_dir = root_dir
            self.file_list = sorted(os.listdir(root_dir))
            self.multiple = False
            self.filenames = None
        else:
            self.root_dir = root_dir
            self.file_list = []
            for dir in root_dir:
                for filename in sorted(os.listdir(dir)):
                    self.file_list.append(os.path.join(dir,filename))
            self.multiple = True
            



    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        sample = np.load(os.path.join(
            self.root_dir, self.file_list[i])) if not self.multiple else np.load(self.file_list[i])
        im = sample['arr_0']
        if len(im.shape) == 3:
            im = im[:,np.newaxis,:,:]
        else:
            im = im.astype(float)
            if im.max() > 2.0:
                im /= 255.0
            im = im.transpose((0,3,1,2)) - 0.5
        return (im,)