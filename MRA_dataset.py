from torch.utils.data import Dataset
import pandas as pd
import torch
import h5py
import itk
import matplotlib.pyplot as plt

class MRA_Dataset(Dataset):
    def __init__(self, h5_list_path, transform=None):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.h5_list = pd.read_csv(h5_list_path, header=None)
        self.transform = transform

    def __len__(self):
        return self.h5_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        i_h5 = self.h5_list.iloc[idx][0]

        with h5py.File(i_h5.rstrip(), 'r' ) as hf:
            image = hf['image'][:]
            label = hf['label'][:]
            
        ### show figure by ITK
#        image_view = itk.GetImageViewFromArray(image[0, :, :, 16])
#        plt.imshow(image_view)
        ######################
            
        sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

#if __name__ == '__main__':
#    dataset = CT_Dataset('./train_list.txt')
#    print(dataset.__getitem__(0)['image'].shape)
#    print(dataset.__getitem__(0)['label'].shape)
#    
#    dataset = CT_Dataset('./val_list.txt')
#    print(dataset.__getitem__(0)['image'].shape)
#    print(dataset.__getitem__(0)['label'].shape)