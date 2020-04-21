# Dataset class would split the original data into train, test, validation splits.
# The original data is the result of the preprocessing step which can be stored in a text, csv, hdf5, etc file

import os
import h5py
import numpy as np
from torch.utils.data import Dataset


class BassetDataset(Dataset):

    # Initializes the BassetDataset
    def __init__(self, path, f5name, split, transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """
        
        self.split = split
        
        split_dict = {'train': ['train_in', 'train_out'], 
                      'test': ['test_in', 'test_out'], 
                      'valid': ['valid_in', 'valid_out']}
        
        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"
        
        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')
        
        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']
        
        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]
        if self.split!='test':
            self.ids = list(range(len(self.inputs)))
        else:
            self.ids = np.char.decode(self.dataset['test_headers'])
            
    def __getitem__(self, i):
        
        id = self.ids[i]

        # Sequence & Target
        sequence, target = self.inputs[id], self.outputs[id]

        return sequence, target

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data', help="Path to the dataset directory (default: './data'.")
    parser.add_argument('--file_name', type=str, default='sample_dataset.h5', help='Name of the h5 file already preprocessed in the preprocessing step (default: sample_dataset.h5).')
    parser.add_argument('--split', type=str, default='train', help='Defines what data split to work with (default: train).')
    args = parser.parse_args()
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Arguments are:\n{}".format(vars(args))) # better to use '--help' instead (try now!) to get the list of args; please remove after understanding
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
    basset_dataset = BassetDataset(args.path, args.file_name, args.split)
    print("The number of samples in the {} split of the input file '{}' is {}.\n".format(args.split, args.file_name, len(basset_dataset)))
