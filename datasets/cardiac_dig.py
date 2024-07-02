import torch
import random
import numpy as np
import scipy.io as sio
import torch.utils.data as data

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from datasets.data_augment import *

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

train_range = {0: [*range(0,101)], 1: [*range(0,72)]+[*range(116,145)], 2: [*range(0,28)]+[*range(43,58)]+[*range(87,145)],
               3: [*range(15,29)]+[*range(58,145)], 4: [*range(29,130)]}
valid_range = {0: [*range(101,116)], 1: [*range(72,87)], 2: [*range(28,43)], 3: [*range(0,15)], 4: [*range(130,145)]}
test_range = {0: [*range(116,145)], 1: [*range(87,116)], 2: [*range(58,87)], 3: [*range(29,58)], 4: [*range(0,29)]}

class CardiacDigDataset(data.Dataset):
    def __init__(self, dataset_path='cardiac-dig.mat', state='train', cross_valid_fold=0):

        assert (state in ['train', 'valid', 'test'])
        assert (cross_valid_fold in [0, 1, 2, 3, 4])
        self.state = state
        self.cvf = cross_valid_fold

        cardiac = sio.loadmat(dataset_path)
        self.images = cardiac['images_LV']
        self.areas = cardiac['areas']
        self.dims = cardiac['dims']
        self.rwts = cardiac['rwt']
        self.pix_spa = np.array(cardiac['pix_spa'])
        self.ratio = np.array(cardiac['ratio_resize_inverse'])
        epi  = np.array(cardiac['epi_LV'], dtype='float32')
        endo = np.array(cardiac['endo_LV'], dtype='float32')
        self.myo = epi - endo

    def __getitem__(self, ind):

        if self.state=='train':
            index = train_range[self.cvf][ind]
        elif self.state=='valid':
            index = valid_range[self.cvf][ind]
        elif self.state=='test':
            index = test_range[self.cvf][ind]

        images = np.array(self.images[:, :, index*20:index*20+20], dtype='float32').squeeze()
        dims = np.array(self.dims[:, index*20:index*20+20]).squeeze()
        areas = np.array(self.areas[:, index*20:index*20+20]).squeeze()
        rwts = np.array(self.rwts[:, index*20:index*20+20]).squeeze()
        annot = np.concatenate((areas, dims, rwts), axis=0)

        d1annot = np.array([annot[3,:], annot[5,:], annot[8,:], annot[0,:], annot[1,:]])
        d2annot = np.array([annot[4,:], annot[6,:], annot[9,:], annot[0,:], annot[1,:]])
        d3annot = np.array([annot[2,:], annot[7,:], annot[10,:], annot[0,:], annot[1,:]])

        images = images.transpose(2,0,1)
        annot = annot.transpose(1,0)
        d1annot, d2annot, d3annot = d1annot.transpose(1,0), d2annot.transpose(1,0), d3annot.transpose(1,0)
        
        if self.state=='train':
            images = DataAugmentation(images)
        
        pix = self.pix_spa[index, :] * self.ratio[:, index]
        return images, annot, d1annot, d2annot, d3annot, pix

    def __len__(self):
        if self.state=='train':
            return 101
        elif self.state=='valid':
            return 15
        elif self.state=='test':
            return 29

class CardiacDigProvider():

    def __init__(self, bs=4, cross_valid_fold=0):
        
        dataset_path='data/cardiac-dig.mat'
        self.train = torch.utils.data.DataLoader(CardiacDigDataset(dataset_path, 'train', cross_valid_fold), 
                                                  batch_size=bs, shuffle=True, 
                                                  num_workers=0,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)
        
        self.valid = torch.utils.data.DataLoader(CardiacDigDataset(dataset_path, 'valid', cross_valid_fold), 
                                                  batch_size=bs, shuffle=False, 
                                                  num_workers=0,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)
        
        self.test = torch.utils.data.DataLoader(CardiacDigDataset(dataset_path, 'test', cross_valid_fold), 
                                                  batch_size=bs, shuffle=False, 
                                                  num_workers=0,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)

