#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:54:58 2020

@author: zhxy
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.functional as F
from torch.utils.data.dataloader import DataLoader
import scipy.io as sio
import os
import json
from PIL import Image
import attack_steps as atkstep
import SteerPyrSpace
import random
import spatial_original as spatial

device = torch.device('cuda')

class TestDataset(data.Dataset):
    def __init__(self, example_path, cross_i ):
        self.example_path = example_path
        
        self.annot_path = "../annotation/"
        self.crs = cross_i

    def __getitem__(self, index):
        img_index = index + (4 - self.crs) * 29
        
        img_file = self.example_path + "FCN_{}.mat".format(img_index)
        image = np.array(sio.loadmat(img_file)['xin'], dtype='float32').squeeze()
        #print(image.shape)
            
        annot = []
        for ii in range(0, 20):
            annot_index = img_index * 20 + ii
            annot_file = self.annot_path + "{}.mat".format(annot_index)
            
            dims = np.array(sio.loadmat(annot_file)['dims']).squeeze()
            areas = np.array(sio.loadmat(annot_file)['areas']).squeeze()
            rwt = np.array(sio.loadmat(annot_file)['rwt']).squeeze()
            annotation = np.concatenate((areas, dims, rwt), axis=0)
            annot.append(annotation)
            pix_spa = np.array(sio.loadmat(annot_file)['pix_spa'])[0]
        annot = np.array(annot)
        
        #print(image.shape)
        
        return image, annot, pix_spa

    def __len__(self):
        l = 29
        return l
    
class L2Pooling(nn.Module):
    def __init__(self):
        super(L2Pooling, self).__init__()
        pass
    def forward(self, x):
        x = torch.mul(x, x)
        x = (torch.sum(torch.sum(x, -1), -1) + 0.00000001) ** 0.5
        return x

class FCNnet(nn.Module):
    def __init__(self):
        super(FCNnet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3,3,3), 
                                             stride=1, padding=(1,0,0)),
                                   nn.BatchNorm3d(32), nn.LeakyReLU(inplace=True),
                                   nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1,3,3),
                                             stride=1, padding=0),
                                   nn.BatchNorm3d(64), nn.LeakyReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=(1,2,2)))
                                   
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), 
                                             stride=1, padding=(1,0,0)),
                                   nn.BatchNorm3d(128), nn.LeakyReLU(inplace=True),
                                   nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,3,3),
                                             stride=1, padding=0),
                                   nn.BatchNorm3d(128), nn.LeakyReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=(1,2,2)))
                                   
        self.conv3 = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3,3,3), 
                                             stride=1, padding=(1,0,0)),
                                   nn.BatchNorm3d(256), nn.LeakyReLU(inplace=True),
                                   nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1,3,3),
                                             stride=1, padding=(0,1,1)),
                                   nn.BatchNorm3d(256), nn.LeakyReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=(1,2,2)))
                                   
        self.conv4 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3,3,3), 
                                             stride=1, padding=(1,0,0)),
                                   nn.BatchNorm3d(512), nn.LeakyReLU(inplace=True),
                                   nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(1,3,3),
                                             stride=1, padding=0),
                                   nn.BatchNorm3d(512), nn.LeakyReLU(inplace=True))
                                   
        self.conv5 = nn.Sequential(nn.Conv3d(in_channels=512, out_channels=800, kernel_size=(1,3,3), 
                                             stride=1, padding=0),
                                   nn.BatchNorm3d(800), nn.LeakyReLU(inplace=True),
                                   
                                   nn.Conv3d(in_channels=800, out_channels=512, kernel_size=(3,1,1), 
                                             stride=1, padding=(1,0,0)),
                                   nn.BatchNorm3d(512), nn.LeakyReLU(inplace=True),
                                   
                                   nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3,1,1), 
                                             stride=1, padding=(1,0,0)),
                                   nn.BatchNorm3d(256), nn.LeakyReLU(inplace=True),
                                   
                                   nn.Conv3d(in_channels=256, out_channels=11, kernel_size=(3,1,1), 
                                             stride=1, padding=(1,0,0)))

    def forward(self, x):
        #print(x.shape)
        x = torch.reshape(x, (1, 1, 20, 80, 80))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        y = self.conv5(x)
        #print(y.shape)
        y = y.squeeze().permute(1, 0)
        
        return y

class Test_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, label):
        
        #print(out.shape)
        #print(label.shape)
        
        label = torch.reshape(label, (-1, 11))
        loss = torch.abs(label - out)
        return loss

## --------------------训练过程--------------------
def main(atk_type, iter_num, atk_range):
    
    all_loss = np.zeros((1, 11))
    for cross_i in [0]:
        example_path = "adv_example/{}_{}_{}/".format(iter_num, atk_type, atk_range)
        train_loader = DataLoader(dataset=TestDataset(example_path, cross_i), batch_size=1, shuffle=False)
    
        model = FCNnet()
        model = model.to(device)
    
        for name, param in model.named_parameters():
            param.requires_grad = False
            
        pretrained_path = "../params/{}_FCN-0080.pkl".format(cross_i)
        if os.path.exists(pretrained_path):
            model = torch.load(pretrained_path)
            #model.load_state_dict(pretrained_dict['model'])
        else:
            print('can not find model')
            break
            
        model.eval()
        lossFunc = Test_loss()
        for i, dataa in enumerate(train_loader):
            x, label, pix = dataa
            
            label = label.to(device)
            x = x.squeeze().unsqueeze(1).to(device)
            out = model(x)
                
            loss = lossFunc(out, label)
            #print(torch.mean(loss).squeeze().detach().cpu().numpy())

            loss = loss.squeeze().detach().cpu().numpy()
            pix_spa = pix.numpy().squeeze()
            loss[:, 0:2] = loss[:, 0:2] * pix_spa * pix_spa * 6400.0
            loss[:, 2:] = loss[:, 2:] * pix_spa * 80.0
            all_loss = np.insert(all_loss, 0, values=loss, axis=0)
        
    all_loss = np.delete(all_loss, -1, axis=0)

    m_loss = np.mean(all_loss, axis=0)
    s_loss = np.std(all_loss, axis=0)
    
    accu = []
    accu.append(np.mean(m_loss[0:2]))
    accu.append(np.mean(m_loss[2:5]))
    accu.append(np.mean(m_loss[5:]))
    print("{}_{}_{}: {}".format(atk_type, iter_num, atk_range, accu))
                

if __name__=="__main__":
    
    atk_types = ['SMIA']   # 'inf', 'unconstraint', 
    iter_nums = [50, 100]
    atk_ranges = [1, 2, 4, 8, 16, 24, 32, 48]
    
    for atk_type in atk_types:
        for iter_num in iter_nums:
            for atk_range in atk_ranges:
                main(atk_type, iter_num, atk_range)


