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
from sqrtm import sqrtm

device = torch.device('cuda')

class TestDataset(data.Dataset):
    def __init__(self, example_path, cross_i ):
        self.example_path = example_path
        
        self.annot_path = "../annotation/"
        self.crs = cross_i

    def __getitem__(self, index):
        img_index = index + (4 - self.crs) * 29
        
        img_file = self.example_path + "HQ_{}.mat".format(img_index)
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
            pix_spa = np.array(sio.loadmat(annot_file)['pix_spa'])[0] #* np.array(sio.loadmat(annot_file)['ratio'])[0]
        annot = np.array(annot)
        return image, annot, pix_spa

    def __len__(self):
        l = 29
        return l

class HCNN(nn.Module):
    def __init__(self, ks, ps):
        super(HCNN, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=20, 
                                        kernel_size=ks, stride=1, padding=ps),
                                        nn.SELU(inplace=True),
                                        nn.BatchNorm3d(20)) 
        
        self.brch1_conv1 = nn.Sequential(nn.MaxPool3d(kernel_size=(1,2,2)),
                                        nn.Conv3d(in_channels=20, out_channels=40, 
                                        kernel_size=ks, stride=1, padding=ps),
                                        nn.SELU(inplace=True),
                                        nn.BatchNorm3d(40),
                                        nn.MaxPool3d(kernel_size=(1,4,4))) #10*10*20*40
        self.brch1_conv2 = nn.Sequential(nn.Conv3d(in_channels=40, out_channels=40, 
                                        kernel_size=ks, stride=1, padding=ps),
                                        nn.SELU(inplace=True),
                                        nn.BatchNorm3d(40),
                                        nn.MaxPool3d(kernel_size=(1,2,2))) #5*5*20*40
        
        self.brch2_conv1 = nn.Sequential(nn.MaxPool3d(kernel_size=(1,4,4)),
                                        nn.Conv3d(in_channels=20, out_channels=40, 
                                        kernel_size=ks, stride=1, padding=ps),
                                        nn.SELU(inplace=True),
                                        nn.BatchNorm3d(40),
                                        nn.MaxPool3d(kernel_size=(1,2,2))) #10*10*20*40
        self.brch2_conv2 = nn.Sequential(nn.Conv3d(in_channels=40, out_channels=40, 
                                        kernel_size=ks, stride=1, padding=ps),
                                        nn.SELU(inplace=True),
                                        nn.BatchNorm3d(40),
                                        nn.MaxPool3d(kernel_size=(1,2,2))) #5*5*20*40
        
        self.brch3_conv1 = nn.Sequential(nn.MaxPool3d(kernel_size=(1,8,8)),
                                        nn.Conv3d(in_channels=20, out_channels=40, 
                                        kernel_size=ks, stride=1, padding=ps),
                                        nn.SELU(inplace=True),
                                        nn.BatchNorm3d(40),
                                        nn.MaxPool3d(kernel_size=(1,2,2))) #5*5*20*40
                                        
        self.brch4_conv1 = nn.Sequential(nn.MaxPool3d(kernel_size=(1,16,16)),
                                        nn.Conv3d(in_channels=20, out_channels=40, 
                                        kernel_size=ks, stride=1, padding=ps),
                                        nn.SELU(inplace=True),
                                        nn.BatchNorm3d(40)) #5*5*20*40     
                                         
    def forward(self, x):
        x = self.conv1(x)
        
        x1 = self.brch1_conv1(x)
        x1 = self.brch1_conv2(x1)
                
        x2 = self.brch2_conv1(x)
        x2 = self.brch2_conv2(x2)
                
        x3 = self.brch3_conv1(x)
        
        x4 = self.brch4_conv1(x)
        
        x5 = torch.cat((x1, x2, x3, x4), 1)
        return x5
        
class Relationship_Matrix(nn.Module):
    def __init__(self):
        super(Relationship_Matrix, self).__init__()
        self.W = nn.Parameter(torch.randn(100, 11))
        self.b = nn.Parameter(torch.randn(1, 11))
        torch.nn.init.xavier_normal_(self.W)
        
    def forward(self, x):
        WT = torch.transpose(self.W, 1, 0)
        #print(WT.size())
        y = torch.matmul(x, self.W)
        bb = self.b.expand(list(y.size()))
        y = y + bb
        
        WTW_sqrt = sqrtm(torch.mm(WT, self.W))
        omiga = WTW_sqrt / torch.trace(WTW_sqrt)
#        print(self.W.size())
#        print(torch.inverse(omiga).size())
        
        loss1 = torch.trace(torch.mm(torch.mm(self.W, torch.inverse(omiga)), WT))
        
        out = y + loss1 * 0.01
        return out

class DCAENet(nn.Module):
    def __init__(self):
        super(DCAENet, self).__init__()
        self.hcnn1 = HCNN((3,5,5), (1,2,2))
        self.hcnn3 = HCNN((3,3,3), (1,1,1))
        
        self.cat_map = nn.Sequential(nn.Conv3d(in_channels=320, out_channels=40,
                                    kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
                                    nn.SELU(inplace=True), nn.BatchNorm3d(40),
                                    nn.Conv3d(in_channels=40, out_channels=200,
                                    kernel_size=(3, 5, 5), stride=1, padding=(1, 0, 0)),
                                    nn.SELU(inplace=True), nn.BatchNorm3d(200))
                                    
        self.lstm = nn.LSTM(200, 100, 4)
        
        self.relationship = Relationship_Matrix()

    def forward(self, x):
        
        #print(x.shape)
        x = torch.reshape(x, (1, 1, 20, 80, 80))
        x1 = self.hcnn1(x)
        x3 = self.hcnn3(x)
        x4 = torch.cat((x1, x3), 1)
        
        x5 = self.cat_map(x4)
        
        x5 = x5.view(-1, 200, 20)
        x5 = x5.permute(2, 0, 1)
        
        x6, (h_) = self.lstm(x5)
        
        x6 = x6.permute(1, 0, 2)
        
        x7 = self.relationship(x6)
        
        return x7

class Test_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, label):
        
        #print(out.shape)
        #print(label.shape)
        
        #label = torch.reshape(label, (-1, 11))
        loss = torch.abs(label - out)
        return loss

## --------------------训练过程--------------------
def main(atk_type, iter_num, atk_range):
    
    all_loss = np.zeros((1, 11))
    for cross_i in [0]:
        example_path = "adv_example/{}_{}_{}/".format(iter_num, atk_type, atk_range)
        train_loader = DataLoader(dataset=TestDataset(example_path, cross_i), batch_size=1, shuffle=False)
    
        model = DCAENet()
        model = model.to(device)
    
        for name, param in model.named_parameters():
            param.requires_grad = False
            
        pretrained_path = "../params/{}_HQnet-0105.pkl".format(cross_i)
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
    
    atk_types = ['SMIA']   #, 'SMIA'
    iter_nums = [50, 100]
    atk_ranges = [1, 2, 4, 8, 16, 24, 32, 48]
    
    for atk_type in atk_types:
        for iter_num in iter_nums:
            for atk_range in atk_ranges:
                main(atk_type, iter_num, atk_range)


