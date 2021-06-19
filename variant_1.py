# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:26:00 2021

@author: xiangyzhu6
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:44:27 2021

@author: zhxy
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import scipy.io as sio
import os
from multiprocessing import Process
from PIL import Image
from sqrtm import sqrtm
import SteerPyrSpace
import random
import torchvision.transforms as transforms
import MTlearn.tensor_op as tensor_op
import spatial_transform.spatial as spatial

device = torch.device("cuda")

def randomShift(image):
    img = np.array(image)
    img0 = np.array(image)
    pos1 = random.uniform(0, 16)
    if pos1 < 8:
        pos = random.uniform(0, 16)
        if pos < 8:    # up
            xs = random.randint(1, 10)
            img[0:xs, :], img[xs:80, :] = img0[80-xs:80, :], img0[0:80-xs, :]
        else:    # down
            xs = random.randint(70, 80)
            img[0:xs, :], img[xs:80, :] = img0[80-xs:80, :], img0[0:80-xs, :]
    
    pos1 = random.uniform(0, 16)
    if pos1 < 8:
        pos = random.uniform(0, 16)
        if pos < 8:   # right
            ys = random.randint(1, 15)
            img[:, 0:ys], img[:, ys:80] = img0[:, 80-ys:80], img0[:, 0:80-ys]
        else:    # left
            ys = random.randint(66, 80)
            img[:, 0:ys], img[:, ys:80] = img0[:, 80-ys:80], img0[:, 0:80-ys]
    img1 = Image.fromarray(np.array(img, dtype='uint8'))
    return img1
def randomCrop(image):
    pos = random.uniform(0, 16)
    if pos < 6:
        RandomCrop = transforms.RandomCrop(size=(76, 76), padding=0)
        random_image = RandomCrop(image)
        random_image = transforms.Pad(padding=2)(random_image)
    elif pos > 6 and pos < 10:
        RandomCrop = transforms.RandomCrop(size=(72, 72), padding=0)
        random_image = RandomCrop(image)
        random_image = transforms.Pad(padding=4)(random_image)
    else:
        RandomCrop = transforms.RandomCrop(size=(78, 78), padding=0)
        random_image = RandomCrop(image)
        random_image = transforms.Pad(padding=1)(random_image)
    return random_image
def randomRotation(image):
    RR = transforms.RandomRotation(degrees=(-30, 30))
    rr_image = RR(image)
    return rr_image
def randomColor(image):
    image1 = image.convert("RGB")
    RC = transforms.ColorJitter(brightness=0.8, contrast=0.8)
    rc_image1 = RC(image1)
    rc_image = rc_image1.convert('L')
    return rc_image

def DataAugmentation_per_chnl(image):
    im1 = image[:, :]
    x1min, x1max = im1.min(), im1.max()
    x1np = (im1 - x1min) / (x1max - x1min) * 255.0
    img1 = Image.fromarray(np.array(x1np, dtype='uint8'))
    if img1.size[0] < 60:
        img1 = img1.resize((80, 80), resample=0)
    possib = random.uniform(0, 16)
    if possib < 6:
        img1 = randomColor(img1)
    possib = random.uniform(0, 16)
    if possib < 5:
        img1 = randomCrop(img1)
    possib = random.uniform(0, 16)
    if possib < 6:
        img1 = randomRotation(img1)
    if possib < 7:
        img1 = randomShift(img1)  
    im1 = np.array([np.array(img1)], dtype="float32") / 255.0
    return im1

def DataAugmentation(image):
    img = np.zeros((image.shape[0], 80, 80))
    if image.shape[0]>10:
        img[:, :] = DataAugmentation_per_chnl(image[:, :]) 
    else:
        for i in range(image.shape[0]):
            img[i, :, :] = DataAugmentation_per_chnl(image[i, :, :])     
    return img

class TrainDataset(data.Dataset):
    def __init__(self, cross_i ):
        self.annot_path = "annotation/"
        self.crs = cross_i

    def __getitem__(self, index):
        
        if index >= (4 - self.crs) * 580:
            index = index + 580
            
        annot_file = self.annot_path + "{}.mat".format(index)
        
        image = np.array(sio.loadmat(annot_file)['image_LV'], dtype='float32').squeeze()/255
        dims = np.array(sio.loadmat(annot_file)['dims']).squeeze()
        areas = np.array(sio.loadmat(annot_file)['areas']).squeeze()
        rwt = np.array(sio.loadmat(annot_file)['rwt']).squeeze()
        pix = np.array(sio.loadmat(annot_file)['pix_spa'])[0]
        annotation = np.concatenate((areas, dims, rwt), axis=0)

        return image, annotation

    def __len__(self):
        l = 2020
        return l
    
class L2Pooling(nn.Module):
    def __init__(self):
        super(L2Pooling, self).__init__()
        pass
    def forward(self, x):
        x = torch.mul(x, x)
        x = (torch.sum(torch.sum(x, -1), -1) + 1e-8) ** 0.5
        return x
    
class DCAENet(nn.Module):
    def __init__(self):
        super(DCAENet, self).__init__()
        
        self.bn1 = nn.Sequential(nn.BatchNorm2d(3),
                                nn.LeakyReLU(inplace=True))
        
        self.bn2 = nn.Sequential(nn.BatchNorm2d(3),
                                nn.LeakyReLU(inplace=True))
        
        self.bn3 = nn.Sequential(nn.BatchNorm2d(3),
                                nn.LeakyReLU(inplace=True))

        self.convd1 = nn.Sequential(nn.Conv2d(3, 60, kernel_size=7, stride=1, padding=3, dilation=1),
                                    nn.BatchNorm2d(60),
                                   nn.LeakyReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, padding=0),  # 40x40x16
                                   
                                   nn.Conv2d(60, 120, kernel_size=5, stride=1, padding=2, dilation=1),
                                   nn.BatchNorm2d(120),
                                   nn.LeakyReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, padding=0), #20*20
                                   
                                   nn.Conv2d(120, 240, kernel_size=5, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(240),
                                   nn.LeakyReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, padding=0)) #10x10
    
        self.conv_share_last = nn.Sequential(nn.Conv2d(240, 480, kernel_size=5, stride=1, padding=0, dilation=1),
                                             nn.BatchNorm2d(480),
                                             nn.LeakyReLU(inplace=True),   # 6x6x64
                                             
                                             L2Pooling())  
        
        self.fc_share = nn.Sequential(nn.Linear(480, 100),
                                   nn.Dropout(0.3),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Linear(100, 100),
                                   nn.Dropout(0.1),
                                   nn.LeakyReLU(inplace=True))
        
        self.fcd1_last = nn.Linear(100, 11)
        self.fcd2_last = nn.Linear(100, 11)
        self.fcd3_last = nn.Linear(100, 11)
        
        self.noise = nn.LeakyReLU(inplace=True)
                
    def forward(self, x):
        
        x = torch.reshape(x, (-1, 1, 80, 80))
        d1, d2, d3 = SteerPyrSpace.getST_v44(x)
        
        d1 = self.noise_f(d1)
        d2 = self.noise_f(d2)
        d3 = self.noise_f(d3)
        
        d1 = self.bn1(d1)
        d2 = self.bn2(d2)
        d3 = self.bn3(d3)
        
        d1 = self.convd1(d1)
        d2 = self.convd1(d2)
        d3 = self.convd1(d3)
        
        d1 = self.conv_share_last(d1).view(-1, 480)
        d2 = self.conv_share_last(d2).view(-1, 480)
        d3 = self.conv_share_last(d3).view(-1, 480)
        
        d1 = self.fc_share(d1)
        d2 = self.fc_share(d2)
        d3 = self.fc_share(d3)
        
        d1_last_in = torch.reshape(d1, (-1, 100))
        d2_last_in = torch.reshape(d2, (-1, 100))
        d3_last_in = torch.reshape(d3, (-1, 100))
        
        outd1 = self.fcd1_last(d1_last_in)
        outd2 = self.fcd2_last(d2_last_in)
        outd3 = self.fcd3_last(d3_last_in)
        
        return outd1, outd2, outd3
    
    def noise_f(self, x, scale=0.07):
        pos = random.uniform(0, 16)
        if pos < 8:
            eps = 10e-5
            x *= 0.35
            x += 0.07
            x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=scale).rsample() * torch.sqrt(F.relu(x.clone()) + eps)
            x -= 0.07
            x /= 0.35      
        return self.noise(x)
    
class Train_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outd1, outd2, outd3, label):
        
        label = torch.reshape(label, (-1, 11))

        d1loss = torch.mean(torch.abs(label - outd1))
        d2loss = torch.mean(torch.abs(label - outd2))
        d3loss = torch.mean(torch.abs(label - outd3))
        area = (outd1+outd2+outd3)/3
        varloss=torch.mean(torch.abs(outd1-area)+torch.abs(outd2-area)+torch.abs(outd3-area))/3
        loss = d1loss + d2loss + d3loss #+ varloss*0.2 #+ 1*mtloss

        return d1loss, d2loss, d3loss, loss # + d2loss + d2loss + 1*mtloss #+ varloss*0.10 #+ dstloss*107
    
## --------------------训练过程--------------------
def train(crossi, batch_size, epoch, learning_rate, device = torch.device("cuda")):
    cross_i = crossi
    lr = learning_rate
    train_loader = DataLoader(dataset=TrainDataset(cross_i), batch_size=batch_size, shuffle=True)

    model = DCAENet()
    model = model.to(device)

    for name, param in model.named_parameters():
        if 'conv' in name:
            param.requires_grad = True
        else:
            param.requires_grad = True
            
    pretrained_path = "params/{}_v44-0155d.pkl".format(cross_i)
    if os.path.exists(pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        model.load_state_dict(pretrained_dict['model'])
        model.task_cov_var.data = torch.tensor(pretrained_dict['task']).to(device)
        model.class_cov_var.data = torch.tensor(pretrained_dict['class']).to(device)
        model.feature_cov_var.data = torch.tensor(pretrained_dict['feature']).to(device)
    else:
        for layer in model.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight.data)
     
    model.train()
    lossFunc = Train_loss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    for e in range(0, epoch):
        print("epoch: {}".format(epoch))
        if e > 2 and (e % 170 == 0 or e % 290 == 0 or e % 360 == 0 or e % 4350 == 0):
            lr = lr * 0.12
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        for i, dataa in enumerate(train_loader):
            optimizer.zero_grad()
            img, label = dataa
            
            imgin = Variable(torch.Tensor(img), requires_grad=True).to(device)
            outd1, outd2, outd3 = model(imgin)
            #print(out1.size())
            label = label.to(device)
            d1loss, d2loss, d3loss, loss = lossFunc(outd1, outd2, outd3, label)
            print("epoch: " + str(e) + "   pureloss: " + str(loss.item()) )
            loss.backward()
            #d1loss.backward(retain_graph=True)
            #d2loss.backward(retain_graph=True)
            #d3loss.backward()
            #print(grad_loss)
            #if e > 0 :
            optimizer.step()
            
        # if e > 0 and e%3==0:
        #     model.update_cov()
            
        print("epoch" + str(e))
        if e % 5 == 0 and e > 80:
            dd = {'model':model.state_dict()}
            file_path = os.path.join("params/", str(cross_i) + '_v44-{:04d}.pkl'.format(e))
            torch.save(dd, file_path)
            #torch.save(state, file_path)

if __name__ == "__main__":
    train(0, 50, 350, 0.0005, torch.device("cuda"))
    #train(4, 3, 1400, 0.0008, torch.device("cuda"))
#    aaa = 0
#    try:
#        for aa in range(aaa, aaa+2):
#            P = Process(target=train, args=(aa, 1, 1, 0.00008, torch.device("cuda:0")))
#            print(aa)
#            P.start()
#    except:
#        print("Multi process error!")
               
