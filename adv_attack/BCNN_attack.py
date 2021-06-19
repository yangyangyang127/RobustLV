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
from multiprocessing import Process
from PIL import Image
import attack_steps as atkstep
import SteerPyrSpace
import random
import spatial_original as spatial
import SMIA
from sqrtm import sqrtm

device = torch.device("cuda")

class TrainDataset(data.Dataset):
    def __init__(self, cross_i ):
        self.annot_path = "../annotation/"
        self.crs = cross_i

    def __getitem__(self, index):
        
        index = index + (4 - self.crs) * 29
        
        img = []
        annot = []
        pixes = []
        for ii in range(0, 20):
            index1 = index * 20 + ii
            annot_file = self.annot_path + "{}.mat".format(index1)
            image = np.array(sio.loadmat(annot_file)['image_LV'], dtype='float32').squeeze()/255
            dims = np.array(sio.loadmat(annot_file)['dims']).squeeze()
            areas = np.array(sio.loadmat(annot_file)['areas']).squeeze()
            rwt = np.array(sio.loadmat(annot_file)['rwt']).squeeze()
            pix = np.array(sio.loadmat(annot_file)['pix_spa'])[0]
            annotation = np.concatenate((areas, dims, rwt), axis=0)

            img.append(image)
            annot.append(annotation)
            pixes.append(pix)
            
        img = np.array(img)
        annot = np.array(annot)
        pixes = np.array(pixes)

        return img, annot

    def __len__(self):
        l = 29
        return l
    
class BCNN(nn.Module):
    def __init__(self):
        super(BCNN, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.01),
                                   nn.MaxPool2d(kernel_size=2, padding=0))  # 40x40x16
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.01),
                                   nn.MaxPool2d(kernel_size=2, padding=0))  # 20x20x16
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.01),
                                   nn.MaxPool2d(kernel_size=2, padding=0))  # 10x10x32
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.01),
                                   nn.MaxPool2d(kernel_size=2, padding=0))  # 5x5x32
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.01))  # 1x1x64
        self.fc1 = nn.Sequential(nn.Linear(128, 11))  # 输出1x11
        self.fc2 = nn.Sequential(nn.Linear(128, 11))  # 输出1x11

    def forward(self, x):
        
        x = torch.reshape(x, (-1, 1, 80, 80))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #print(x.size())
        #print(x)
        x = x.view(-1, 128)
        y = self.fc1(x)

        return y

class Train_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, label):
        
        label = label.view(-1, 11)
        
        loss = torch.mean(torch.abs(label - out))
        return loss
    
## -------------------- 训练过程 --------------------
def main(atk_type, iter_num, atk_range):

    for cross_i in [0]:
        lr = 0.001
        train_loader = DataLoader(dataset=TrainDataset(cross_i), batch_size=1, shuffle=False)
    
        model = BCNN()
        #model = model.to(device)
    
        for name, param in model.named_parameters():
            param.requires_grad = True
            
        pretrained_path = "../params/{}_BCNN-0080.pkl".format(cross_i)
        if os.path.exists(pretrained_path):
            model = torch.load(pretrained_path)
            model = model.to(device)
        else:
            print('can not find model')
            break
        
        model.train()
        lossTrain = Train_loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        for i, dataa in enumerate(train_loader):
            index = i + (4 - cross_i) * 29
            print("i-th: {}".format(i))
            
            dst_path = "adv_example/{}_{}_{}/".format(iter_num, atk_type, atk_range)
            out_file = dst_path + "BCNN_{}.mat".format(index)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            
            if not os.path.exists(out_file):
            
                x, label, = dataa
                label = label.to(device)
                x = x.unsqueeze(1)
                orig_xin = Variable(torch.Tensor(x)).to(device)
                
                if atk_type =='l2':
                    step = atkstep.L2Step(orig_xin, atk_range/255.0, atk_range/255.0)
                    noise = step.random_perturb(orig_xin)
                    x1 = orig_xin + noise
                elif atk_type =='inf':
                    step = atkstep.LinfStep(orig_xin, atk_range/255.0, atk_range/255.0*0.01)
                    noise = step.random_perturb(orig_xin)
                    x1 = orig_xin + noise
                elif atk_type =='unconstraint':
                    step = atkstep.UnconstrainedStep(orig_xin, atk_range/255.0, atk_range/255.0)
                    noise = step.random_perturb(orig_xin)
                    x1 = orig_xin + noise
                elif atk_type == 'SMIA':
                    step = SMIA.SMIA(model, atk_range/255.0, atk_range/255*0.01, lossTrain)
                    x1 = orig_xin
                    
                xin = Variable(torch.Tensor(x1.cpu()), requires_grad=True).to(device)
                for e in range(0, iter_num):
                    if atk_type == 'SMIA':
                        xin = step.perturb(xin, label, a1=1, a2=0.2, niters=iter_num)
                        break
                    else:
                        optimizer.zero_grad()
                        
                        output = torch.zeros((20, 11)).to(device)
                        for ii in range(0, 20):
                            out = model(xin)
                            output = output + out.squeeze()
                        output = output / 20
                        loss = lossTrain(out, label)
        
                        xin.retain_grad()
                        loss.backward()
                        
                        print("epoch: " + str(e) + "   pureloss: " + str(loss.item()))
                        
                        g = xin.grad.data
                        noise = step.step(noise, g)
                        #print(noise)
                        noise = torch.clamp(noise, -atk_range/255.0, atk_range/255.0)
        
                        xin1 = orig_xin + noise
                        xin1 = step.project(xin1)
                        xin.data = xin1
                    
                dd = {'xin':xin.squeeze().detach().cpu().numpy()}
                sio.savemat(out_file, dd)

if __name__=="__main__":
    
    atk_types = ['SMIA'] #['SMIA', 'inf']   #'inf', 
    iter_nums = [50, 100]
    atk_ranges = [1, 2, 4, 8, 16, 24, 32, 48]
    
    for iter_num in iter_nums:
        for atk_type in atk_types:
            for atk_range in atk_ranges:
                main(atk_type, iter_num, atk_range)
                
#    for atk_type in atk_types:
#        for iter_num in iter_nums:
#            for atk_range in atk_ranges:
#                main(atk_type, iter_num, atk_range)
                
#    try:
#        for aa in range(0, 1):
#            P = Process(target = main, args=(aa, 1, 0.0003, 100,))
#            print(aa)
#            P.start()
#            #_thread.start_new_thread(main, (aa*200, 1))
#    except:
#        print("Thread wrong!")

