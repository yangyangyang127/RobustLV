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
        
        print(x.shape)

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

class Train_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, label):
        
        loss = torch.mean(torch.abs(label - out))
        return loss
    
## -------------------- 训练过程 --------------------
def main(atk_type, iter_num, atk_range):

    for cross_i in [0]:
        lr = 0.001
        train_loader = DataLoader(dataset=TrainDataset(cross_i), batch_size=1, shuffle=False)
    
        model = DCAENet()
        #model = model.to(device)
    
        for name, param in model.named_parameters():
            param.requires_grad = True
            
        pretrained_path = "../params/{}_HQnet-0105.pkl".format(cross_i)
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
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            out_file = dst_path + "HQ_{}.mat".format(index)
            
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
                        out = model(xin)
                           
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
                    
                print(xin.shape)
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

