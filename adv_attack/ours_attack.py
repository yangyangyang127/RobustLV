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
                                   nn.LeakyReLU(inplace=True))

        self.lstm_share = nn.Sequential(nn.LSTM(100, 100, num_layers=1))
        
        self.fcd1_last = nn.Linear(100, 5)
        self.fcd2_last = nn.Linear(100, 5)
        self.fcd3_last = nn.Linear(100, 5)

        self.task_cov_var = Variable(torch.eye(3)).to(device)
        self.class_cov_var = Variable(torch.eye(5)).to(device)
        self.feature_cov_var = Variable(torch.eye(100)).to(device)
        
        self.noise = nn.LeakyReLU(inplace=True)
                
    def forward(self, x):
        
        x = torch.reshape(x, (-1, 1, x.shape[3], x.shape[4]))
        d1, d2, d3 = SteerPyrSpace.getST_v38(x)
        
        #angle_rand = random.randint(-10, 10)
        
        #d1 = spatial.get_transformation_v24(d1, angle=60+angle_rand)
        #d2 = spatial.get_transformation_v24(d2, angle=angle_rand)
        #d3 = spatial.get_transformation_v24(d3, angle=-60+angle_rand)
        
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
        
        d1 = torch.reshape(d1, (20, -1, 100))
        d2 = torch.reshape(d2, (20, -1, 100))
        d3 = torch.reshape(d3, (20, -1, 100))
        
        d1_lstm_out, (_, _) = self.lstm_share(d1)
        d2_lstm_out, (_, _) = self.lstm_share(d2)
        d3_lstm_out, (_, _) = self.lstm_share(d3)
        
        d1_last_in = torch.reshape(d1_lstm_out, (-1, 100))
        d2_last_in = torch.reshape(d2_lstm_out, (-1, 100))
        d3_last_in = torch.reshape(d3_lstm_out, (-1, 100))
        
        outd1 = self.fcd1_last(d1_last_in)
        outd2 = self.fcd2_last(d2_last_in)
        outd3 = self.fcd3_last(d3_last_in)
        
        return outd1, outd2, outd3
    
    def noise_f(self, x, scale=0.05):
        pos = random.uniform(0, 16)
        if pos < 6:
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
        
        area = (outd1+outd2+outd3)/3
        stdloss = torch.mean(torch.abs(area-label))
        return stdloss
    
## -------------------- 训练过程 --------------------
def main(atk_type, iter_num, atk_range):

    for cross_i in [0]:
        lr = 0.001
        train_loader = DataLoader(dataset=TrainDataset(cross_i), batch_size=1, shuffle=False)
    
        model = DCAENet()
        model = model.to(device)
    
        for name, param in model.named_parameters():
            param.requires_grad = True
            
        pretrained_path = "..params/{}_v40-0345.pkl".format(cross_i)
        if os.path.exists(pretrained_path):
            pretrained_dict = torch.load(pretrained_path)
            model.load_state_dict(pretrained_dict['model'])
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
            out_file = dst_path + "v40_{}.mat".format(index)
            
            if True: #not os.path.exists(out_file):
            
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
                        outd1, outd2, outd3 = model(xin)
                           
                        loss = lossTrain(outd1, outd2, outd3, label)
        
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
                        
                    # if e == 50:
                    #     dst_path1 = "adv_example/50_{}_{}/".format(atk_type, atk_range)
                    #     if not os.path.exists(dst_path):
                    #         os.mkdir(dst_path)
                    #     out_file1 = dst_path1 + "v38_{}.mat".format(index)
                    #     dd = {'xin':xin.squeeze().detach().cpu().numpy()}
                    #     print(out_file)
                    #     sio.savemat(out_file1, dd)
                    
                dd = {'xin':xin.squeeze().detach().cpu().numpy()}
                print(out_file)
                sio.savemat(out_file, dd)

if __name__=="__main__":
    
    atk_types = ['unconstraint', 'inf', 'SMIA'] #['SMIA', '']   # , 'unconstraint', 'inf'
    iter_nums = [50]
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

