import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#from sqrtm import sqrtm
from SPT import SteerPyrSpace
import MTlearn.tensor_op as tensor_op
from .vmanba import SS2D


device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

class L2Pooling(nn.Module):
    def __init__(self):
        super(L2Pooling, self).__init__()
        pass
    def forward(self, x):
        x = torch.mul(x, x)
        x = (torch.sum(torch.sum(x, -1), -1) + 1e-8) ** 0.5
        return x
    

class Encoder123(nn.Module):
    def __init__(self):
        super(Encoder123, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv0 = nn.Sequential(nn.Conv2d(5, 60, kernel_size=7, stride=1, padding=3, dilation=1),
                                    nn.BatchNorm2d(60),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # nn.Conv2d(60, 60, kernel_size=7, stride=1, padding=3, dilation=1),
                                    # nn.BatchNorm2d(60),
                                    # nn.LeakyReLU(0.2, inplace=True)
                                    )   # 40x40x16
                                   
        self.conv1 = nn.Sequential(nn.Conv2d(60, 120, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.BatchNorm2d(120),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # nn.Conv2d(120, 120, kernel_size=5, stride=1, padding=2, dilation=1),
                                    # nn.BatchNorm2d(120),
                                    # nn.LeakyReLU(0.2, inplace=True)
                                    )   #20*20
        self.skip1 = nn.Sequential(nn.Conv2d(60, 120, kernel_size=1, stride=1, padding=0, dilation=1),
                                    nn.BatchNorm2d(120),
                                    nn.LeakyReLU(0.2, inplace=True))
                                   
        self.conv2 = nn.Sequential(nn.Conv2d(120, 240, kernel_size=3, stride=1, padding=1, dilation=1),
                                    nn.BatchNorm2d(240),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1, dilation=1),
                                    # nn.BatchNorm2d(240),
                                    # nn.LeakyReLU(0.2, inplace=True)
                                    )   #10*10
        self.skip2 = nn.Sequential(nn.Conv2d(120, 240, kernel_size=1, stride=1, padding=0, dilation=1),
                                    nn.BatchNorm2d(240),
                                    nn.LeakyReLU(0.2, inplace=True))
                                    
        self.conv3 = nn.Sequential(nn.Conv2d(240, 480, kernel_size=3, stride=1, padding=1, dilation=1),
                                    nn.BatchNorm2d(480),
                                    nn.LeakyReLU(0.2, inplace=True))   # 5x5
        self.skip3 = nn.Sequential(nn.Conv2d(240, 480, kernel_size=1, stride=1, padding=0, dilation=1),
                                    nn.BatchNorm2d(480),
                                    nn.LeakyReLU(0.2, inplace=True))
                                    
        self.conv4 = nn.Sequential(nn.Conv2d(480, 480, kernel_size=3, stride=1, padding=1, dilation=1),
                                    nn.BatchNorm2d(480),
                                    nn.LeakyReLU(0.2, inplace=True),   # 6x6x64
                                    L2Pooling()) #10x10
        
        self.fc = nn.Sequential(nn.BatchNorm1d(480),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(480, 100),
                                nn.BatchNorm1d(100),
                                nn.Dropout(0.3),
                                nn.LeakyReLU(0.2, inplace=True))
                        
    def forward(self, x):
        #print(x.shape)
        BL, C, H, W = x.shape
        x = self.conv0(x)
        x = self.maxpool(x)
        
        x1 = self.conv1(x)
        x = x1 + self.skip1(x)
        x = self.maxpool(x)
        
        x2 = self.conv2(x)
        x = x2 + self.skip2(x)
        x = self.maxpool(x)
        
        x3 = self.conv3(x)
        x = x3 + self.skip3(x)
        x = self.maxpool(x)
        
        x = self.conv4(x).view(-1, 480)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x

class Encoder04(nn.Module):
    def __init__(self):
        super(Encoder04, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv0 = nn.Sequential(nn.Conv2d(5, 60, kernel_size=7, stride=1, padding=3, dilation=1),
                                    nn.BatchNorm2d(60),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(60, 60, kernel_size=7, stride=1, padding=3, dilation=1),
                                    nn.BatchNorm2d(60),
                                    nn.LeakyReLU(0.2, inplace=True)
                                    )   # 40x40x16
                                   
        self.conv1 = nn.Sequential(nn.Conv2d(60, 120, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.BatchNorm2d(120),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # nn.Conv2d(120, 120, kernel_size=5, stride=1, padding=2, dilation=1),
                                    # nn.BatchNorm2d(120),
                                    # nn.LeakyReLU(0.2, inplace=True)
                                    )   #20*20
                                   
        self.conv2 = nn.Sequential(nn.Conv2d(120, 240, kernel_size=3, stride=1, padding=1, dilation=1),
                                    nn.BatchNorm2d(240),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    # nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1, dilation=1),
                                    # nn.BatchNorm2d(240),
                                    # nn.LeakyReLU(0.2, inplace=True)
                                    )   #10*10
                                    
        self.conv3 = nn.Sequential(nn.Conv2d(240, 480, kernel_size=3, stride=1, padding=1, dilation=1),
                                    nn.BatchNorm2d(480),
                                    nn.LeakyReLU(0.2, inplace=True)
                                    )   # 5x5
                                    
        self.conv4 = nn.Sequential(nn.Conv2d(480, 480, kernel_size=3, stride=1, padding=1, dilation=1),
                                    nn.BatchNorm2d(480),
                                    nn.LeakyReLU(0.2, inplace=True),   # 6x6x64
                                    L2Pooling()) #10x10
        
        self.fc = nn.Sequential(nn.Linear(480, 100),
                                nn.Dropout(0.3),
                                nn.LeakyReLU(0.2, inplace=True))
                        
    def forward(self, x):

        BL, C, H, W = x.shape
        x = self.conv0(x)
        x = self.maxpool(x)
        
        x1 = self.conv1(x)
        x = self.maxpool(x1)
        
        x2 = self.conv2(x)
        x = self.maxpool(x2)
        
        x3 = self.conv3(x)
        x = self.maxpool(x3)
        
        x = self.conv4(x).view(-1, 480)
        x = self.fc(x)
        return x


class SinPositionEncoding(nn.Module):
    def __init__(self, max_sequence_length=20, d_model=100, base=1000):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.base = base

    def forward(self):
        pe = torch.zeros(self.max_sequence_length, self.d_model, dtype=torch.float)
        exp_1 = torch.arange(self.d_model // 2, dtype=torch.float)
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base ** exp_value)  # size(dmodel/2)
        out = torch.arange(self.max_sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin 
        pe[:, 1::2] = embedding_cos 
        return pe

class MHSA(nn.Module):
    def __init__(self, spacial_dim=20, embed_dim=100, num_heads=10, output_dim=100):
        super().__init__()
        self.positional_embedding = SinPositionEncoding()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # L x BS x C
        x = x + self.positional_embedding()[:, None, :].cuda()
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        return x, 0


class TemporalModel(nn.Module):
    def __init__(self, model='lstm', dim=100, nlayers=1):
        super(TemporalModel, self).__init__()
        
        if model == 'lstm':
            self.model = nn.LSTM(dim, dim, num_layers=nlayers)
        elif model == 'attention':
            self.model = MHSA(spacial_dim=20, embed_dim=dim, num_heads=10, output_dim=dim)
        elif model == 'manba':
            self.model = SS2D()
    
    def forward(self, x):
        return self.model(x)

    
class RobustNet(nn.Module):
    def __init__(self, params):
        super(RobustNet, self).__init__()
        
        self.bn1 = nn.Sequential(nn.BatchNorm2d(5),
                                 nn.LeakyReLU(0.2, inplace=True))
        
        self.bn2 = nn.Sequential(nn.BatchNorm2d(5),
                                 nn.LeakyReLU(0.2, inplace=True))
        
        self.bn3 = nn.Sequential(nn.BatchNorm2d(5),
                                  nn.LeakyReLU(0.2, inplace=True))

        if params['cross_valid'] in [0, 1, 2, 3, 4]:
            print('Encoder 04.   CrossValid: {}'.format(params['cross_valid']))
            self.encoder = Encoder04()
        else:
            self.encoder = Encoder123()
        
        self.temporal = TemporalModel(model='lstm', dim=100, nlayers=1) 
        
        self.fcd1_last = nn.Linear(100, 5)
        self.fcd2_last = nn.Linear(100, 5)
        self.fcd3_last = nn.Linear(100, 5)

        self.task_cov_var = Variable(torch.eye(3)).to(device)
        self.class_cov_var = Variable(torch.eye(5)).to(device)
        self.feature_cov_var = Variable(torch.eye(100)).to(device)
        
        self.noise = nn.LeakyReLU(inplace=True)
                
    def forward(self, x):

        batch, frm_slc = x.shape[0], x.shape[1]
        x = x.reshape(batch*frm_slc, 1, x.shape[-2], x.shape[-1])
        d1, d2, d3 = SteerPyrSpace.getSPT(x)
        
        d1 = self.noise_f(d1)
        d2 = self.noise_f(d2)
        d3 = self.noise_f(d3)
        
        d1 = self.bn1(d1)
        d2 = self.bn2(d2)
        d3 = self.bn3(d3)
        
        d1 = self.encoder(d1)
        d2 = self.encoder(d2)
        d3 = self.encoder(d3)
        
        d1 = d1.reshape(-1, frm_slc, 100).permute(1,0,2)
        d2 = d2.reshape(-1, frm_slc, 100).permute(1,0,2)
        d3 = d3.reshape(-1, frm_slc, 100).permute(1,0,2)
        
        d1_temp_out, _ = self.temporal(d1)
        d2_temp_out, _ = self.temporal(d2)
        d3_temp_out, _ = self.temporal(d3)
        
        d1_temp_out, d2_temp_out, d3_temp_out = d1_temp_out.permute(1,0,2), d2_temp_out.permute(1,0,2), d3_temp_out.permute(1,0,2)
        
        d1_last_in = torch.reshape(d1_temp_out, (-1, 100))
        d2_last_in = torch.reshape(d2_temp_out, (-1, 100))
        d3_last_in = torch.reshape(d3_temp_out, (-1, 100))
        
        outd1 = self.fcd1_last(d1_last_in)
        outd2 = self.fcd2_last(d2_last_in)
        outd3 = self.fcd3_last(d3_last_in)
        
        mt_loss = self.multitask_loss()

        return outd1, outd2, outd3, mt_loss
    
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
    
    def select_func(self, x):
            if x > 0.1:
                return 1. / x
            else:
                return x
    
    def multitask_loss(self):
        wd1 = self.fcd1_last.weight.view(1, 5, 100)
        wd2 = self.fcd2_last.weight.view(1, 5, 100)
        wd3 = self.fcd3_last.weight.view(1, 5, 100)
        weights = torch.cat((wd1, wd2, wd3), dim=0).contiguous()
   
        multi_task_loss = tensor_op.MultiTaskLoss(weights, self.task_cov_var, self.class_cov_var, self.feature_cov_var)
        return multi_task_loss
    
    def update_cov(self):
        # get updated weights
        wd1 = self.fcd1_last.weight.view(1, 5, 100)
        wd2 = self.fcd2_last.weight.view(1, 5, 100)
        wd3 = self.fcd3_last.weight.view(1, 5, 100)
        weights = torch.cat((wd1, wd2, wd3), dim=0).contiguous()

        # update cov parameters
        temp_task_cov_var = tensor_op.UpdateCov(weights.data, self.class_cov_var.data, self.feature_cov_var.data)
        temp_class_cov_var = tensor_op.UpdateCov(weights.data.permute(1, 0, 2).contiguous(), self.task_cov_var.data, self.feature_cov_var.data)
        temp_feature_cov_var = tensor_op.UpdateCov(weights.data.permute(2, 0, 1).contiguous(), self.task_cov_var.data, self.class_cov_var.data)

        # task covariance
        u, s, v = torch.svd(temp_task_cov_var)
        s = s.cpu().apply_(self.select_func).to(device)
        self.task_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
        this_trace = torch.trace(self.task_cov_var)
        if this_trace > 3000.0:        
            self.task_cov_var = Variable(self.task_cov_var / this_trace * 3000.0).to(device)
        else:
            self.task_cov_var = Variable(self.task_cov_var).to(device)

        # class covariance
        u, s, v = torch.svd(temp_class_cov_var)
        s = s.cpu().apply_(self.select_func).to(device)
        self.class_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
        this_trace = torch.trace(self.class_cov_var)
        if this_trace > 3000.0:        
            self.class_cov_var = Variable(self.class_cov_var / this_trace * 3000.0).to(device)
        else:
            self.class_cov_var = Variable(self.class_cov_var).to(device)
        
        # feature covariance
        u, s, v = torch.svd(temp_feature_cov_var)
        s = s.cpu().apply_(self.select_func).to(device)
        temp_feature_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
        this_trace = torch.trace(temp_feature_cov_var)
        if this_trace > 1000.0:        
            self.feature_cov_var += 0.001 * Variable(temp_feature_cov_var / this_trace * 1000.0).to(device)
        else:
            self.feature_cov_var += 0.001 * Variable(temp_feature_cov_var).to(device)
