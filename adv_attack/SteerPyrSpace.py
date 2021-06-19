import numpy as np
from SteerPyrUtils import sp5_filters, sp3_filters, sp1_filters, sp0_filters
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision import utils as vutils
import spatial_original as spatial

def steer(basis, angle, harmonics, steermtx):
    steervect = np.zeros((1, harmonics.shape[0]*2))
    arg = angle * harmonics
    for i in range(0, harmonics.shape[0]):
        steervect[0, i*2] = np.cos(arg[i])
        steervect[0, i*2+1] = np.sin(arg[i])
    
    steervect = np.dot(steervect, steermtx)
    for i in range(0, harmonics.shape[0]*2):
        basis[i] = basis[i] * steervect[0, i]
        
    res = basis[0]
    for i in range(1, harmonics.shape[0]*2):
        res = res + basis[i]  
        
    return res

def corrDn(image, filt, step=1, channels=1):
    filt_ = torch.from_numpy(filt).float().unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1).to(image.device)
    p = (filt_.shape[2]-1)//2
    image = F.pad(image, (p,p,p,p),'reflect')
    img = F.conv2d(image, filt_, stride=step, padding=0, groups = channels)
    return img

def SteerablePyramidSpace(image, height=4, order=4, channels=1, filter_name='sp3_filters'):
    num_orientations = order + 1
    if filter_name == 'sp0_filters':
        filters = sp0_filters()
    elif filter_name == 'sp1_filters':
        filters = sp1_filters()
    elif filter_name == 'sp3_filters':
        filters = sp3_filters()
    elif filter_name == 'sp5_filters':
        filters = sp5_filters()
    harmonics, steermtx = filters['harmonics'], filters['mtx']
   
    hi0 = corrDn(image, filters['hi0filt'], step=1, channels=channels)
    pyr_coeffs = []
    pyr_coeffs.append(hi0)
    lo = corrDn(image, filters['lo0filt'], step=1, channels=channels)
    l_s0 = lo
    for _ in range(height):
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))
        for b in range(num_orientations):
            filt = filters['bfilts'][:, b].reshape(bfiltsz, bfiltsz).T
            band = corrDn(lo, filt, step=1, channels=channels)
            pyr_coeffs.append(band)
        lo = corrDn(lo, filters['lofilt'], step=2, channels=channels)

    pyr_coeffs.append(lo)
    pyr_coeffs.append(l_s0)
    return pyr_coeffs, harmonics, steermtx

def getSteerablePyr(img, imgSize=(128,128), device=torch.device("cuda")):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    x = x.to(device)
    x.requires_grad_(True)
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=5, channels=1)
    
    s1, s2 = y[1:7], y[7:13]
    s1_180 = steer(s1, np.pi, harmonics, steermtx)
    y.append(s1_180)
    s2_180 = steer(s2, np.pi, harmonics, steermtx)
    y.append(s2_180)
    
    return y

def getSteerMap(img, imgSize=(80,80), device=torch.device("cuda")):
    
    x = img
    x = x.to(device)
    x.requires_grad_(True)
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=5, channels=1)
    
    s1, s2 = y[1:7], y[7:13]
    s1_180 = steer(s1, np.pi, harmonics, steermtx)
    y.append(s1_180)
    s2_180 = steer(s2, np.pi, harmonics, steermtx)
    y.append(s2_180)
    
    s = y
    s1d1 = s[1].squeeze().unsqueeze(0) + s[2].squeeze().unsqueeze(0) + s[3].squeeze().unsqueeze(0)
    s1d2 = s[3].squeeze().unsqueeze(0) + s[4].squeeze().unsqueeze(0) + s[5].squeeze().unsqueeze(0)
    s1d3 = s[5].squeeze().unsqueeze(0) + s[6].squeeze().unsqueeze(0) + s[14].squeeze().unsqueeze(0)

    s2d1 = s[7].squeeze().unsqueeze(0) + s[8].squeeze().unsqueeze(0) + s[9].squeeze().unsqueeze(0)
    s2d2 = s[9].squeeze().unsqueeze(0) + s[10].squeeze().unsqueeze(0) + s[11].squeeze().unsqueeze(0)
    s2d3 = s[11].squeeze().unsqueeze(0) + s[12].squeeze().unsqueeze(0) + s[15].squeeze().unsqueeze(0)
    
    s2d1 = F.interpolate(s2d1.squeeze().unsqueeze(1), size=[80, 80], mode='bilinear')
    s2d2 = F.interpolate(s2d2.squeeze().unsqueeze(1), size=[80, 80], mode='bilinear')
    s2d3 = F.interpolate(s2d3.squeeze().unsqueeze(1), size=[80, 80], mode='bilinear')
    
    d1 = torch.cat((s1d2.squeeze().unsqueeze(1), s2d2), 1)
    d2 = torch.cat((s1d3.squeeze().unsqueeze(1), s2d3), 1)
    d3 = torch.cat((s1d1.squeeze().unsqueeze(1), s2d1), 1)

    return d1, d2, d3

def getSteerMap_v5(img, imgSize=(80,80), device=torch.device("cuda")):
    
    x = img
    x = x.to(device)
    x.requires_grad_(True)    
    
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=5)
    
    s1, s2 = y[1:7], y[7:13]
    s1_180 = steer(s1, np.pi, harmonics, steermtx)
    y.append(s1_180)
    s2_180 = steer(s2, np.pi, harmonics, steermtx)
    y.append(s2_180)
    
    s = y
    s1d1_1 = s[1].squeeze().unsqueeze(1)
    s1d1_2 = s[2].squeeze().unsqueeze(1)
    s1d1_3 = s[3].squeeze().unsqueeze(1)
    s1d2_1 = s[3].squeeze().unsqueeze(1)
    s1d2_2 = s[4].squeeze().unsqueeze(1)
    s1d2_3 = s[5].squeeze().unsqueeze(1)
    s1d3_1 = s[5].squeeze().unsqueeze(1)
    s1d3_2 = s[6].squeeze().unsqueeze(1)
    s1d3_3 = s1_180.squeeze().unsqueeze(1)
    
    s2d1_1 = s[7].squeeze().unsqueeze(1)
    s2d1_2 = s[8].squeeze().unsqueeze(1)
    s2d1_3 = s[9].squeeze().unsqueeze(1)
    s2d2_1 = s[9].squeeze().unsqueeze(1)
    s2d2_2 = s[10].squeeze().unsqueeze(1)
    s2d2_3 = s[11].squeeze().unsqueeze(1)
    s2d3_1 = s[11].squeeze().unsqueeze(1)
    s2d3_2 = s[12].squeeze().unsqueeze(1)
    s2d3_3 = s2_180.squeeze().unsqueeze(1)
    
    s1d1 = torch.cat((s1d1_1, s1d1_2, s1d1_3), 1)
    s1d2 = torch.cat((s1d2_1, s1d2_2, s1d2_3), 1)
    s1d3 = torch.cat((s1d3_1, s1d3_2, s1d3_3), 1)
    
    s2d1 = torch.cat((s2d1_1, s2d1_2, s2d1_3), 1)
    s2d2 = torch.cat((s2d2_1, s2d2_2, s2d2_3), 1)
    s2d3 = torch.cat((s2d3_1, s2d3_2, s2d3_3), 1)
    s2d1 = F.interpolate(s2d1, size=[80, 80], mode='bilinear')
    s2d2 = F.interpolate(s2d2, size=[80, 80], mode='bilinear')
    s2d3 = F.interpolate(s2d3, size=[80, 80], mode='bilinear')
    
    s3 = y[13]
    s3 = F.interpolate(s3.squeeze().unsqueeze(1), size=[80, 80], mode='bilinear')
    
    d1 = torch.cat((s1d1, s2d1, s3), 1)
    d2 = torch.cat((s1d2, s2d2, s3), 1)
    d3 = torch.cat((s1d3, s2d3, s3), 1)
    
    return d1, d2, d3

def getSteerMap_v9(img, imgSize=(80,80), device=torch.device("cuda")):
    
    x = img
    x = x.to(device)
    x.requires_grad_(True)    
    
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=5)
    
    s1, s2 = y[1:7], y[7:13]
    s1_180 = steer(s1, np.pi, harmonics, steermtx)
    s1_030 = steer(s1, -np.pi/6, harmonics, steermtx)
    s1_060 = steer(s1, -np.pi*2/6, harmonics, steermtx)
    s1_090 = steer(s1, -np.pi*3/6, harmonics, steermtx)
    s1_0120 = steer(s1, -np.pi*4/6, harmonics, steermtx)
    s1_0150 = steer(s1, -np.pi*5/6, harmonics, steermtx)

    s2_180 = steer(s2, np.pi, harmonics, steermtx)
    s2_030 = steer(s2, -np.pi/6, harmonics, steermtx)
    s2_060 = steer(s2, -np.pi*2/6, harmonics, steermtx)
    s2_090 = steer(s2, -np.pi*3/6, harmonics, steermtx)
    s2_0120 = steer(s2, -np.pi*4/6, harmonics, steermtx)
    s2_0150 = steer(s2, -np.pi*5/6, harmonics, steermtx)
    
    s = y
    s1d1_1 = s[1].squeeze().unsqueeze(1)
    s1d1_2 = s[2].squeeze().unsqueeze(1)
    s1d1_3 = s[3].squeeze().unsqueeze(1)
    s1d1_4 = s1_0120.squeeze().unsqueeze(1)
    s1d1_5 = s1_0150.squeeze().unsqueeze(1)
    s1d1_6 = s1_180.squeeze().unsqueeze(1)
    
    s1d2_1 = s[3].squeeze().unsqueeze(1)
    s1d2_2 = s[4].squeeze().unsqueeze(1)
    s1d2_3 = s[5].squeeze().unsqueeze(1)
    s1d2_4 = s1_0120.squeeze().unsqueeze(1)
    s1d2_5 = s1_090.squeeze().unsqueeze(1)
    s1d2_6 = s1_060.squeeze().unsqueeze(1)
    
    s1d3_1 = s[5].squeeze().unsqueeze(1)
    s1d3_2 = s[6].squeeze().unsqueeze(1)
    s1d3_3 = s1_180.squeeze().unsqueeze(1)
    s1d3_4 = s1_060.squeeze().unsqueeze(1)
    s1d3_5 = s1_030.squeeze().unsqueeze(1)
    s1d3_6 = s[1].squeeze().unsqueeze(1)
    
    s2d1_1 = s[7].squeeze().unsqueeze(1)
    s2d1_2 = s[8].squeeze().unsqueeze(1)
    s2d1_3 = s[9].squeeze().unsqueeze(1)
    s2d1_4 = s2_0120.squeeze().unsqueeze(1)
    s2d1_5 = s2_0150.squeeze().unsqueeze(1)
    s2d1_6 = s2_180.squeeze().unsqueeze(1)
    
    s2d2_1 = s[9].squeeze().unsqueeze(1)
    s2d2_2 = s[10].squeeze().unsqueeze(1)
    s2d2_3 = s[11].squeeze().unsqueeze(1)
    s2d2_4 = s2_0120.squeeze().unsqueeze(1)
    s2d2_5 = s2_090.squeeze().unsqueeze(1)
    s2d2_6 = s2_060.squeeze().unsqueeze(1)
    
    s2d3_1 = s[11].squeeze().unsqueeze(1)
    s2d3_2 = s[12].squeeze().unsqueeze(1)
    s2d3_3 = s2_180.squeeze().unsqueeze(1)
    s2d3_4 = s2_060.squeeze().unsqueeze(1)
    s2d3_5 = s2_030.squeeze().unsqueeze(1)
    s2d3_6 = s[7].squeeze().unsqueeze(1)
    
    s1d1 = torch.cat((s1d1_1, s1d1_2, s1d1_3, s1d1_4, s1d1_5, s1d1_6), 1)
    s1d2 = torch.cat((s1d2_1, s1d2_2, s1d2_3, s1d2_4, s1d2_5, s1d2_6), 1)
    s1d3 = torch.cat((s1d3_1, s1d3_2, s1d3_3, s1d3_4, s1d3_5, s1d3_6), 1)
    s1 = torch.cat((s1d1, s1d2, s1d3), 1)
    
    s2d1 = torch.cat((s2d1_1, s2d1_2, s2d1_3, s2d1_4, s2d1_5, s2d1_6), 1)
    s2d2 = torch.cat((s2d2_1, s2d2_2, s2d2_3, s2d2_4, s2d2_5, s2d2_6), 1)
    s2d3 = torch.cat((s2d3_1, s2d3_2, s2d3_3, s2d3_4, s2d3_5, s2d3_6), 1)
    s2 = torch.cat((s2d1, s2d2, s2d3), 1)
    s2 = F.interpolate(s2, size=[80, 80], mode='bilinear')
    
    s3 = y[13]
    s3 = F.interpolate(s3.squeeze().unsqueeze(1), size=[80, 80], mode='bilinear')
    
    mm = torch.cat((s1, s2, s3), 1)
    return mm

def getSteerMap_v10(img, imgSize=(80,80), device=torch.device("cuda")):
    
    x = img
    x = x.to(device)
    x.requires_grad_(True)    
    
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=5)
    
    s1, s2, s = y[1:7], y[7:13], y
    s1_180 = steer(s1, np.pi, harmonics, steermtx)
    s2_180 = steer(s2, np.pi, harmonics, steermtx)
    
    s1d1_1 = s1[0].squeeze().unsqueeze(1)
    s1d1_2 = s1[2].squeeze().unsqueeze(1)
    
    s1d2_1 = s1[2].squeeze().unsqueeze(1)
    s1d2_2 = s1[4].squeeze().unsqueeze(1)
    
    s1d3_1 = s1[4].squeeze().unsqueeze(1)
    s1d3_2 = s1_180.squeeze().unsqueeze(1)
    
    s2d1_1 = s2[0].squeeze().unsqueeze(1)
    s2d1_2 = s2[2].squeeze().unsqueeze(1)
    
    s2d2_1 = s2[2].squeeze().unsqueeze(1)
    s2d2_2 = s2[4].squeeze().unsqueeze(1)
    
    s2d3_1 = s2[4].squeeze().unsqueeze(1)
    s2d3_2 = s2_180.squeeze().unsqueeze(1)
    
    s1d1 = torch.cat((s1d1_1, s1d1_2), 1)
    s1d2 = torch.cat((s1d2_1, s1d2_2), 1)
    s1d3 = torch.cat((s1d3_1, s1d3_2), 1)
    
    s2d1 = torch.cat((s2d1_1, s2d1_2), 1)
    s2d2 = torch.cat((s2d2_1, s2d2_2), 1)
    s2d3 = torch.cat((s2d3_1, s2d3_2), 1)
    s2d1 = F.interpolate(s2d1, size=[80, 80], mode='bilinear')
    s2d2 = F.interpolate(s2d2, size=[80, 80], mode='bilinear')
    s2d3 = F.interpolate(s2d3, size=[80, 80], mode='bilinear')
    
    d1 = torch.cat((s1d1, s2d1), 1)
    d2 = torch.cat((s1d2, s2d2), 1)
    d3 = torch.cat((s1d3, s2d3), 1)
    return d1, d2, d3

def getSteerMap_v11(img, imgSize=(80,80), device=torch.device("cuda")):
    
    x = img
    x = x.to(device)
    x.requires_grad_(True)    
    
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=5)
    
    s1, s2 = y[1:7], y[7:13]
    s1d1 = s1[1].squeeze().unsqueeze(1)
    s1d2 = s1[3].squeeze().unsqueeze(1)
    s1d3 = s1[5].squeeze().unsqueeze(1)
    
    s2d1 = s2[1].squeeze().unsqueeze(1)
    s2d2 = s2[3].squeeze().unsqueeze(1)    
    s2d3 = s2[5].squeeze().unsqueeze(1)
    
    s2d1 = F.interpolate(s2d1, size=[80, 80], mode='bilinear')
    s2d2 = F.interpolate(s2d2, size=[80, 80], mode='bilinear')
    s2d3 = F.interpolate(s2d3, size=[80, 80], mode='bilinear')
    
    d1 = torch.cat((s1d1, s2d1), 1)
    d2 = torch.cat((s1d2, s2d2), 1)
    d3 = torch.cat((s1d3, s2d3), 1) 
    return d1, d2, d3

def filter_five(img, imgSize=(80,80)):

    y, harmonics, steermtx = SteerablePyramidSpace(img, height=2, order=5, 
                                                   channels=1, filter_name='sp5_filters')
    
    s1, s2 = y[1:7], y[7:13]
    s1d1_1, s1d1_2, s1d1_3 = s1[0].squeeze().unsqueeze(1), s1[1].squeeze().unsqueeze(1), s1[2].squeeze().unsqueeze(1)
    s1d2_1, s1d2_2, s1d2_3 = s1[2].squeeze().unsqueeze(1), s1[3].squeeze().unsqueeze(1), s1[4].squeeze().unsqueeze(1)
    s1_180 = steer(s1, -np.pi, harmonics, steermtx)
    s1d3_1, s1d3_2, s1d3_3 = s1[4].squeeze().unsqueeze(1), s1[5].squeeze().unsqueeze(1), s1_180.squeeze().unsqueeze(1)
    
    s2d1_1, s2d1_2 = s2[0].squeeze().unsqueeze(1), s2[2].squeeze().unsqueeze(1)
    s2d2_1, s2d2_2 = s2[2].squeeze().unsqueeze(1), s2[4].squeeze().unsqueeze(1)
    s2_180 = steer(s2, -np.pi, harmonics, steermtx)
    s2d3_1, s2d3_2 = s2[4].squeeze().unsqueeze(1), s2_180.squeeze().unsqueeze(1)
    
    s1d1 = torch.cat((s1d1_1, s1d1_2, s1d1_3), 1)
    s1d2 = torch.cat((s1d2_1, s1d2_2, s1d2_3), 1)
    s1d3 = torch.cat((s1d3_1, s1d3_2, s1d3_3), 1)
    
    s2d1 = torch.cat((s2d1_1, s2d1_2), 1)
    s2d2 = torch.cat((s2d2_1, s2d2_2), 1)
    s2d3 = torch.cat((s2d3_1, s2d3_2), 1)
    s3 = y[13].squeeze().unsqueeze(1)

    s2d1 = F.interpolate(s2d1, size=[80, 80], mode='bilinear')
    s2d2 = F.interpolate(s2d2, size=[80, 80], mode='bilinear')
    s2d3 = F.interpolate(s2d3, size=[80, 80], mode='bilinear')
    s3 = F.interpolate(s3, size=[80, 80], mode='bilinear')
    
    return s1d1, s1d2, s1d3, s2d1, s2d2, s2d3, s3

def filter_three(img, hei=2, imgSize=(80,80)):
    y, harmonics, steermtx = SteerablePyramidSpace(img, height=hei, order=3, 
                                                   channels=1, filter_name='sp3_filters')
    s1, s2 = y[1:5], y[5:9]
    s1d1 = steer(s1, np.pi/6, harmonics, steermtx)
    s1d2 = steer(s1, np.pi*3/6, harmonics, steermtx)
    s1d3 = steer(s1, np.pi*5/6, harmonics, steermtx)
    
    s2d1 = steer(s2, np.pi/6, harmonics, steermtx)
    s2d2 = steer(s2, np.pi*3/6, harmonics, steermtx)
    s2d3 = steer(s2, np.pi*5/6, harmonics, steermtx)
    
    s1d1, s1d2 = s1d1.squeeze().unsqueeze(1), s1d2.squeeze().unsqueeze(1)
    s1d3 = s1d3.squeeze().unsqueeze(1)
    s2d1, s2d2 = s2d1.squeeze().unsqueeze(1), s2d2.squeeze().unsqueeze(1)
    s2d3 = s2d3.squeeze().unsqueeze(1)
    
    s2d1 = F.interpolate(s2d1, size=[80, 80], mode='bilinear')
    s2d2 = F.interpolate(s2d2, size=[80, 80], mode='bilinear')
    s2d3 = F.interpolate(s2d3, size=[80, 80], mode='bilinear')
    
    s3 = y[9].squeeze().unsqueeze(1)
    s3 = F.interpolate(s3, size=[80, 80], mode='bilinear')
        
    return s1d1, s1d2, s1d3, s2d1, s2d2, s2d3, s3

def getSteerMap_v12(img, imgSize=(80,80), device=torch.device("cuda")):
    
    x = img
    x = x.to(device)
    x.requires_grad_(True)
    
    s1d1_5, s1d2_5, s1d3_5, s2d1_5, s2d2_5, s2d3_5, s3_5 = filter_five(x)
    s1d1_3, s1d2_3, s1d3_3, s2d1_3, s2d2_3, s2d3_3 = filter_three(x)
    
    d1 = torch.cat((s1d1_5, s2d1_5, s1d1_3, s2d1_3, s3_5), 1)
    d2 = torch.cat((s1d2_5, s2d2_5, s1d2_3, s2d2_3, s3_5), 1)
    d3 = torch.cat((s1d3_5, s2d3_5, s1d3_3, s2d3_3, s3_5), 1) 
    return d1, d2, d3


def getST_v25(img, hei=3, imgSize=(80,80)):

    y, harmonics, steermtx = SteerablePyramidSpace(img, height=3, order=5, channels=1, filter_name='sp5_filters')
    
    s1, s2, s3 = y[1:7], y[7:13], y[13:19]
    s1d1_1, s1d2_1, s1d3_1 = s1[0].squeeze().unsqueeze(1), s1[2].squeeze().unsqueeze(1), s1[4].squeeze().unsqueeze(1)
    s1d1_2, s1d2_2, s1d3_2 = s1[1].squeeze().unsqueeze(1), s1[3].squeeze().unsqueeze(1), s1[5].squeeze().unsqueeze(1)
    s2d1_1, s2d2_1, s2d3_1 = s2[0].squeeze().unsqueeze(1), s2[2].squeeze().unsqueeze(1), s2[4].squeeze().unsqueeze(1)
    s2d1_2, s2d2_2, s2d3_2 = s2[1].squeeze().unsqueeze(1), s2[3].squeeze().unsqueeze(1), s2[5].squeeze().unsqueeze(1)
    s3d1_1, s3d2_1, s3d3_1 = s3[0].squeeze().unsqueeze(1), s3[2].squeeze().unsqueeze(1), s3[4].squeeze().unsqueeze(1)
    s3d1_2, s3d2_2, s3d3_2 = s3[1].squeeze().unsqueeze(1), s3[3].squeeze().unsqueeze(1), s3[5].squeeze().unsqueeze(1)
  
    s2d1_1 = F.interpolate(s2d1_1, size=[80, 80], mode='bilinear')
    s2d2_1 = F.interpolate(s2d2_1, size=[80, 80], mode='bilinear')
    s2d3_1 = F.interpolate(s2d3_1, size=[80, 80], mode='bilinear')
    
    s2d1_2 = F.interpolate(s2d1_2, size=[80, 80], mode='bilinear')
    s2d2_2 = F.interpolate(s2d2_2, size=[80, 80], mode='bilinear')
    s2d3_2 = F.interpolate(s2d3_2, size=[80, 80], mode='bilinear')
    
    s3d1_1 = F.interpolate(s3d1_1, size=[80, 80], mode='bilinear')
    s3d2_1 = F.interpolate(s3d2_1, size=[80, 80], mode='bilinear')
    s3d3_1 = F.interpolate(s3d3_1, size=[80, 80], mode='bilinear')
    
    s3d1_2 = F.interpolate(s3d1_2, size=[80, 80], mode='bilinear')
    s3d2_2 = F.interpolate(s3d2_2, size=[80, 80], mode='bilinear')
    s3d3_2 = F.interpolate(s3d3_2, size=[80, 80], mode='bilinear')
    
    d1 = torch.cat((s1d1_1, s1d1_2, s2d1_1, s2d1_2, s3d1_1, s3d1_2), 1)
    d2 = torch.cat((s1d2_1, s1d2_2, s2d2_1, s2d2_2, s3d2_1, s3d2_2), 1)
    d3 = torch.cat((s1d3_1, s1d3_2, s2d3_1, s2d3_2, s3d3_1, s3d3_2), 1)
    all_d = torch.cat((d1, d2, d3), 1)
    return all_d

def getST_v26(img, hei=3, imgSize=(80,80)):

    y, harmonics, steermtx = SteerablePyramidSpace(img, height=3, order=5, channels=1, filter_name='sp5_filters')
    s1, s2, s3 = y[1:7], y[7:13], y[13:19]   
    s1d1_1, s1d2_1, s1d3_1 = s1[0].squeeze().unsqueeze(1), s1[2].squeeze().unsqueeze(1), s1[4].squeeze().unsqueeze(1)
    s1d1_2, s1d2_2, s1d3_2 = s1[1].squeeze().unsqueeze(1), s1[3].squeeze().unsqueeze(1), s1[5].squeeze().unsqueeze(1)
    s2d1_1, s2d2_1, s2d3_1 = s2[0].squeeze().unsqueeze(1), s2[2].squeeze().unsqueeze(1), s2[4].squeeze().unsqueeze(1)
    s2d1_2, s2d2_2, s2d3_2 = s2[1].squeeze().unsqueeze(1), s2[3].squeeze().unsqueeze(1), s2[5].squeeze().unsqueeze(1)
    s3d1_1, s3d2_1, s3d3_1 = s3[0].squeeze().unsqueeze(1), s3[2].squeeze().unsqueeze(1), s3[4].squeeze().unsqueeze(1)
    s3d1_2, s3d2_2, s3d3_2 = s3[1].squeeze().unsqueeze(1), s3[3].squeeze().unsqueeze(1), s3[5].squeeze().unsqueeze(1)
  
    s2d1_1 = F.interpolate(s2d1_1, size=[80, 80], mode='bilinear')
    s2d2_1 = F.interpolate(s2d2_1, size=[80, 80], mode='bilinear')
    s2d3_1 = F.interpolate(s2d3_1, size=[80, 80], mode='bilinear')
    
    s2d1_2 = F.interpolate(s2d1_2, size=[80, 80], mode='bilinear')
    s2d2_2 = F.interpolate(s2d2_2, size=[80, 80], mode='bilinear')
    s2d3_2 = F.interpolate(s2d3_2, size=[80, 80], mode='bilinear')
    
    s3d1_1 = F.interpolate(s3d1_1, size=[80, 80], mode='bilinear')
    s3d2_1 = F.interpolate(s3d2_1, size=[80, 80], mode='bilinear')
    s3d3_1 = F.interpolate(s3d3_1, size=[80, 80], mode='bilinear')
    
    s3d1_2 = F.interpolate(s3d1_2, size=[80, 80], mode='bilinear')
    s3d2_2 = F.interpolate(s3d2_2, size=[80, 80], mode='bilinear')
    s3d3_2 = F.interpolate(s3d3_2, size=[80, 80], mode='bilinear')
    
    s1 = torch.cat((s1d1_1, s1d1_2, s1d2_1, s1d2_2, s1d3_1, s1d3_2), 1)
    s2 = torch.cat((s2d1_1, s2d1_2, s2d2_1, s2d2_2, s2d3_1, s2d3_2), 1)
    s3 = torch.cat((s3d1_1, s3d1_2, s3d2_1, s3d2_2, s3d3_1, s3d3_2), 1)    
    return s1, s2, s3

def getST_v27(img, hei=3, imgSize=(80,80)):

    y, harmonics, steermtx = SteerablePyramidSpace(img, height=3, order=5, channels=1, filter_name='sp5_filters')
    s1, s2, s3 = y[1:7], y[7:13], y[13:19]   
    s1d1_1, s1d2_1, s1d3_1 = s1[0].squeeze().unsqueeze(1), s1[2].squeeze().unsqueeze(1), s1[4].squeeze().unsqueeze(1)
    s1d1_2, s1d2_2, s1d3_2 = s1[1].squeeze().unsqueeze(1), s1[3].squeeze().unsqueeze(1), s1[5].squeeze().unsqueeze(1)
    s2d1_1, s2d2_1, s2d3_1 = s2[0].squeeze().unsqueeze(1), s2[2].squeeze().unsqueeze(1), s2[4].squeeze().unsqueeze(1)
    s2d1_2, s2d2_2, s2d3_2 = s2[1].squeeze().unsqueeze(1), s2[3].squeeze().unsqueeze(1), s2[5].squeeze().unsqueeze(1)
    s3d1_1, s3d2_1, s3d3_1 = s3[0].squeeze().unsqueeze(1), s3[2].squeeze().unsqueeze(1), s3[4].squeeze().unsqueeze(1)
    s3d1_2, s3d2_2, s3d3_2 = s3[1].squeeze().unsqueeze(1), s3[3].squeeze().unsqueeze(1), s3[5].squeeze().unsqueeze(1)
  
    s2d1_1 = F.interpolate(s2d1_1, size=[80, 80], mode='bilinear')
    s2d2_1 = F.interpolate(s2d2_1, size=[80, 80], mode='bilinear')
    s2d3_1 = F.interpolate(s2d3_1, size=[80, 80], mode='bilinear')
    
    s2d1_2 = F.interpolate(s2d1_2, size=[80, 80], mode='bilinear')
    s2d2_2 = F.interpolate(s2d2_2, size=[80, 80], mode='bilinear')
    s2d3_2 = F.interpolate(s2d3_2, size=[80, 80], mode='bilinear')
    
    s3d1_1 = F.interpolate(s3d1_1, size=[80, 80], mode='bilinear')
    s3d2_1 = F.interpolate(s3d2_1, size=[80, 80], mode='bilinear')
    s3d3_1 = F.interpolate(s3d3_1, size=[80, 80], mode='bilinear')
    
    s3d1_2 = F.interpolate(s3d1_2, size=[80, 80], mode='bilinear')
    s3d2_2 = F.interpolate(s3d2_2, size=[80, 80], mode='bilinear')
    s3d3_2 = F.interpolate(s3d3_2, size=[80, 80], mode='bilinear')
    
    d1 = torch.cat((s1d1_1, s2d1_1, s3d1_1), 1)
    d2 = torch.cat((s1d1_2, s2d1_2, s3d1_2), 1)
    d3 = torch.cat((s1d2_1, s2d2_1, s3d2_1), 1)
    d4 = torch.cat((s1d2_2, s2d2_2, s3d2_2), 1)
    d5 = torch.cat((s1d3_1, s2d3_1, s3d3_1), 1)
    d6 = torch.cat((s1d3_2, s2d3_2, s3d3_2), 1)
    return d1, d2, d3, d4, d5, d6

def getST_v31(img, hl, hei=3, imgSize=(80,80)):

    y, harmonics, steermtx = SteerablePyramidSpace(img, height=3, order=5, channels=1, filter_name='sp5_filters')
    
    h, l = y[0].squeeze().unsqueeze(1), y[19].squeeze().unsqueeze(1)
    s1, s2, s3 = y[1:7], y[7:13], y[13:19]
    s1d1_1, s1d2_1, s1d3_1 = s1[0].squeeze().unsqueeze(1), s1[2].squeeze().unsqueeze(1), s1[4].squeeze().unsqueeze(1)
    s1d1_2, s1d2_2, s1d3_2 = s1[1].squeeze().unsqueeze(1), s1[3].squeeze().unsqueeze(1), s1[5].squeeze().unsqueeze(1)
    s2d1_1, s2d2_1, s2d3_1 = s2[0].squeeze().unsqueeze(1), s2[2].squeeze().unsqueeze(1), s2[4].squeeze().unsqueeze(1)
    s2d1_2, s2d2_2, s2d3_2 = s2[1].squeeze().unsqueeze(1), s2[3].squeeze().unsqueeze(1), s2[5].squeeze().unsqueeze(1)
    s3d1_1, s3d2_1, s3d3_1 = s3[0].squeeze().unsqueeze(1), s3[2].squeeze().unsqueeze(1), s3[4].squeeze().unsqueeze(1)
    s3d1_2, s3d2_2, s3d3_2 = s3[1].squeeze().unsqueeze(1), s3[3].squeeze().unsqueeze(1), s3[5].squeeze().unsqueeze(1)
  
    s2d1_1 = F.interpolate(s2d1_1, size=[80, 80], mode='bilinear')
    s2d2_1 = F.interpolate(s2d2_1, size=[80, 80], mode='bilinear')
    s2d3_1 = F.interpolate(s2d3_1, size=[80, 80], mode='bilinear')
    
    s2d1_2 = F.interpolate(s2d1_2, size=[80, 80], mode='bilinear')
    s2d2_2 = F.interpolate(s2d2_2, size=[80, 80], mode='bilinear')
    s2d3_2 = F.interpolate(s2d3_2, size=[80, 80], mode='bilinear')
    
    s3d1_1 = F.interpolate(s3d1_1, size=[80, 80], mode='bilinear')
    s3d2_1 = F.interpolate(s3d2_1, size=[80, 80], mode='bilinear')
    s3d3_1 = F.interpolate(s3d3_1, size=[80, 80], mode='bilinear')
    
    s3d1_2 = F.interpolate(s3d1_2, size=[80, 80], mode='bilinear')
    s3d2_2 = F.interpolate(s3d2_2, size=[80, 80], mode='bilinear')
    s3d3_2 = F.interpolate(s3d3_2, size=[80, 80], mode='bilinear')
    
    l = F.interpolate(l, size=[80, 80], mode='bilinear')
    
    d1 = torch.cat((s1d1_1, s1d1_2, s2d1_1, s2d1_2, s3d1_1, s3d1_2), 1)
    d2 = torch.cat((s1d2_1, s1d2_2, s2d2_1, s2d2_2, s3d2_1, s3d2_2), 1)
    d3 = torch.cat((s1d3_1, s1d3_2, s2d3_1, s2d3_2, s3d3_1, s3d3_2), 1)
    
    if hl=='h':
        all_d = torch.cat((h, d1, d2, d3), 1)
    elif hl=='l':
        all_d = torch.cat((l, d1, d2, d3), 1)
    return all_d

def getST_v33(img, hei=3, imgSize=(80,80)):

    y, harmonics, steermtx = SteerablePyramidSpace(img, height=3, order=5, channels=1, filter_name='sp5_filters')
    h, l = y[0].squeeze().unsqueeze(1), y[19].squeeze().unsqueeze(1)
    return h

def getST_v34(img, hei=3, imgSize=(80,80)):

    y, harmonics, steermtx = SteerablePyramidSpace(img, height=3, order=5, channels=1, filter_name='sp5_filters')
    s_l0 = y[20].squeeze().unsqueeze(1)
    return s_l0

def getST_v38(img, imgSize=(80,80), device=torch.device("cuda")):
    
    x = img
    x = x.to(device)
    x.requires_grad_(True)
    
    #s1d1_5, s1d2_5, s1d3_5, s2d1_5, s2d2_5, s2d3_5, s3_5 = filter_five(x)
    s1d1_3, s1d2_3, s1d3_3, s2d1_3, s2d2_3, s2d3_3, s3_3 = filter_three(x)
    
    d1 = torch.cat((s1d1_3, s2d1_3, s3_3), 1)
    d2 = torch.cat((s1d2_3, s2d2_3, s3_3), 1)
    d3 = torch.cat((s1d3_3, s2d3_3, s3_3), 1)
    return d1, d2, d3


if __name__ == '__main__':
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgpath = 'C:\\Users\\12782\\Desktop\\steerpyr_test.png'
    
    aa = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
    bb = torch.Tensor(aa).unsqueeze(0).unsqueeze(0)
    b = torch.zeros((6, 6)).unsqueeze(0).unsqueeze(0)
    bb1 = bb.expand_as(b)
    print(bb1)
    
