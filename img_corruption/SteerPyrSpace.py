import numpy as np
from SteerPyrUtils import sp5_filters, sp3_filters, sp1_filters, sp0_filters
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision import utils as vutils

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

def SteerablePyramidSpace(image, height=3, order=5, channels=1, filter_name='sp3_filters'):
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
    for _ in range(height):
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))
        for b in range(num_orientations):
            filt = filters['bfilts'][:, b].reshape(bfiltsz, bfiltsz).T
            band = corrDn(lo, filt, step=1, channels=channels)
            pyr_coeffs.append(band)
        lo = corrDn(lo, filters['lofilt'], step=2, channels=channels)

    pyr_coeffs.append(lo)
    return pyr_coeffs, harmonics, steermtx

def filter_five(img, imgSize=(80,80)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    x = x.to(device)
    x.requires_grad_(True)
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=5, 
                                                   channels=1, filter_name='sp5_filters')
    
    s1, s2, s = y[1:7], y[7:13], y
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
    
    s1d1_1 = s[1].squeeze().unsqueeze(0)
    s1d1_2 = s[2].squeeze().unsqueeze(0)
    s1d1_3 = s[3].squeeze().unsqueeze(0)
    s1d1_4 = s1_0120.squeeze().unsqueeze(0)
    s1d1_5 = s1_0150.squeeze().unsqueeze(0)
    s1d1_6 = s1_180.squeeze().unsqueeze(0)
    
    s1d2_1 = s[3].squeeze().unsqueeze(0)
    s1d2_2 = s[4].squeeze().unsqueeze(0)
    s1d2_3 = s[5].squeeze().unsqueeze(0)
    s1d2_4 = s1_0120.squeeze().unsqueeze(0)
    s1d2_5 = s1_090.squeeze().unsqueeze(0)
    s1d2_6 = s1_060.squeeze().unsqueeze(0)
    
    s1d3_1 = s[5].squeeze().unsqueeze(0)
    s1d3_2 = s[6].squeeze().unsqueeze(0)
    s1d3_3 = s1_180.squeeze().unsqueeze(0)
    s1d3_4 = s1_060.squeeze().unsqueeze(0)
    s1d3_5 = s1_030.squeeze().unsqueeze(0)
    s1d3_6 = s[1].squeeze().unsqueeze(0)
    
    s2d1_1 = s[7].squeeze().unsqueeze(0)
    s2d1_2 = s[8].squeeze().unsqueeze(0)
    s2d1_3 = s[9].squeeze().unsqueeze(0)
    s2d1_4 = s2_0120.squeeze().unsqueeze(0)
    s2d1_5 = s2_0150.squeeze().unsqueeze(0)
    s2d1_6 = s2_180.squeeze().unsqueeze(0)
    
    s2d2_1 = s[9].squeeze().unsqueeze(0)
    s2d2_2 = s[10].squeeze().unsqueeze(0)
    s2d2_3 = s[11].squeeze().unsqueeze(0)
    s2d2_4 = s2_0120.squeeze().unsqueeze(0)
    s2d2_5 = s2_090.squeeze().unsqueeze(0)
    s2d2_6 = s2_060.squeeze().unsqueeze(0)
    
    s2d3_1 = s[11].squeeze().unsqueeze(0)
    s2d3_2 = s[12].squeeze().unsqueeze(0)
    s2d3_3 = s2_180.squeeze().unsqueeze(0)
    s2d3_4 = s2_060.squeeze().unsqueeze(0)
    s2d3_5 = s2_030.squeeze().unsqueeze(0)
    s2d3_6 = s[7].squeeze().unsqueeze(0)
    
    s1d1 = torch.cat((s1d1_1, s1d1_2, s1d1_3, s1d1_4, s1d1_5, s1d1_6), 0)
    s1d2 = torch.cat((s1d2_1, s1d2_2, s1d2_3, s1d2_4, s1d2_5, s1d2_6), 0)
    s1d3 = torch.cat((s1d3_1, s1d3_2, s1d3_3, s1d3_4, s1d3_5, s1d3_6), 0)
    
    s2d1 = torch.cat((s2d1_1, s2d1_2, s2d1_3, s2d1_4, s2d1_5, s2d1_6), 0)
    s2d2 = torch.cat((s2d2_1, s2d2_2, s2d2_3, s2d2_4, s2d2_5, s2d2_6), 0)
    s2d3 = torch.cat((s2d3_1, s2d3_2, s2d3_3, s2d3_4, s2d3_5, s2d3_6), 0)
    s3 = y[13]
    return s1d1, s1d2, s1d3, s2d1, s2d2, s2d3, s3

def filter_three(img, imgSize=(80,80)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    x = x.to(device)
    x.requires_grad_(True)
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=3, 
                                                   channels=1, filter_name='sp3_filters')
    s1, s2, s = y[1:5], y[5:9], y
    s1_30 = steer(s1, np.pi/6, harmonics, steermtx)
    s1_90 = steer(s1, np.pi*3/6, harmonics, steermtx)
    s1_150 = steer(s1, np.pi*5/6, harmonics, steermtx)
    s1_030 = steer(s1, -np.pi/6, harmonics, steermtx)
    s1_090 = steer(s1, -np.pi*3/6, harmonics, steermtx)
    s1_0150 = steer(s1, -np.pi*5/6, harmonics, steermtx)
    
    s2_30 = steer(s2, np.pi/6, harmonics, steermtx)
    s2_90 = steer(s2, np.pi*3/6, harmonics, steermtx)
    s2_150 = steer(s2, np.pi*5/6, harmonics, steermtx)
    s2_030 = steer(s2, -np.pi/6, harmonics, steermtx)
    s2_090 = steer(s2, -np.pi*3/6, harmonics, steermtx)
    s2_0150 = steer(s2, -np.pi*5/6, harmonics, steermtx)
    
    s1d1_1 = s1_30.squeeze().unsqueeze(0)
    s1d1_2 = s1_0150.squeeze().unsqueeze(0)
    
    s1d2_1 = s1_90.squeeze().unsqueeze(0)
    s1d2_2 = s1_090.squeeze().unsqueeze(0)
    
    s1d3_1 = s1_150.squeeze().unsqueeze(0)
    s1d3_2 = s1_030.squeeze().unsqueeze(0)
    
    s2d1_1 = s2_30.squeeze().unsqueeze(0)
    s2d1_2 = s2_0150.squeeze().unsqueeze(0)
    
    s2d2_1 = s2_90.squeeze().unsqueeze(0)
    s2d2_2 = s2_090.squeeze().unsqueeze(0)
    
    s2d3_1 = s2_150.squeeze().unsqueeze(0)
    s2d3_2 = s2_030.squeeze().unsqueeze(0)
    
    s1d1 = torch.cat((s1d1_1, s1d1_2), 0)
    s1d2 = torch.cat((s1d2_1, s1d2_2), 0)
    s1d3 = torch.cat((s1d3_1, s1d3_2), 0)
    
    s2d1 = torch.cat((s2d1_1, s2d1_2), 0)
    s2d2 = torch.cat((s2d2_1, s2d2_2), 0)
    s2d3 = torch.cat((s2d3_1, s2d3_2), 0)
    return s1d1, s1d2, s1d3, s2d1, s2d2, s2d3

def filter_one(img, imgSize=(80,80)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    x = x.to(device)
    x.requires_grad_(True)
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=1, 
                                                   channels=1, filter_name='sp1_filters')
    s1, s2 = y[1:3], y[3:5]
    s1_30 = steer(s1, np.pi/6, harmonics, steermtx)
    s1_90 = steer(s1, np.pi*3/6, harmonics, steermtx)
    s1_150 = steer(s1, np.pi*5/6, harmonics, steermtx)
    s1_030 = steer(s1, -np.pi/6, harmonics, steermtx)
    s1_090 = steer(s1, -np.pi*3/6, harmonics, steermtx)
    s1_0150 = steer(s1, -np.pi*5/6, harmonics, steermtx)
    
    s2_30 = steer(s2, np.pi/6, harmonics, steermtx)
    s2_90 = steer(s2, np.pi*3/6, harmonics, steermtx)
    s2_150 = steer(s2, np.pi*5/6, harmonics, steermtx)
    s2_030 = steer(s2, -np.pi/6, harmonics, steermtx)
    s2_090 = steer(s2, -np.pi*3/6, harmonics, steermtx)
    s2_0150 = steer(s2, -np.pi*5/6, harmonics, steermtx)
    
    s1d1_1 = s1_30.squeeze().unsqueeze(0)
    s1d1_2 = s1_0150.squeeze().unsqueeze(0)
    
    s1d2_1 = s1_90.squeeze().unsqueeze(0)
    s1d2_2 = s1_090.squeeze().unsqueeze(0)
    
    s1d3_1 = s1_150.squeeze().unsqueeze(0)
    s1d3_2 = s1_030.squeeze().unsqueeze(0)
    
    s2d1_1 = s2_30.squeeze().unsqueeze(0)
    s2d1_2 = s2_0150.squeeze().unsqueeze(0)
    
    s2d2_1 = s2_90.squeeze().unsqueeze(0)
    s2d2_2 = s2_090.squeeze().unsqueeze(0)
    
    s2d3_1 = s2_150.squeeze().unsqueeze(0)
    s2d3_2 = s2_030.squeeze().unsqueeze(0)
    
    s1d1 = torch.cat((s1d1_1, s1d1_2), 0)
    s1d2 = torch.cat((s1d2_1, s1d2_2), 0)
    s1d3 = torch.cat((s1d3_1, s1d3_2), 0)
    
    s2d1 = torch.cat((s2d1_1, s2d1_2), 0)
    s2d2 = torch.cat((s2d2_1, s2d2_2), 0)
    s2d3 = torch.cat((s2d3_1, s2d3_2), 0)
    return s1d1, s1d2, s1d3, s2d1, s2d2, s2d3

def filter_zero(img, imgSize=(80,80)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    x = x.to(device)
    x.requires_grad_(True)
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=0, 
                                                   channels=1, filter_name='sp0_filters')
    s1, s2 = y[1].squeeze().unsqueeze(0), y[2].squeeze().unsqueeze(0)
    return s1, s2

def getSteerablePyr(img, imgSize=(80,80)):
    s1d1_5, s1d2_5, s1d3_5, s2d1_5, s2d2_5, s2d3_5, s3_5 = filter_five(img)
    s1d1_3, s1d2_3, s1d3_3, s2d1_3, s2d2_3, s2d3_3 = filter_three(img)
    s1d1_1, s1d2_1, s1d3_1, s2d1_1, s2d2_1, s2d3_1 = filter_one(img)
    s1_0, s2_0 = filter_zero(img)
    
    s1d1 = torch.cat((s1d1_5, s1d1_3, s1d1_1, s1_0), 0)
    s1d2 = torch.cat((s1d2_5, s1d2_3, s1d2_1, s1_0), 0)
    s1d3 = torch.cat((s1d3_5, s1d3_3, s1d3_1, s1_0), 0)
    
    s2d1 = torch.cat((s2d1_5, s2d1_3, s2d1_1, s2_0), 0)
    s2d2 = torch.cat((s2d2_5, s2d2_3, s2d2_1, s2_0), 0)
    s2d3 = torch.cat((s2d3_5, s2d3_3, s2d3_1, s2_0), 0)
    return s1d1, s1d2, s1d3, s2d1, s2d2, s2d3, s3_5

def getSteerablePyr10(img, imgSize=(80,80)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    x = x.to(device)
    x.requires_grad_(True)
    y, harmonics, steermtx = SteerablePyramidSpace(x, height=2, order=5, 
                                                   channels=1, filter_name='sp5_filters')
    
    hp, s1, s2, s = y[0].squeeze().unsqueeze(0), y[1:7], y[7:13], y
    s1_180 = steer(s1, np.pi, harmonics, steermtx)
    s2_180 = steer(s2, np.pi, harmonics, steermtx)
    
    s1d1_1 = s1[0].squeeze().unsqueeze(0)
    s1d1_2 = s1[2].squeeze().unsqueeze(0)
    s1d1_3 = s1[1].squeeze().unsqueeze(0)
    
    s1d2_1 = s1[2].squeeze().unsqueeze(0)
    s1d2_2 = s1[4].squeeze().unsqueeze(0)
    s1d2_3 = s1[3].squeeze().unsqueeze(0)
    
    s1d3_1 = s1[4].squeeze().unsqueeze(0)
    s1d3_2 = s1_180.squeeze().unsqueeze(0)
    s1d3_3 = s1[5].squeeze().unsqueeze(0)
    
    s2d1_1 = s2[0].squeeze().unsqueeze(0)
    s2d1_2 = s2[2].squeeze().unsqueeze(0)
    s2d1_3 = s2[1].squeeze().unsqueeze(0)
    
    s2d2_1 = s2[2].squeeze().unsqueeze(0)
    s2d2_2 = s2[4].squeeze().unsqueeze(0)
    s2d2_3 = s2[3].squeeze().unsqueeze(0)
    
    s2d3_1 = s2[4].squeeze().unsqueeze(0)
    s2d3_2 = s2_180.squeeze().unsqueeze(0)
    s2d3_3 = s2[5].squeeze().unsqueeze(0)
    
    s1d1 = torch.cat((s1d1_3, hp), 0)
    s1d2 = torch.cat((s1d2_3, hp), 0)
    s1d3 = torch.cat((s1d3_3, hp), 0)
    
    s2d1 = torch.cat((s2d1_1, s2d1_2), 0)
    s2d2 = torch.cat((s2d2_1, s2d2_2), 0)
    s2d3 = torch.cat((s2d3_1, s2d3_2), 0)
    return s1d1, s1d2, s1d3, s2d1_3, s2d2_3, s2d3_3

#if __name__ == '__main__':
#    from PIL import Image
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#    imgpath = 'C:\\Users\\12782\\Desktop\\steerpyr_test.png'
#
#    x = torch.from_numpy(np.array(Image.open('im1.png').convert("L"))).float().unsqueeze(0).unsqueeze(0)
#    x = x.to(device)#.repeat(4,1,1,1)
#    x.requires_grad_(True)
#    y = SteerablePyramidSpace(x,channels=1)
#
#    for i in range(0,22):
#        c0 = y[i][0][0]
#        print(c0.size())
#        input_tensor = c0.to(torch.device('cpu'))
#        vutils.save_image(input_tensor, "c" + str(i) + ".png")

    # c0 = y[0][0][0]
    # c1 = y[1][0][0]
    # c2 = y[2][0][0]
    # c3 = y[3][0][0]
    # c4 = y[4][0][0]
    #
    # print(y[0].size())
    # print(y[1].size())
    # print(y[2].size())
    # print(y[3].size())
    # print(y[4].size())


