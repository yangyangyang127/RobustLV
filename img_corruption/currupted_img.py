#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:41:47 2020

@author: zhxy
"""
import torch
import numpy as np
import scipy.io as sio
import os
from PIL import Image
import SteerPyrSpace as pyr
from tqdm import tqdm
import random
import __init__ as currupt


if __name__=="__main__":

    datapath = "/public/xiangyzhu6/cardiac_data.mat"

    cardiac_data = sio.loadmat(datapath)
    images = np.array(cardiac_data["images_LV"]) * 255
    areas = np.array(cardiac_data["areas"])
    rwt = np.array(cardiac_data["rwt"])
    dims = np.array(cardiac_data["dims"])
    pix = np.array(cardiac_data["pix_spa"])
    endo_lv = np.array(cardiac_data["endo_LV"]) * 255
    epi_lv = np.array(cardiac_data["epi_LV"]) * 255
    ratio = np.array(cardiac_data["ratio_resize_inverse"])
    ee = epi_lv - endo_lv
    
###################  read .mat and save and generate ST map #############
    for ii in tqdm(range(0, 2900)):
        img = images[:, :, ii]
        for jj in range(1, 11):
            dst_path = 'impulse_{}'.format(jj)
            matfile = "{}/STspace_{}.mat".format(dst_path, ii)
            crpted_img = currupt.corrupt(img, severity=jj, corruption_name='impulse_noise')
            img = np.array(crpted_img, dtype='float32')
#            s1d1, s1d2, s1d3, s2d1, s2d2, s2d3 = pyr.getSteerablePyr10(img, (80, 80))
#            s1d1 = s1d1.detach().cpu().squeeze().numpy()
#            s1d2 = s1d2.detach().cpu().squeeze().numpy()
#            s1d3 = s1d3.detach().cpu().squeeze().numpy()
#            print(s1d1.shape)
#        
#            s2d1 = s2d1.detach().cpu().squeeze().numpy()
#            s2d2 = s2d2.detach().cpu().squeeze().numpy()
#            s2d3 = s2d3.detach().cpu().squeeze().numpy()
#                       
#            dd = {'s1d1':np.array(s1d1), 's1d2':np.array(s1d2), 's1d3':np.array(s1d3), 
#                  's2d3':np.array(s2d3), 's2d2':np.array(s2d2), 's2d1':np.array(s2d1)}        
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
#            sio.savemat(matfile, dd)

            x1np = np.array(crpted_img)
            x1min, x1max = x1np.min(), x1np.max()
            x1np = (x1np - x1min) / (x1max - x1min) * 255
            im1 = Image.fromarray(np.array(x1np, dtype='uint8'))
            im1.save(dst_path + "/{}.png".format(ii))
            #im1.save("images/{}_{}.png".format(ii, jj))
