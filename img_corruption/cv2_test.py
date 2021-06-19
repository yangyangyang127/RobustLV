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
import cv2
from array import array

if __name__=="__main__":

    datapath = "../cardiac_data.mat"

    cardiac_data = sio.loadmat(datapath)
    images = np.array(cardiac_data["images_LV"]) * 255
    
###################  read .mat and save and generate ST map #############
#    for ii in tqdm(range(0, 2900)):
#        img = images[:, :, ii]
#        for jj in range(1, 11):
#            dst_path = 'gsblur_{}'.format(jj)
#            matfile = "{}/STspace_{}.mat".format(dst_path, ii)
#            crpted_img = currupt.corrupt(img, severity=jj, corruption_name='gaussian_blur')
#            img = np.array(crpted_img, dtype='float32')
##            s1d1, s1d2, s1d3, s2d1, s2d2, s2d3 = pyr.getSteerablePyr10(img, (80, 80))
##            s1d1 = s1d1.detach().cpu().squeeze().numpy()
##            s1d2 = s1d2.detach().cpu().squeeze().numpy()
##            s1d3 = s1d3.detach().cpu().squeeze().numpy()
##            print(s1d1.shape)
##        
##            s2d1 = s2d1.detach().cpu().squeeze().numpy()
##            s2d2 = s2d2.detach().cpu().squeeze().numpy()
##            s2d3 = s2d3.detach().cpu().squeeze().numpy()
##                       
##            dd = {'s1d1':np.array(s1d1), 's1d2':np.array(s1d2), 's1d3':np.array(s1d3), 
##                  's2d3':np.array(s2d3), 's2d2':np.array(s2d2), 's2d1':np.array(s2d1)}        
#            if not os.path.exists(dst_path):
#                os.mkdir(dst_path)
##            sio.savemat(matfile, dd)
#
#            x1np = np.array(crpted_img)
#            x1min, x1max = x1np.min(), x1np.max()
#            x1np = (x1np - x1min) / (x1max - x1min) * 255
#            im1 = Image.fromarray(np.array(x1np, dtype='uint8'))
            #im1.save(dst_path + "/{}.png".format(ii))
            
    for ii in tqdm(range(0, 2900)):
        img = np.array(images[:, :, ii], dtype='uint8')
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for jj in range(1, 11):
            dst_path = 'ps_{}'.format(jj)
            matfile = "{}/STspace_{}.mat".format(dst_path, ii)
            crpted_img = currupt.corrupt(img, severity=jj, corruption_name='poisson_noise')
      
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)

            x1np = np.array(crpted_img)
            x1min, x1max = x1np.min(), x1np.max()
            x1np = (x1np - x1min) / (x1max - x1min) * 255
            im1 = Image.fromarray(np.array(x1np, dtype='uint8'))
            im1.save(dst_path + "/{}.png".format(ii))
#
#    dst_path = 'impulse_{}'.format(1)
#    img = cv2.imread(dst_path + "/{}.png".format(1))
#    cv2.imshow('img', img)
