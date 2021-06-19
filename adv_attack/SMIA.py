#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 4 21:18:44 2021

@author: xiangyzhu6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math
from torch.autograd import Variable
from torch import cuda
import scipy.stats as st

def gkern(kernlen = 3, nsig = 1):
  """ Returns a 2D Gaussian kernel array. """
  x = np.linspace(-nsig, nsig, kernlen)
  #print(x)
  kern1d = st.norm.pdf(0,x)
  kern2d = st.norm.pdf(0,x)
  #print(kern1d)
  kernel_raw = np.outer(kern1d, kern2d)
  kernel = kernel_raw / kernel_raw.sum()
  
  return kernel

def blur(tensor_image, step, stack_kerne):
	min_batch = tensor_image.shape[0]
	channels = tensor_image.shape[1]
	out_channel = channels
	kernel = torch.FloatTensor(stack_kerne).cuda()
	weight = nn.Parameter(data=kernel, requires_grad=False)
    
	tensor_image1 = torch.reshape(tensor_image, (-1, 1, tensor_image.shape[3], tensor_image.shape[4]))
	data_grad1 = F.conv2d(tensor_image1, weight, bias=None, stride=1, padding=(2,2), dilation=2)
	data_grad = torch.reshape(data_grad1, (-1, 1, 20, data_grad1.shape[2], data_grad1.shape[3]))
    
	sign_data_grad = data_grad.sign()
	
	perturbed_image = tensor_image + step * sign_data_grad
	return data_grad * step

class SMIA(object):
    def __init__(self, model=None, epsilon=None, step=None, loss_fn=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.step = step
        self.loss_fn = loss_fn

    def perturb(self, X, y, a1=1, a2=0, epsilons=None, niters=None):
        """
        Given examples (X, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        use_cuda = torch.cuda.is_available()
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
  
        X_original = Variable(torch.tensor(X.clone()), volatile=True).cuda()
        
        X_pert = Variable(torch.tensor(X.clone()), volatile=True).cuda()
        X_pert.requires_grad = True

        for i in range(niters):
            output_perturbed = None
            #output_perturbed = self.model(X_pert.cuda())
            outd1, outd2, outd3 = self.model(X_pert.cuda())
            if i == 0:
                #loss = self.loss_fn(output_perturbed, y)
                loss = self.loss_fn(outd1, outd2, outd3, y)
            else:
                
                #loss = a1 * self.loss_fn(output_perturbed, y) - a2 * self.loss_fn(output_perturbed, output_perturbed_last)
                loss = a1 * self.loss_fn(outd1, outd2, outd3, y) - a2 * self.loss_fn(outd1, outd2, outd3, output_perturbed_last) 
                print("epoch: " + str(i) + "  pureloss: " + str(loss.item()))
            loss.backward()
            X_pert_grad = X_pert.grad.detach().sign()
            pert = X_pert.grad.detach().sign() * self.step
            
            kernel = gkern(3, 1).astype(np.float32)
            #print(kernel)
            #stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
            #print(stack_kernel)
            #stack_kernel = np.expand_dims(stack_kernel, 3)
            #print(stack_kernel)
            stack_kernel = kernel[None, None, :, :]
            
            gt1 = X_pert_grad.detach()
            gt1 = blur(gt1, self.step, stack_kernel)
            X_pert = Variable(torch.tensor(X_pert + gt1), volatile=True).cuda()
            X_pert.requires_grad_(False)
            #_, output_perturbed_last = torch.max(output_perturbed, dim=1)
            #print(output_perturbed_last.shape)
            
            diff = X_pert - X_original
            diff = torch.clamp(diff, -self.epsilon, self.epsilon)
            X_pert = torch.clamp(diff + X_original, 0, 1)
            outd1_last, outd2_last, outd3_last = self.model(X_pert)
            output_perturbed_last = (outd1_last + outd2_last + outd3_last) / 3
            #output_perturbed_last = self.model(X_pert)
            

            X_pert.requires_grad = True
      
        return X_pert
    