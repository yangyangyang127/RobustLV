import torch as torch
from PIL import Image
import scipy.io as sio
import numpy as np
import sys
import os
from multiprocessing import Process
sys.path.append("..")
from tqdm import tqdm

device = torch.device("cuda")

_MESHGRIDS = {}

def make_meshgrid(x):
    bs, _, _, dim = x.shape
    #device = x.get_device()

    key = (dim, bs, device)
    if key in _MESHGRIDS:
        return _MESHGRIDS[key]

    space = torch.linspace(-1, 1, dim)
    meshgrid = torch.meshgrid([space, space])
    gridder = torch.cat([meshgrid[1][..., None], meshgrid[0][..., None]], dim=2)
    grid = gridder[None, ...].repeat(bs, 1, 1, 1)
    ones = torch.ones(grid.shape[:3] + (1,))
    final_grid = torch.cat([grid, ones], dim=3)
    expanded_grid = final_grid[..., None].to(device)

    _MESHGRIDS[key] = expanded_grid

    return expanded_grid

def unif(size, mini, maxi):
    args = {"from": mini, "to":maxi}
    return torch.cuda.FloatTensor(size=size).uniform_(**args)

def make_slice(a, b, c):
    to_cat = [a[None, ...], b[None, ...], c[None, ...]]
    return torch.cat(to_cat, dim=0)

def make_mats(rots, txs):
    # rots: degrees
    # txs: % of image dim
    rots = rots * 0.01745327778 # deg to rad
    txs = txs * 2

    cosses = torch.cos(rots)
    sins = torch.sin(rots)

    top_slice = make_slice(cosses, -sins, txs[:, 0])[None, ...].permute([2, 0, 1])
    bot_slice = make_slice(sins, cosses, txs[:, 1])[None, ...].permute([2, 0, 1])
    #print(top_slice)
    #print(bot_slice)

    mats = torch.cat([top_slice, bot_slice], dim=1)
    #print(mats)

    mats = mats[:, None, None, :, :]
    #print(mats.shape)
    return mats

def transform(x, rots, txs):
    assert x.shape[2] == x.shape[3]

    #with torch.no_grad():
    meshgrid = make_meshgrid(x)
    tfm_mats = make_mats(rots, txs)
    tfm_mats = tfm_mats.repeat(1, x.shape[2], x.shape[2], 1, 1)

    new_coords = torch.matmul(tfm_mats, meshgrid)
    new_coords = new_coords.squeeze_(-1)

    new_image = torch.nn.functional.grid_sample(x, new_coords)
    return new_image
    
trans_constraint = 0.1
rot_constraint = 30

def wrapper(transform):
    def tfm(x, angle=[0], distance=[0, 0]):
        max_trans = trans_constraint
        max_rot = rot_constraint

        bs = x.shape[0]
        #rots = unif((bs,), -max_rot, max_rot)
        #txs = unif((bs, 2), -max_trans, max_trans)
        rots = torch.Tensor(np.array(angle, dtype='float32')).to(device)
        txs = torch.Tensor(np.array([distance], dtype='float32')).to(device)
        return transform(x, rots, txs)
    return tfm


def main(aa, bb):
    
    datapath = "../data/MMdataset/"
    corruption = 'rotation'
    
    for i in range(1, 273):
        print(i)
        
        samp_path = datapath + 'test/{}_ob.mat'.format(i)
        cardiac = sio.loadmat(samp_path)
        images_LV = np.array(cardiac['images_LV'])
        myo_LV = cardiac['myo_LV']
        pix_spa = cardiac['pix_spa']
        areas = cardiac['areas']
        dims = cardiac['dims']
        rwts = cardiac['rwt']
        
        for mm in range(0, 1):
            for nn in range(0, 1):
                for aa in range(0, 11):
                    image = torch.Tensor(images_LV)
                    img = image.to(device).unsqueeze(1)
                    new_img = wrapper(transform)(img, [aa*3.], [mm*0.02, nn*0.02])
                    new_img1 = new_img.cpu().squeeze().numpy()
                    
                    dst_path = corruption+'_{}/'.format(aa)
                    if not os.path.exists(dst_path):
                        os.mkdir(dst_path)
                    
                    sio.savemat(dst_path+'{}_ob.mat'.format(i), {'images_LV': new_img1,
                                'myo_LV': myo_LV, 'areas':areas, 'dims':dims, 'rwt':rwts, 
                                'pix_spa':pix_spa})
            
        
        
                        
if __name__ == "__main__":
    main(0, 1)
#    try:
#        for aa in range(0, 29):
#            P = Process(target = main, args=( aa * 100, 1, ))
#            print(aa)
#            P.start()
#            #_thread.start_new_thread(main, (aa*200, 1))
#    except:
#        print("Thread wrong!")
        
