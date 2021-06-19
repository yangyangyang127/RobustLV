import torch as ch
import numpy as np

_MESHGRIDS = {}
device = ch.device('cuda')

def make_meshgrid(x):
    bs, _, _, dim = x.shape
    device = x.get_device()

    key = (dim, bs, device)
    if key in _MESHGRIDS:
        return _MESHGRIDS[key]

    space = ch.linspace(-1, 1, dim)
    meshgrid = ch.meshgrid([space, space])
    gridder = ch.cat([meshgrid[1][..., None], meshgrid[0][..., None]], dim=2)
    print(meshgrid[1][..., None])
    
    grid = gridder[None, ...].repeat(bs, 1, 1, 1)
    ones = ch.ones(grid.shape[:3] + (1,))
    final_grid = ch.cat([grid, ones], dim=3)
    expanded_grid = final_grid[..., None].to(device)

    _MESHGRIDS[key] = expanded_grid

    return expanded_grid

def unif(size, mini, maxi):
    args = {"from": mini, "to":maxi}
    return ch.to(device).FloatTensor(size=size).uniform_(**args)

def make_slice(a, b, c):
    to_cat = [a[None, ...], b[None, ...], c[None, ...]]
    return ch.cat(to_cat, dim=0)

def make_mats(rots, txs):

    rots = rots * 0.01745327778 # deg to rad
    txs = txs * 2

    cosses = ch.cos(rots)
    sins = ch.sin(rots)

    top_slice = make_slice(cosses, -sins, txs[:, 0])[None, ...].permute([2, 0, 1])
    bot_slice = make_slice(sins, cosses, txs[:, 1])[None, ...].permute([2, 0, 1])

    mats = ch.cat([top_slice, bot_slice], dim=1)

    mats = mats[:, None, None, :, :]
    mats = mats.repeat(1, 80, 80, 1, 1)
    return mats

def transform(x, rots, txs, cent=[0, 0]):
    assert x.shape[2] == x.shape[3]
    
    meshgrid = make_meshgrid(x)
        
    rot0 = ch.Tensor(np.array([0], dtype='float32')).to(device)
    cent_mats = make_mats(rot0, cent)
    new_coords1 = ch.matmul(cent_mats, meshgrid)
    ones = ch.ones(new_coords1.shape[:3] + (1,)).unsqueeze(3).to(device)
    meshgrid2 = ch.cat([new_coords1, ones], dim=3)

    tfm_mats = make_mats(rots, txs)

    new_coords = ch.matmul(tfm_mats, meshgrid2)
    new_coords = new_coords.squeeze_(-1)

    new_image = ch.nn.functional.grid_sample(x, new_coords)
    return new_image

trans_constraint = 0.1
rot_constraint = 30

def wrapper(transform):
    def tfm(x, angle=[0], distance=[0, 0]):
        rots = ch.Tensor(np.array(angle, dtype='float32')).to(device)
        txs = ch.Tensor(np.array([distance], dtype='float32')).to(device)
        return transform(x, rots, txs)
    return tfm


