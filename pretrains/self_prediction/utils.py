import numpy as np 
import torch 

NBIN = 8
def get_bin(deg):
    return torch.trunc((deg/180)*NBIN).to(torch.long)

def cosinus(x0, x1, x2):
    e0 = x0 - x1
    e1 = x2 - x1
    e0 = e0 / np.linalg.norm(e0)
    e1 = e1 / np.linalg.norm(e1)
    cosinus = np.dot(e0, e1)
    angle = np.arccos(cosinus)

    return 180 - np.degrees(angle)

def dihedral(x0, x1, x2, x3):
    b0 = -1.0 * (x1 - x0)
    b1 = x2 - x1
    b2 = x3 - x2
    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)
    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2) #
    y = np.dot(b0xb1_x_b1xb2, b1) * (1.0 / np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)
    grad = 180 - np.degrees(np.arctan2(y, x))
    return grad

def cosinus_torch(x0, x1, x2):
    e0 = x0 - x1
    e1 = x2 - x1
    e0 = torch.div(e0, e0.pow(2).sum(dim=1).sqrt().unsqueeze(1).repeat(1,3))
    e1 = torch.div(e1, e1.pow(2).sum(dim=1).sqrt().unsqueeze(1).repeat(1,3))
    cosinus = (e0 * e1).sum(axis=-1)
    angle = torch.arccos(cosinus)
    return 180 - torch.rad2deg(angle)


def dihedral_torch(x0, x1, x2, x3):
    b0 = -1.0 * (x1 - x0)
    b1 = x2 - x1
    b2 = x3 - x2
    b0xb1 = torch.cross(b0, b1)
    b1xb2 = torch.cross(b2, b1)
    b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2)
    x = (b0xb1 * b1xb2).sum(axis=-1)
    y = (b0xb1_x_b1xb2 * b1).sum(axis=-1) 
    y = torch.div(y, b1.pow(2).sum(dim=1).sqrt())
    angle = torch.atan2(y, x)
    return 180 - torch.abs(torch.rad2deg(angle))


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)


def torch_lexsort(a, dim=-1):
    assert dim == -1  # Transpose if you want differently
    assert a.ndim == 2  # Not sure what is numpy behaviour with > 2 dim
    # To be consistent with numpy, we flip the keys (sort by last row first)
    a_unq, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)
    return torch.argsort(inv)
