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


