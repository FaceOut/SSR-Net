from scipy.io import loadmat
import os, os.path as osp
import numpy as np

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["gender"], d["age"], d["img_size"]

def makedirs(dir):
    if not osp.isdir(dir):
        os.makedirs(dir)
