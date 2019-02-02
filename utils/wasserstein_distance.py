# coding: utf-8
import numpy as np
from pyemd import emd
from tqdm import tqdm

def ground_distance(dim):
    # 00:00 - 24:00
    # Mon. - Sun.
    dis = np.zeros([dim, dim], float)
    for i in range(dim):
        for j in range(dim):
            # 0到24指从0搬到24
            dis[i, j] = min((j - i) % dim, (i - j) % dim)
    return dis

def get_distance_matrix(data, ground_distance):
    wasserstein_distance = np.zeros([data.shape[0], data.shape[0]], float)
    for i, p in tqdm(enumerate(data), total=data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            wasserstein_distance[j, i] = wasserstein_distance[i, j] = emd(p, data[j], ground_distance)
    return wasserstein_distance