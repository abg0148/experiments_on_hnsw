# utils.py

import numpy as np

def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
        if data.size == 0:
            return np.array([], dtype=np.float32).reshape(0, 0)
        dim = data[0]
    data = np.fromfile(file_path, dtype=np.float32)
    return data.reshape(-1, dim + 1)[:, 1:]


def read_ivecs(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:].astype(np.int32)
