import numpy as np


def compute_composite(dapi, ck, cd45, fitc):

    dtype = dapi.dtype
    max_val = np.iinfo(dapi.dtype).max

    dapi = dapi.astype(np.float32)
    ck = ck.astype(np.float32)
    cd45 = cd45.astype(np.float32)
    fitc = fitc.astype(np.float32)

    rgb = np.zeros((dapi.shape[0], dapi.shape[1], 3), dtype='float')

    rgb[...,0] = ck+fitc
    rgb[...,1] = cd45+fitc
    rgb[...,2] = dapi.astype(np.float32)+fitc  # why is there a random np.float32 here?
    rgb[rgb > max_val] = max_val # Clips overflow 

    rgb = rgb.astype(dtype)
    return rgb
