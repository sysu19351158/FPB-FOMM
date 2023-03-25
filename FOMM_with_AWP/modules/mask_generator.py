import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import matplotlib.pyplot as plt
def _gaussian(dis, sigma):
    return np.exp(-np.sqrt(dis) / (2 * sigma ** 2))
import time

import dlib

def feature_generator(pic,dets, model):
    
    au_num = np.empty([18, 2], dtype = np.float32)
    predictor = model
    if len(dets) == 0:
        au_num = torch.load(r'./kp.pt').to("cpu")
    else:
        for k, d in enumerate(dets):
            shape = predictor(pic, d)
        i = 0
        vec = np.empty([68, 2], dtype=int)
        for b in range(68):
            vec[b][0] = shape.part(b).x
            vec[b][1] = shape.part(b).y
            if b in [19, 24, 27, 38, 41, 43, 46, 48, 54, 55, 57, 59]:
                au_num[i] = vec[b]
                i += 1
        for c in range(68):
            if c in [31, 35, 41, 46, 48, 54]:
                au_num[i] = vec[c]
                i += 1
        eye_distance = vec[42][0] - vec[39][0]

        au_num[12][0] = au_num[12][0] - eye_distance / 2
        au_num[13][0] = au_num[13][0] + eye_distance / 2
        au_num[14][1] = au_num[14][1] + eye_distance / 2
        au_num[15][1] = au_num[15][1] + eye_distance / 2
        au_num[16][0] = au_num[16][0] - eye_distance / 2
        au_num[17][0] = au_num[17][0] + eye_distance / 2
        au_num = torch.tensor(au_num)
    a = 0
    
    return [a, au_num.requires_grad_(True)]

def get_distance_map(points, size, sigma, kpnums):
    gmap1 = np.array(list(itertools.product(range(1, size[0] + 1), (range(1, size[1] + 1)))))
    gmap1 = np.expand_dims(gmap1, 0)
    gmap1 = np.repeat(gmap1, kpnums, 0)
    np_points = points.detach().numpy()
    gmap1 = gmap1 - np.expand_dims(np_points, 1)
    gmap1 = abs(gmap1)
    #gmap1 = gmap1[:,:,0]+gmap1[:,:,1]
    gmap1 = (gmap1[:,:,0]**2+gmap1[:,:,1]**2)**0.5
    gmap1 = _gaussian(gmap1,sigma)
    gmap1_max = np.max(gmap1, axis=1)
    gmap1_max = np.expand_dims(gmap1_max, 1)
    gmap1 = gmap1/gmap1_max
    mean = np.mean(gmap1, axis=1)
    mean = np.expand_dims(mean,1)
    thresh = (gmap1 >= mean)
    gmap1 = gmap1*thresh
    gmap1 = gmap1.reshape((kpnums,size[0],size[1]))
    return gmap1


def make_masks(key_points, img_size, feature_size, sigma):
    key_points = key_points*64/img_size[0]
    point_nums = 18
    mask = np.zeros((point_nums, *(feature_size[0], feature_size[1])))
    key_points_new = torch.zeros([point_nums,2])
    key_points_new[:, 0] = key_points[:, 1]
    key_points_new[:, 1] = key_points[:, 0]
    mask = get_distance_map(key_points_new, feature_size, sigma, point_nums)
    out = torch.tensor(mask).unsqueeze(dim=0)

    return out


if __name__ == '__main__':
    a = torch.load("kp.pt")
    print(a.shape)
    b = a[0]
    make_masks(b, (256,256), (64,64), 1.2)
