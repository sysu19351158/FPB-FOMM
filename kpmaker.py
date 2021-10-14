#
import torch
import numpy as np
import os
import dlib
import matplotlib.pyplot as plt

seg_model = r'.\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(seg_model)

root_dir_list = [r".\data\my_SAMM\train", r".\data\my_SAMM\test"]

for root_dir in root_dir_list:
    for i in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, i)
        img_path = os.path.join(dir_path, os.listdir(dir_path)[0])
        img = plt.imread(img_path)*255
        img = img.astype('uint8')
        torch.save(feature_generator(img, predictor)[1], os.path.join(r'.\keypoint_folder',str(i)+".pt"))


def feature_generator(pic, model):
    """

    :rtype: object
    """
    au_num = np.empty([18, 2], dtype = np.float32)
    pic =pic[:,:,0:3]
    predictor = model
    detector = dlib.get_frontal_face_detector()
    dets = detector(pic, 1)
    if len(dets) == 0:
        au_num = torch.load(r'.\kp.pt').to("cpu")
        au_num = au_num[0]
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
