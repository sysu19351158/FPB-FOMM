from torch import nn
import torch
import torch.nn.functional as F
import dlib
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
import sys
import time
import copy
sys.path.append(r".\modules")
from mask_generator import make_masks
from mask_generator import feature_generator
# from cropped import feature_generator

class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=4,#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)
        self.learn = nn.Conv2d(in_channels=int(num_kp), out_channels=int(num_kp/2), kernel_size=(3,3),padding=1)
        self.learn1 = nn.Conv2d(in_channels=int(num_kp/2), out_channels=1, kernel_size=(3,3),padding=1)
        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        seg_model = r'./shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.pre = dlib.shape_predictor(seg_model)
    def forward(self, x):
        origin_shape = x.shape
        kp=torch.zeros((origin_shape[0],18,2)).cuda()
        
        for i in range(origin_shape[0]):
            y = x[i].detach().cpu().numpy()
            y *= 255
            y = y.transpose((1, 2, 0))
            y = y.astype('uint8')
            dets = self.detector(y,1)
            y = feature_generator(y,dets,self.pre)[1]
            kp[i] = y.cuda()
        #kp = kp.requires_grad_(True)
        
        
        if self.scale_factor != 1:
            x = self.down(x)
        extract_map=torch.zeros((kp.shape[0],kp.shape[1],x.shape[2],x.shape[3])).cuda()
        for i in range(origin_shape[0]):
            extract_map[i] = make_masks(kp[i], (origin_shape[2], origin_shape[3]),(x.shape[2], x.shape[3]), 0.75)
        extract_map = self.learn1(self.learn(extract_map))
        extract_map=torch.cat((x,extract_map),1)
        feature_map = self.predictor(extract_map)

        prediction = self.kp(feature_map)



        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)
        kp = (kp * 2 / 256 - 1)
        
        out = {'value': kp}

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)
            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian
        return out
