"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np
from skimage import img_as_ubyte
from sync_batchnorm import DataParallelWithCallback


def get_animation_region_params(source_region_params, driving_region_params, driving_region_params_initial,
                                mode='standard', avd_network=None, adapt_movement_scale=True):
    assert mode in ['standard', 'relative', 'avd']
    new_region_params = {k: v for k, v in driving_region_params.items()}
    if mode == 'standard':
        return new_region_params
    elif mode == 'relative':
        source_area = ConvexHull(source_region_params['shift'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(driving_region_params_initial['shift'][0].data.cpu().numpy()).volume
        movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

        shift_diff = (driving_region_params['shift'] - driving_region_params_initial['shift'])
        shift_diff *= movement_scale
        new_region_params['shift'] = shift_diff + source_region_params['shift']

        affine_diff = torch.matmul(driving_region_params['affine'],
                                   torch.inverse(driving_region_params_initial['affine']))
        new_region_params['affine'] = torch.matmul(affine_diff, source_region_params['affine'])
        return new_region_params
    elif mode == 'avd':
        new_region_params = avd_network(source_region_params, driving_region_params)
        return new_region_params


def animate(config, generator, region_predictor, avd_network, checkpoint, log_dir, dataset):
    animate_params = config['animate_params']
    log_dir = os.path.join(log_dir, 'animation')

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, region_predictor=region_predictor,
                        avd_network=avd_network)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        region_predictor = DataParallelWithCallback(region_predictor)
        avd_network = DataParallelWithCallback(avd_network)

    generator.eval()
    region_predictor.eval()
    avd_network.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']
            source_frame = x['source_video'][:, :, 0, :, :]
            source_region_params = region_predictor(source_frame)
            driving_region_params_initial = region_predictor(driving_video[:, :, 0])
            
            driving_frame = driving_video[:, :, 0]
            driving_region_params = region_predictor(driving_frame)

            new_region_params = driving_region_params
            first_pridiction = generator(source_frame, source_region_params=source_region_params,
                            driving_region_params=new_region_params)['prediction']
            for frame_idx in range(driving_video.shape[2]):
                first_driving_frame = driving_video[:, :, 0]
                driving_frame = driving_video[:, :, frame_idx]
                driving_region_params = region_predictor(driving_frame)

                new_region_params = driving_region_params
                out = generator(source_frame, source_region_params=source_region_params,
                                driving_region_params=new_region_params)

                out['driving_region_params'] = driving_region_params
                out['source_region_params'] = source_region_params
                out['new_region_params'] = new_region_params
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame,first_driving_frame=first_driving_frame,first_pridiction=first_pridiction,driving=driving_frame, out=out)
                                                                                    
                visualizations.append(visualization)
            predictions_ = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            image_name = result_name + animate_params['format']
            #imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
            imageio.mimsave(os.path.join(log_dir, image_name), [img_as_ubyte(frame) for frame in predictions], fps = 100)