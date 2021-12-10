import argparse
import os
import sys
from collections import OrderedDict
from copy import deepcopy

import cv2
import mmcv
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import utils

from demo.utils import normal_image

# yapf: enable
# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa
import agilegan # isort:skip  # noqa

from mmgen.apis import init_model, sample_uncoditional_model  # isort:skip  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description='AgileGAN with layer swap'
                                     ' Demo')
    parser.add_argument('source_path', help='source image path')
    parser.add_argument('modelA', help='decoder config file path')
    parser.add_argument('modelB', help='decoder config file path')
    parser.add_argument(
        '--intervals', type=int, default=8, help='interval of interpolation')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')
    parser.add_argument(
        '--save-folder',
        type=str,
        default='./work_dirs/demos/dni/',
        help='path to save image transfer result')
    args = parser.parse_args()
    return args


def load_image(image_path):
    img = cv2.imread(image_path)
    assert img is not None
    normal = normal_image.Normal_Image()
    img = normal.run(img)

    T = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = img.convert('RGB')
    img = T(img)
    img = img.unsqueeze(0).float()
    return img


def dni_iterp(net_A, net_B, alpha):
    net_interp = OrderedDict()
    for k, v_A in net_A.items():
        v_B = net_B[k]
        net_interp[k] = alpha * v_A + (1 - alpha) * v_B
    return net_interp


def get_latent(decoder, codes):
    if hasattr(decoder, 'style_mapping'):
        codes = [decoder.style_mapping(s) for s in codes]
    elif hasattr(decoder, 'style'):
        codes = [decoder.style(s) for s in codes]
    else:
        raise AttributeError('Expect decoder has a style mapping module but '
                             'not found')
    codes = torch.stack(codes, dim=0)
    return codes


def main():
    args = parse_args()
    # init models
    modelA = init_model(args.modelA, checkpoint=None, device=args.device)\
        .eval()
    modelB = init_model(args.modelB, checkpoint=None, device=args.device)\
        .eval()

    src_img = load_image(args.source_path).to(args.device)
    codes = modelA.encoder(src_img)
    latents = get_latent(modelA.decoder, codes)

    net_A = deepcopy(modelA.decoder.state_dict())
    net_B = deepcopy(modelB.decoder.state_dict())

    interp_alphas = np.linspace(0, 1, args.intervals)

    mmcv.mkdir_or_exist(os.path.dirname(args.save_folder))

    for i, alpha in enumerate(interp_alphas):
        net_interp = dni_iterp(net_A, net_B, alpha)
        modelA.decoder.load_state_dict(net_interp)
        image = modelA.decoder([latents], input_is_latent=True)
        # our generator's default output channel order is bgr
        image = image[:, [2, 1, 0], ...]
        utils.save_image(image, os.path.join(args.save_folder, f'{i}.png'),
                         normalize=True)


if __name__ == '__main__':
    main()
