import argparse
import os
import sys

import cv2
import mmcv
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
    parser = argparse.ArgumentParser(description='AgileGAN with layer swap '
                                     'Demo')
    parser.add_argument('source_path', help='source image path')
    parser.add_argument('modelA', help='Encoder config file path')
    parser.add_argument('modelB', help='Transfer config file path')
    parser.add_argument(
        '--swap-layer', type=int, default=4, help='swap layer')
    parser.add_argument(
        '--show-input', type=bool, default=False, help='Whether show input')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/agile_result.png',
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

    _, save_swap_layer = modelA.decoder.swap_forward(
        [latents],
        input_is_latent=True,
        swap=True, swap_layer_num=args.swap_layer,
    )

    image, _ = modelB.decoder.swap_forward(
        [latents],
        input_is_latent=True,
        swap=True, swap_layer_num=args.swap_layer,
        swap_layer_tensor=save_swap_layer,
    )
    # our generator's default output channel order is bgr
    image = image[:, [2, 1, 0], ...]

    mmcv.mkdir_or_exist(os.path.dirname(args.save_path))
    utils.save_image(image, args.save_path, normalize=True)


if __name__ == '__main__':
    main()
