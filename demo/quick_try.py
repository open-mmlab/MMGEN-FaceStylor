import argparse
import os
import sys
from copy import deepcopy

import cv2
import mmcv
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

from mmgen.apis import init_model

sys.path.append(os.path.abspath(os.path.join(__file__,
                                             '../../')))  # isort:skip  # noqa
import agilegan  # isort:skip  # noqa
import demo.utils.normal_image as normal_image  # isort:skip  # noqa

# flake8: disable
_SUPPORTED_STYLE = {
    'toonify':
    'work_dirs/pre-trained/agile_transfer_toonify1024x1024_zplus_lpips0.5_free'
    'zeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134449-cb6785b6.pth',
    'oil':
    'work_dirs/pre-trained/agile_transfer_metfaces-oil1024x1024_zplus_lpips0.5'
    '_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134350-2b99cb9b.pth',
    'sketch':
    'work_dirs/pre-trained/agile_transfer_metfaces-sketch1024x1024_zplus_lpips'
    '0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134426-081af2a2.pth',
    'cartoon':
    'work_dirs/pre-trained/agile_transfer_photo2cartoon256x256_zplus_lpips0.5_'
    'freezeD5_ada_bs4x2_lr_1e-4_800_iter_20211201_140719-062c09fa.pth',
    'bitmoji':
    'work_dirs/pre-trained/agile_transfer_bitmoji256x256_z_wolpips_freezeD3_ad'
    'a_bs4x2_lr_1e-4_iter_1600_20211202_195819-9010a9fe.pth',
    'comic':
    'work_dirs/pre-trained/agile_transfer_face2comics256x256_z_wolpips_freezeD'
    '3_ada_bs4x2_lr_1e-4_30kiter_best_fid_iter_15000_20211201_111145-4905b63a'
    '.pth'
}
_CKPT_URL = {
    'encoder256':
    'https://download.openmmlab.com/mmgen/agilegan/agile_encoder_celebahq256x2'
    '56_lr_1e-4_150k_20211104_134520-9cce67da.pth',
    'encoder1024':
    'https://download.openmmlab.com/mmgen/agilegan/agile_encoder_ffhq1024x1024'
    '_lr_1e-4_500kiter_20211201_112111-fb1312dc.pth',
    'toonify':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_toonify1024x'
    '1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134449-c'
    'b6785b6.pth',
    'oil':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_metfaces-oil'
    '1024x1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134'
    '350-2b99cb9b.pth',
    'sketch':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_metfaces-ske'
    'tch1024x1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_'
    '134426-081af2a2.pth',
    'cartoon':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_photo2carto'
    'on256x256_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_800_iter_20211201_14'
    '0719-062c09fa.pth',
    'bitmoji':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_bitmoji256x'
    '256_z_wolpips_freezeD3_ada_bs4x2_lr_1e-4_iter_1600_20211202_195819-9010a'
    '9fe.pth',
    'comic':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_face2comics'
    '256x256_z_wolpips_freezeD3_ada_bs4x2_lr_1e-4_30kiter_best_fid_iter_15000'
    '_20211201_111145-4905b63a.pth'
}
# flake8: enable

_RES256 = ['cartoon', 'bitmoji', 'comic']
_SWAP_LAYER = {
    'toonify': 3,
    'oil': 2,
    'sketch': 1,
    'cartoon': 1,
    'bitmoji': 1,
    'comic': 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description='AgileGAN Demo')
    parser.add_argument('img_path', help='source image path')
    parser.add_argument('--style', type=str, help='style')
    parser.add_argument('--resize',
                        action='store_true',
                        help='whether resize result to 256x256')
    parser.add_argument('--batch',
                        action='store_true',
                        help='whether batch process images')
    parser.add_argument('--swap-layer',
                        type=int,
                        default=-1,
                        help='Layer index for swapping forward')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='CUDA device id')
    parser.add_argument('--save-path',
                        type=str,
                        default='./work_dirs/demos/agile_result.png',
                        help='path to save image transfer result')
    args = parser.parse_args()
    return args


def download_ckpt(filename, url):
    if os.path.exists(filename):
        return
    import wget
    mmcv.print_log(f'Downloading {url} into {filename}')
    wget.download(url, filename)


def load_image(img):
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


class AgileGANTest:
    def __init__(self,
                 encoder_config=None,
                 transfer_config=None,
                 encoder_ckpt=None,
                 transfer_ckpt=None):
        self.encoder_model = init_model(encoder_config,
                                        checkpoint=encoder_ckpt,
                                        device='cpu').eval()
        self.transfer_model = init_model(transfer_config,
                                         checkpoint=transfer_ckpt,
                                         device='cpu').eval()

    def load_in_cuda(self):
        self.encoder = deepcopy(self.encoder_model.encoder).cuda()
        self.src_gen = deepcopy(self.encoder_model.decoder).cuda()
        self.style_gen = deepcopy(self.transfer_model.generator_ema).cuda()

    def move_out_cuda(self):
        del self.encoder
        del self.src_gen
        del self.style_gen
        torch.cuda.empty_cache()

    def aligned(self, img):
        mmcv.print_log('Align Image')
        image = load_image(img)
        return image

    def inversion(self, image, resize=True):
        mmcv.print_log('Performing Projection')
        codes = self.encoder(image.cuda())
        codes = [self.src_gen.style_mapping(s) for s in codes]
        codes = torch.stack(codes, dim=0)
        return codes

    def stylization(self, codes, save_path=None, resize=True):
        mmcv.print_log('Performing Stylization')
        style_image = self.style_gen([codes],
                                     input_is_latent=True,
                                     randomize_noise=True,
                                     return_latents=False)
        if save_path is not None:
            # post-process
            style_image = style_image[:, [2, 1, 0], ...]
            if resize:
                style_image = F.adaptive_avg_pool2d(style_image, (256, 256))
            save_image(style_image, save_path, normalize=True)
        return style_image

    def layerSwap(self, codes, save_path=None, resize=True, swap_layer=1):
        mmcv.print_log('Performing Layer Swapping')
        _, save_swap_layer = self.src_gen.swap_forward(
            [codes],
            input_is_latent=True,
            swap=True,
            swap_layer_num=swap_layer,
            randomize_noise=False)

        style_image, _ = self.style_gen.swap_forward(
            [codes],
            input_is_latent=True,
            swap=True,
            swap_layer_num=swap_layer,
            swap_layer_tensor=save_swap_layer,
            randomize_noise=False)
        if save_path is not None:
            # post-process
            style_image = style_image[:, [2, 1, 0], ...]
            if resize:
                style_image = F.adaptive_avg_pool2d(style_image, (256, 256))
            save_image(style_image, save_path, normalize=True)
        return style_image

    @torch.no_grad()
    def run(self, image, save_path, style, swap_layer=-1, resize=False):
        self.load_in_cuda()
        image = self.aligned(image)
        codes = self.inversion(image)
        mmcv.mkdir_or_exist(os.path.dirname(save_path))
        if _SWAP_LAYER[style] > 0:
            style_image = self.layerSwap(
                codes,
                save_path=save_path,
                swap_layer=swap_layer
                if swap_layer > 0 else _SWAP_LAYER[style],
                resize=resize)
        else:
            style_image = self.stylization(codes,
                                           save_path=args.save_path,
                                           resize=args.resize)
        self.move_out_cuda()
        return style_image


if __name__ == '__main__':
    args = parse_args()
    assert args.style in _SUPPORTED_STYLE

    if args.style in _RES256:
        encoder_config = 'configs/demo/agile_encoder_256x256.py'
        encoder_ckpt = 'work_dirs/pre-trained/agile_encoder_celebahq256x256_lr_1e-4_150k_20211104_134520-9cce67da.pth'  # noqa

        download_ckpt(encoder_ckpt, _CKPT_URL['encoder256'])
        transfer_config = 'configs/demo/agile_transfer_256x256.py'
    else:
        encoder_config = 'configs/demo/agile_encoder_1024x1024.py'
        encoder_ckpt = 'work_dirs/pre-trained/agile_encoder_ffhq1024x1024_lr_1e-4_500kiter_20211201_112111-fb1312dc.pth'  # noqa

        download_ckpt(encoder_ckpt, _CKPT_URL['encoder1024'])
        transfer_config = 'configs/demo/agile_transfer_1024x1024.py'

    transfer_ckpt = _SUPPORTED_STYLE[args.style]
    download_ckpt(transfer_ckpt, _CKPT_URL[args.style])

    testor = AgileGANTest(encoder_config=encoder_config,
                          encoder_ckpt=encoder_ckpt,
                          transfer_config=transfer_config,
                          transfer_ckpt=transfer_ckpt)
    if args.batch:
        for filename in os.listdir(args.img_path):
            try:
                image = cv2.imread(os.path.join(args.img_path, filename))
                testor.run(image,
                           os.path.join(args.save_path, filename),
                           args.style,
                           args.swap_layer,
                           resize=args.resize)
            except Exception:
                pass
    else:
        image = cv2.imread(args.img_path)
        testor.run(image,
                   args.save_path,
                   args.style,
                   args.swap_layer,
                   resize=args.resize)
