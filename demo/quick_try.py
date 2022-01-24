import argparse
import os
import sys
from copy import deepcopy

import cv2
import mmcv
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from mmgen.apis import init_model
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(__file__,
                                             '../../')))  # isort:skip  # noqa
import agilegan  # isort:skip  # noqa
import demo.utils.normal_image as normal_image  # isort:skip  # noqa

# flake8: disable

# style and its corresponding checkpoint download url
_supported_style = {
    'toonify':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_toonify1024x1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134449-cb6785b6.pth',  # noqa
    'oil':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_metfaces-oil1024x1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134350-2b99cb9b.pth',  # noqa
    'sketch':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_metfaces-sketch1024x1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134426-081af2a2.pth',  # noqa
    'cartoon':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_photo2cartoon256x256_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_800_iter_20211201_140719-062c09fa.pth',  # noqa
    'bitmoji':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_bitmoji256x256_z_wolpips_freezeD3_ada_bs4x2_lr_1e-4_iter_1600_20211202_195819-9010a9fe.pth',  # noqa
    'comic':
    'https://download.openmmlab.com/mmgen/agilegan/agile_transfer_face2comics256x256_z_wolpips_freezeD3_ada_bs4x2_lr_1e-4_30kiter_best_fid_iter_15000_20211201_111145-4905b63a.pth'  # noqa
}

_transfer_config = dict()
_transfer_config[256] = 'configs/demo/agile_transfer_256x256.py'
_transfer_config[1024] = 'configs/demo/agile_transfer_1024x1024.py'

# flake8: enable
# style and its generator resolution
_resolution = dict(cartoon=256,
                   bitmoji=256,
                   comic=256,
                   toonify=1024,
                   oil=1024,
                   sketch=1024)
# number of layers which keep the original weights
_swap_layer = dict(toonify=3, oil=2, sketch=1, cartoon=1, bitmoji=1, comic=2)

# checkpoint and config url
_encoder_ckpt = dict()
_encoder_ckpt[
    256] = 'https://download.openmmlab.com/mmgen/agilegan/agile_encoder_celebahq256x256_lr_1e-4_150k_20211104_134520-9cce67da.pth'  # noqa
_encoder_ckpt[
    1024] = 'https://download.openmmlab.com/mmgen/agilegan/agile_encoder_ffhq1024x1024_lr_1e-4_500kiter_20211201_112111-fb1312dc.pth'  # noqa

_encoder_config = dict()
_encoder_config[256] = 'configs/demo/agile_encoder_256x256.py'
_encoder_config[1024] = 'configs/demo/agile_encoder_1024x1024.py'


def parse_args():
    parser = argparse.ArgumentParser(description='AgileGAN Demo')
    parser.add_argument('img_path', help='source image path')
    parser.add_argument('--style', type=str, help='style')
    parser.add_argument('--resize',
                        action='store_true',
                        help='whether resize result to 256x256')
    parser.add_argument('--swap-layer',
                        type=int,
                        default=-1,
                        help='Layer index for swapping forward')
    parser.add_argument('--ckpt-path',
                        type=str,
                        default='work_dirs/pre-trained',
                        help='path to save pre-trained models')
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
    mmcv.print_log(f'Downloading {url} into {filename}', 'mmgen')
    wget.download(url, filename)


class AgileGANInference:
    """This class is used to perform complete inference stage. There are three
    stages for agilegan. First crop and align face in the input image. Then get
    the latent code of the aligned image. Finally use generators to create
    stylized image by decoding latent code.

    Args:
        encoder_config ([type], optional): [description]. Defaults to None.
        transfer_config ([type], optional): [description]. Defaults to None.
        encoder_ckpt ([type], optional): [description]. Defaults to None.
        transfer_ckpt ([type], optional): [description]. Defaults to None.
    """
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
        """load encoder, source generator, stylization generator into GPU."""
        self.encoder = deepcopy(self.encoder_model.encoder).cuda()
        self.src_gen = deepcopy(self.encoder_model.decoder).cuda()
        self.style_gen = deepcopy(self.transfer_model.generator_ema).cuda()

    def move_out_cuda(self):
        """move models out of GPU and empty cache."""
        del self.encoder
        del self.src_gen
        del self.style_gen
        torch.cuda.empty_cache()

    @staticmethod
    def load_image(img_path):
        mmcv.print_log('Load and align image', 'mmgen')
        img = cv2.imread(img_path)
        assert img is not None
        # extract face and align image
        normal = normal_image.Normal_Image()
        img = normal.run(img)

        # convert arrary to tensor
        T = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        img = img.convert('RGB')
        img = T(img)
        img = img.unsqueeze(0).float()
        return img

    def inversion(self, image, resize=True):
        """get generator latent code for image."""
        mmcv.print_log('Performing Projection', 'mmgen')
        codes = self.encoder(image.cuda())
        codes = [self.src_gen.style_mapping(s) for s in codes]
        codes = torch.stack(codes, dim=0)
        return codes

    def stylization(self, codes):
        """enter the latent code to stylization generator."""
        mmcv.print_log('Performing Stylization', 'mmgen')
        style_image = self.style_gen([codes],
                                     input_is_latent=True,
                                     randomize_noise=True,
                                     return_latents=False)
        return style_image

    def layerSwap(self, codes, swap_layer=1):
        """First pass the latent code through source generator's shallow
        layers, then make the mediate feature map forward the deep layers of
        style generator.

        Args:
            codes (torch.Tensor): Latent code of input image.
            swap_layer (int, optional): The number of shallow layers.
                Defaults to 1.

        Returns:
            [type]: [description]
        """
        ''''''
        mmcv.print_log('Performing Layer Swapping', 'mmgen')
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
        return style_image

    @torch.no_grad()
    def run(self, image_path, style, swap_layer=-1):
        self.load_in_cuda()
        image = self.load_image(image_path)
        codes = self.inversion(image)
        if _swap_layer[style] > 0:
            style_image = self.layerSwap(
                codes,
                swap_layer=swap_layer
                if swap_layer > 0 else _swap_layer[style])
        else:
            style_image = self.stylization(codes)
        self.move_out_cuda()
        return style_image


def main():
    args = parse_args()
    assert args.style in _supported_style

    size = _resolution[args.style]
    # download transfer model's weight
    transfer_ckpt_name = os.path.basename(_supported_style[args.style])
    download_ckpt(os.path.join(args.ckpt_path, transfer_ckpt_name),
                  _supported_style[args.style])

    # download encoder model's weight
    encoder_ckpt_name = os.path.basename(_encoder_ckpt[size])
    download_ckpt(os.path.join(args.ckpt_path, encoder_ckpt_name),
                  _encoder_ckpt[size])

    # build inference worker
    testor = AgileGANInference(
        encoder_config=_encoder_config[size],
        encoder_ckpt=os.path.join(args.ckpt_path, encoder_ckpt_name),
        transfer_config=_transfer_config[size],
        transfer_ckpt=os.path.join(args.ckpt_path, transfer_ckpt_name))

    style_image = testor.run(args.img_path, args.style, args.swap_layer)
    # change RGB channel and resize
    style_image = style_image[:, [2, 1, 0], ...]
    if args.resize:
        style_image = F.adaptive_avg_pool2d(style_image, (256, 256))
    # save image
    mmcv.mkdir_or_exist(os.path.dirname(args.save_path))
    save_image(style_image, args.save_path, normalize=True)


if __name__ == '__main__':
    main()
