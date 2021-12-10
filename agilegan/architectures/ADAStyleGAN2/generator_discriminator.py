import random

import mmcv
import numpy as np
import torch
from mmgen.models.architectures.common import get_module_device
from mmgen.models.architectures.stylegan.generator_discriminator_v2 import (
    StyleGAN2Discriminator, StyleGANv2Generator)
from mmgen.models.builder import MODULES, build_module
from torch import nn

from .ada.augment import AugmentPipe
from .ada.misc import constant


@MODULES.register_module()
class SwapStyleGANv2Generator(StyleGANv2Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def swap_forward(self,
                     styles,
                     num_batches=-1,
                     return_noise=False,
                     return_latents=False,
                     inject_index=None,
                     truncation=1,
                     truncation_latent=None,
                     input_is_latent=False,
                     injected_noise=None,
                     randomize_noise=True,
                     swap=False,
                     swap_layer_tensor=None,
                     swap_layer_num=1):
        """Forward function incoporated with layer-swapping.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torch.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.
            swap (bool, optional): Whether use layer Swapping.
                Defaults to False.
            swap_layer_tensor (torch.Tensor, optional): Tensor used to replace
                median feature in swap forward. Defaults to None.
            swap_layer_num (int, optional): The index of swap layer.
                Defaults to 1.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary \
                containing more data.
        """
        # receive noise and conduct sanity check.
        if isinstance(styles, torch.Tensor):
            assert styles.shape[1] == self.style_channels
            styles = [styles]
        elif mmcv.is_seq_of(styles, torch.Tensor):
            for t in styles:
                assert t.shape[-1] == self.style_channels
        # receive a noise generator and sample noise.
        elif callable(styles):
            device = get_module_device(self)
            noise_generator = styles
            assert num_batches > 0
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    noise_generator((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [noise_generator((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]
        # otherwise, we will adopt default noise sampler.
        else:
            device = get_module_device(self)
            assert num_batches > 0 and not input_is_latent
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    torch.randn((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [torch.randn((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]

        if not input_is_latent:
            noise_batch = styles
            styles = [self.style_mapping(s) for s in styles]
        else:
            noise_batch = None

        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            else:
                injected_noise = [
                    getattr(self, f'injected_noise_{i}')
                    for i in range(self.num_injected_noises)
                ]
        # use truncation trick
        if truncation < 1:
            style_t = []
            # calculate truncation latent on the fly
            if truncation_latent is None and not hasattr(
                    self, 'truncation_latent'):
                self.truncation_latent = self.get_mean_latent()
                truncation_latent = self.truncation_latent
            elif truncation_latent is None and hasattr(self,
                                                       'truncation_latent'):
                truncation_latent = self.truncation_latent

            for style in styles:
                style_t.append(truncation_latent + truncation *
                               (style - truncation_latent))

            styles = style_t
        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # 4x4 stage
        out = self.constant_input(latent)
        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher resolutions
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbs):
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            # swap layer
            if swap and _index == (2 * swap_layer_num - 1):
                save_swap_layer = out
                if swap_layer_tensor is not None:
                    out = swap_layer_tensor

            skip = to_rgb(out, latent[:, _index + 2], skip)
            _index += 2

        # make sure the output image is torch.float32 to avoid RunTime Error
        # in other modules
        img = skip.to(torch.float32)

        if return_latents or return_noise:
            output_dict = dict(fake_img=img,
                               latent=latent,
                               inject_index=inject_index,
                               noise_batch=noise_batch,
                               save_swap_layer=save_swap_layer)
            return output_dict

        return img, save_swap_layer


@MODULES.register_module()
class ADAStyleGAN2Discriminator(StyleGAN2Discriminator):
    def __init__(self, in_size, *args, data_aug=None, **kwargs):
        """StyleGANv2 Discriminator with adaptive augmentation.

        Args:
            in_size (int): The input size of images.
            data_aug (dict, optional): Config for data
                augmentation. Defaults to None.
        """
        super().__init__(in_size, *args, **kwargs)
        self.with_ada = data_aug is not None
        if self.with_ada:
            self.ada_aug = build_module(data_aug)

        self.log_size = int(np.log2(in_size))

    def forward(self, x):
        """Forward function."""
        if self.with_ada:
            x = self.ada_aug.aug_pipeline(x)
        return super().forward(x)


@MODULES.register_module()
class ADAAug(nn.Module):
    """Data Augmentation Module for Adaptive Discriminator augmentation.

    Args:
        aug_pipeline (dict, optional): Config for augmentation pipeline.
            Defaults to None.
        update_interval (int, optional): Interval for updating
            augmentation probability. Defaults to 4.
        augment_initial_p (float, optional): Initial augmentation
            probability. Defaults to 0..
        ada_target (float, optional): ADA target. Defaults to 0.6.
        ada_kimg (int, optional): ADA training duration. Defaults to 500.
    """
    def __init__(self,
                 aug_pipeline=None,
                 update_interval=4,
                 augment_initial_p=0.,
                 ada_target=0.6,
                 ada_kimg=500,
                 use_slow_aug=False):
        super().__init__()
        self.aug_pipeline = AugmentPipe(**aug_pipeline)
        self.update_interval = update_interval
        self.ada_kimg = ada_kimg
        self.ada_target = ada_target

        self.aug_pipeline.p.copy_(torch.tensor(augment_initial_p))

        # this log buffer stores two numbers: num_scalars, sum_scalars.
        self.register_buffer('log_buffer', torch.zeros((2, )))

    def update(self, iteration=0, num_batches=0):
        """Update Augment probability.

        Args:
            iteration (int, optional): Training iteration.
                Defaults to 0.
            num_batches (int, optional): The number of reals batches.
                Defaults to 0.
        """

        if (iteration + 1) % self.update_interval == 0:

            adjust_step = float(num_batches * self.update_interval) / float(
                self.ada_kimg * 1000.)

            # get the mean value as the ada heuristic
            ada_heuristic = self.log_buffer[1] / self.log_buffer[0]
            adjust = np.sign(ada_heuristic.item() -
                             self.ada_target) * adjust_step
            # update the augment p
            # Note that p may be bigger than 1.0
            self.aug_pipeline.p.copy_((self.aug_pipeline.p + adjust).max(
                constant(0, device=self.log_buffer.device)))

            self.log_buffer = self.log_buffer * 0.
