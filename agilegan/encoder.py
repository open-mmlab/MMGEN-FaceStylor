from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmgen.models.builder import MODELS, build_module
from torch.nn.parallel.distributed import _find_tensors

from .losses import id_loss
from .losses.lpips.lpips import LPIPS


@MODELS.register_module()
class PSPEncoderDecoder(nn.Module):
    """Encoder-Decoder Module for training an encoder for pre-trained
    StyleGANv2.

       Ref: https://github.com/eladrich/pixel2style2pixel/blob/master/training/coach.py # noqa

    Args:
        encoder (dict): Config for encoder.
        decoder (dict): Config for decoder.
        pool_size (tuple, optional): The pooling size for decoder output.
            Defaults to (256, 256).
        learning_rate (float, optional): Learning Rate. Defaults to 0.0001.
        l2_lambda (float, optional): Weight for l2 loss. Defaults to 1.0.
        id_lambda (float, optional): Weight for id loss. Defaults to 0.1.
        id_ckpt (str, optional): Checkpoint path used for id loss.
            Defaults to ''.
        lpips_lambda (float, optional): Weight for lpips loss.
            Defaults to 0.8.
        kl_loss (dict, optional): Config for kl loss. Defaults to None.
        train_decoder (bool, optional): Whether train decoder.
            Defaults to False.
        optim_name (str, optional): Name for desired optimizer.
            Defaults to 'adam'.
        train_cfg (dict | None, optional): Config for training schedule.
            Defaults to None.
        test_cfg (dict | None, optional): Config for testing schedule. Defaults
            to None.
    """
    def __init__(self,
                 encoder,
                 decoder,
                 pool_size=(256, 256),
                 learning_rate=0.0001,
                 l2_lambda=1.0,
                 id_lambda=0.1,
                 id_ckpt=None,
                 lpips_lambda=0.8,
                 kl_loss=None,
                 train_decoder=False,
                 optim_name='adam',
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self._encoder_cfg = deepcopy(encoder)
        self.encoder = build_module(encoder)
        self._decoder_cfg = deepcopy(decoder)
        self.decoder = build_module(decoder)

        self.pool_size = pool_size
        self.face_pool = torch.nn.AdaptiveAvgPool2d(pool_size)

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

        self.lpips_lambda = lpips_lambda
        self.id_lambda = id_lambda
        self.l2_lambda = l2_lambda
        self.id_ckpt = id_ckpt
        # loss settings
        self.rec_loss = nn.MSELoss()
        # Initialize loss
        if self.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').eval()
        if self.id_lambda > 0:
            self.id_loss = id_loss.IDLoss(self.id_ckpt).eval()
        if kl_loss is not None:
            self.kl_loss = build_module(kl_loss)

        # Initialize optimizer
        self.train_decoder = train_decoder
        self.optim_name = optim_name
        self.learning_rate = learning_rate
        self.optimizer = self.configure_optimizers()

    def configure_optimizers(self):
        params = list(self.encoder.parameters())
        if self.train_decoder:
            params += list(self.decoder.parameters())
        if self.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        else:
            raise NotImplementedError(
                f"we hasn't support {self.optim_name} optimizer yet.")
        return optimizer

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()

        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

    def forward(self,
                x,
                resize=True,
                input_code=False,
                randomize_noise=True,
                return_latents=False):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor or image code.
            resize (bool, optional): Whether resize decoder output.
                Defaults to True.
            input_code (bool, optional): whether the input is image code.
                Defaults to False.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True. Defaults to True.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a tuple with decoder output. Defaults to False.

        Returns:
            torch.Tensor | tuple: Image tensor or tuple containing image
                tensor and latent.
        """
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)

            if hasattr(self.decoder, 'module'):
                decoder = self.decoder.module
            else:
                decoder = self.decoder

            codes = [decoder.style_mapping(s) for s in codes]
            codes = torch.stack(codes, dim=0)

        input_is_latent = not input_code

        results = self.decoder([codes],
                               input_is_latent=input_is_latent,
                               randomize_noise=randomize_noise,
                               return_latents=return_latents)
        if return_latents:
            images = results['fake_img']
            result_latent = results['latent']
        else:
            images = results

        # post-process
        images = images[:, [2, 1, 0], ...]
        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def calc_loss(self, x, y, y_hat):
        loss_dict = {}
        loss = 0.0
        # id_logs = None
        if self.id_lambda > 0:
            loss_id = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            # loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.id_lambda
        if self.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.l2_lambda
        if self.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.lpips_lambda
            loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
        x = data_batch[self.real_img_key]
        y = x.clone()

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
                curr_iter = self.iteration

        self.train()
        optimizer.zero_grad()

        y_hat = self(x, return_latents=False)

        loss, loss_dict = self.calc_loss(x, y, y_hat)

        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss))

        loss.backward()
        optimizer.step()

        log_vars = {}
        log_vars.update(loss_dict)

        results = dict(orig_image=y,
                       inversion_image=y_hat,
                       orig_image_bgr=y[:, [2, 1, 0], ...],
                       inversion_image_bgr=y_hat[:, [2, 1, 0], ...])
        outputs = dict(curr_iter=curr_iter,
                       log_vars=log_vars,
                       num_samples=x.shape[0],
                       results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs
