from copy import deepcopy

import torch
import torch.nn.functional as F
from mmgen.models.builder import MODELS, build_module
from mmgen.models.common import set_requires_grad
from mmgen.models.gans.static_unconditional_gan import StaticUnconditionalGAN
from torch.nn.parallel.distributed import _find_tensors

from .losses.lpips.lpips import LPIPS

from mmgen.models.architectures.common import get_module_device  # isort:skip  # noqa


def requires_grad(model, flag=True, target_layer=None):
    """Set the `requires_grad` of the model target layer to flag.

    Args:
        model (nn.Module): Model to be set.
        flag (bool, optional): Flag for `requires_grad`.
            Defaults to True.
        target_layer (str | None, optional): Name or Key words of
            target layer. Defaults to None.
    """
    for name, param in model.named_parameters():
        if target_layer is None or target_layer in name:
            param.requires_grad = flag


@MODELS.register_module()
class PSPTransfer(StaticUnconditionalGAN):
    def __init__(self,
                 src_generator,
                 generator,
                 discriminator,
                 gan_loss,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 lpips_lambda=0.8,
                 freezeG=-1,
                 freezeD=-1,
                 freezeStyle=-1,
                 structure_loss_layer=-1,
                 sample_space='zplus',
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(generator,
                         discriminator,
                         gan_loss,
                         disc_auxiliary_loss=disc_auxiliary_loss,
                         gen_auxiliary_loss=gen_auxiliary_loss,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg)
        self.src_g_cfg = deepcopy(src_generator)
        self.source_generator = build_module(src_generator)
        self.lpips_lambda = lpips_lambda
        if self.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='vgg').eval()
        else:
            self.lpips_loss = None
        self.structure_loss_layer = structure_loss_layer
        set_requires_grad(self.source_generator, False)
        set_requires_grad(self.generator.style_mapping, False)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # freeze parameters
        self.freezeD = freezeD
        self.freezeG = freezeG
        self.freezeStyle = freezeStyle
        self.sample_space = sample_space

    def latent_generator(self, batch_size):
        device = get_module_device(self.generator)
        z_plus_code = torch.randn(batch_size, 18, 512).to(device)
        w_plus_code = [
            self.source_generator.style_mapping(s) for s in z_plus_code
        ]
        w_plus_code = [torch.stack(w_plus_code, dim=0)]
        return w_plus_code

    def _get_gen_loss(self, outputs_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake_g'] = self.gan_loss(
            outputs_dict['disc_pred_fake_g'],
            target_is_real=True,
            is_disc=False)
        # TODO: add modified LPIPS
        source_results = self.source_generator(outputs_dict['latents'],
                                               input_is_latent=True)
        resized_source_results = self.face_pool(source_results)
        resized_target_results = self.face_pool(outputs_dict['fake_imgs'])
        # lpip loss
        if self.lpips_loss is not None:
            losses_dict['loss_sim'] = self.lpips_lambda * self.lpips_loss(
                x=resized_source_results, y=resized_target_results)
        # structure loss
        if self.structure_loss_layer > 0:
            losses_dict['loss_structure'] = 0.
            for layer in range(self.structure_loss_layer):
                generator_source = outputs_dict['gen_source']
                generator = outputs_dict['gen'].module
                _, latent_med_sor = generator_source.swap_forward(
                    outputs_dict['latents'],
                    input_is_latent=True,
                    swap=True,
                    swap_layer_num=layer + 1)
                _, latent_med_tar = generator.swap_forward(
                    outputs_dict['latents'],
                    input_is_latent=True,
                    swap=True,
                    swap_layer_num=layer + 1)
                losses_dict['loss_structure'] += F.mse_loss(
                    latent_med_tar, latent_med_sor)
        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs_dict)
                if loss_ is None:
                    continue

                # mmcv.print_log(f'get loss for {loss_module.name()}')
                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var, source_results

    def freeze_before_train_d(self):
        requires_grad(self.generator, False)
        requires_grad(self.discriminator, False)

        # obtain some params
        g_log_size = self.generator.module.log_size
        if hasattr(self.generator.module, 'num_layers'):
            g_num_layers = self.generator.module.num_layers
        else:
            g_num_layers = self.generator.module.num_injected_noises
        d_log_size = self.discriminator.module.log_size
        # Freeze !!!
        # set_requires_grad(self.discriminator, True)
        if self.freezeG > 0 and self.freezeD > 0:
            # G
            for layer in range(self.freezeG):
                requires_grad(self.generator,
                              False,
                              target_layer=f'convs.{g_num_layers-2-2*layer}')
                requires_grad(self.generator,
                              False,
                              target_layer=f'convs.{g_num_layers-3-2*layer}')
                requires_grad(self.generator,
                              False,
                              target_layer=f'to_rgbs.{g_log_size-3-layer}')
            # D
            for layer in range(self.freezeD):
                requires_grad(self.discriminator,
                              True,
                              target_layer=f'convs.{d_log_size-2-layer}')
            requires_grad(self.discriminator, True,
                          target_layer='final_')  # final_conv, final_linear

        elif self.freezeG > 0:
            # G
            for layer in range(self.freezeG):
                requires_grad(self.generator,
                              False,
                              target_layer=f'convs.{g_num_layers-2-2*layer}')
                requires_grad(self.generator,
                              False,
                              target_layer=f'convs.{g_num_layers-3-2*layer}')
                requires_grad(self.generator,
                              False,
                              target_layer=f'to_rgbs.{g_log_size-3-layer}')
            # D
            requires_grad(self.discriminator, True)

        elif self.freezeD > 0:
            # G
            requires_grad(self.generator, False)
            # D
            for layer in range(self.freezeD):
                requires_grad(self.discriminator,
                              True,
                              target_layer=f'convs.{d_log_size-2-layer}')
            requires_grad(self.discriminator, True,
                          target_layer='final_')  # final_conv, final_linear

        else:
            # G
            requires_grad(self.generator, False)
            # D
            requires_grad(self.discriminator, True)

    def freeze_before_train_g(self):
        # Freeze !!!
        requires_grad(self.generator, False)
        requires_grad(self.discriminator, False)

        # obtain some params
        g_log_size = self.generator.module.log_size
        if hasattr(self.generator.module, 'num_layers'):
            g_num_layers = self.generator.module.num_layers
        else:
            g_num_layers = self.generator.module.num_injected_noises
        d_log_size = self.discriminator.module.log_size

        if self.freezeG > 0 and self.freezeD > 0:
            # G
            for layer in range(self.freezeG):
                requires_grad(self.generator,
                              True,
                              target_layer=f'convs.{g_num_layers-2-2*layer}')
                requires_grad(self.generator,
                              True,
                              target_layer=f'convs.{g_num_layers-3-2*layer}')
                requires_grad(self.generator,
                              True,
                              target_layer=f'to_rgbs.{g_log_size-3-layer}')
            # D
            for layer in range(self.freezeD):
                requires_grad(self.discriminator,
                              False,
                              target_layer=f'convs.{d_log_size-2-layer}')
            requires_grad(self.discriminator, False,
                          target_layer='final_')  # final_conv, final_linear

        elif self.freezeG > 0:
            # G
            for layer in range(self.freezeG):
                requires_grad(self.generator,
                              True,
                              target_layer=f'convs.{g_num_layers-2-2*layer}')
                requires_grad(self.generator,
                              True,
                              target_layer=f'convs.{g_num_layers-3-2*layer}')
                requires_grad(self.generator,
                              True,
                              target_layer=f'to_rgbs.{g_log_size-3-layer}')
            # D
            requires_grad(self.discriminator, False)

        elif self.freezeD > 0:
            # G
            requires_grad(self.generator, True)
            # D
            for layer in range(self.freezeD):
                requires_grad(self.discriminator,
                              False,
                              target_layer=f'convs.{d_log_size-2-layer}')
            requires_grad(self.discriminator, False,
                          target_layer='final_')  # final_conv, final_linear

        else:
            # G
            requires_grad(self.generator, True)
            # D
            requires_grad(self.discriminator, False)

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        """Train step function.

        This function implements the standard training iteration for
        asynchronous adversarial training. Namely, in each iteration, we first
        update discriminator and then compute loss for generator with the newly
        updated discriminator.

        As for distributed training, we use the ``reducer`` from ddp to
        synchronize the necessary params in current computational graph.

        Args:
            data_batch (dict): Input data from dataloader.
            optimizer (dict): Dict contains optimizer for generator and
                discriminator.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            loss_scaler (:obj:`torch.cuda.amp.GradScaler` | None, optional):
                The loss/gradient scaler used for auto mixed-precision
                training. Defaults to ``None``.
            use_apex_amp (bool, optional). Whether to use apex.amp. Defaults to
                ``False``.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Contains 'log_vars', 'num_samples', and 'results'.
        """
        # get data from data_batch
        real_imgs = data_batch[self.real_img_key]
        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        batch_size = real_imgs.shape[0]

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # disc training
        self.freeze_before_train_d()
        optimizer['discriminator'].zero_grad()
        # TODO: add noise sampler to customize noise sampling
        with torch.no_grad():
            if self.sample_space == 'zplus':
                latents = self.latent_generator(batch_size)
                fake_imgs = self.generator(latents, input_is_latent=True)
            else:
                out_dict = self.generator(None,
                                          num_batches=batch_size,
                                          return_latents=True)
                latents = [out_dict['latent']]
                fake_imgs = out_dict['fake_img']

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)
        # get data dict to compute losses for disc
        data_dict_ = dict(gen=self.generator,
                          disc=self.discriminator,
                          disc_pred_fake=disc_pred_fake,
                          disc_pred_real=disc_pred_real,
                          fake_imgs=fake_imgs,
                          real_imgs=real_imgs,
                          iteration=curr_iter,
                          batch_size=batch_size,
                          loss_scaler=loss_scaler)

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(loss_disc,
                                optimizer['discriminator'],
                                loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_disc.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['discriminator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['discriminator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['discriminator'].step()

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(fake_imgs=fake_imgs.cpu(),
                           real_imgs=real_imgs.cpu())
            outputs = dict(log_vars=log_vars_disc,
                           num_samples=batch_size,
                           results=results)
            if hasattr(self, 'iteration'):
                self.iteration += 1
            return outputs

        # generator training

        self.freeze_before_train_g()
        optimizer['generator'].zero_grad()

        # TODO: add noise sampler to customize noise sampling
        if self.sample_space == 'zplus':
            latents = self.latent_generator(batch_size)
            fake_imgs = self.generator(latents, input_is_latent=True)
        else:
            out_dict = self.generator(None,
                                      num_batches=batch_size,
                                      return_latents=True)
            latents = [out_dict['latent']]
            fake_imgs = out_dict['fake_img']

        disc_pred_fake_g = self.discriminator(fake_imgs)

        data_dict_ = dict(gen=self.generator,
                          disc=self.discriminator,
                          gen_source=self.source_generator,
                          fake_imgs=fake_imgs,
                          disc_pred_fake_g=disc_pred_fake_g,
                          iteration=curr_iter,
                          batch_size=batch_size,
                          loss_scaler=loss_scaler,
                          latents=latents)

        loss_gen, log_vars_g, source_results = self._get_gen_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))

        if loss_scaler:
            loss_scaler.scale(loss_gen).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(loss_gen, optimizer['generator'],
                                loss_id=1) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_gen.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['generator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['generator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['generator'].step()

        # update ada p
        if hasattr(self.discriminator.module,
                   'with_ada') and self.discriminator.module.with_ada:
            self.discriminator.module.ada_aug.log_buffer[0] += 1
            self.discriminator.module.ada_aug.log_buffer[
                1] += disc_pred_real.sign()
            self.discriminator.module.ada_aug.update(iteration=curr_iter,
                                                     num_batches=batch_size)
            log_vars_disc['ada_prob'] = (
                self.discriminator.module.ada_aug.aug_pipeline.p.data)

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)

        results = dict(fake_imgs=fake_imgs.cpu(),
                       real_imgs=real_imgs.cpu(),
                       src_g_imgs=source_results.cpu(),
                       src_g_imgs_bgr=source_results[:, [2, 1, 0], ...].cpu())
        outputs = dict(log_vars=log_vars,
                       num_samples=batch_size,
                       results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs
