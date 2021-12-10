model = dict(type='PSPTransfer',
             src_generator=dict(type='StyleGANv2Generator',
                                out_size=1024,
                                style_channels=512,
                                num_mlps=8),
             generator=dict(type='SwapStyleGANv2Generator',
                            out_size=1024,
                            style_channels=512,
                            num_mlps=8),
             discriminator=dict(type='ADAStyleGAN2Discriminator',
                                in_size=1024),
             gan_loss=dict(type='GANLoss', gan_type='hinge'),
             lpips_lambda=0.1)

train_cfg = dict(use_ema=True)
test_cfg = None
