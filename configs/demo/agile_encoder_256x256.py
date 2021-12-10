model = dict(type='PSPEncoderDecoder',
             encoder=dict(type='VAEStyleEncoder', num_layers=50),
             decoder=dict(type='SwapStyleGANv2Generator',
                          out_size=256,
                          style_channels=512,
                          num_mlps=8),
             pool_size=(256, 256),
             id_lambda=0.1,
             lpips_lambda=0.1,
             kl_loss=None,
             train_cfg=None,
             test_cfg=None)
train_cfg = None
test_cfg = None
