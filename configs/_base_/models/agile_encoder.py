encoder_ckpt_path = 'work_dirs/pre-trained/model_ir_se50.pth'
stylegan_weights = 'work_dirs/pre-trained/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth'  # noqa

model = dict(type='PSPEncoderDecoder',
             encoder=dict(type='VAEStyleEncoder',
                          num_layers=50,
                          pretrained=dict(ckpt_path=encoder_ckpt_path,
                                          strict=False)),
             decoder=dict(type='SwapStyleGANv2Generator',
                          out_size=1024,
                          style_channels=512,
                          num_mlps=8,
                          pretrained=dict(ckpt_path=stylegan_weights,
                                          prefix='generator_ema')),
             pool_size=(256, 256),
             id_lambda=0.1,
             lpips_lambda=0.8,
             id_ckpt=encoder_ckpt_path,
             kl_loss=None,
             train_cfg=None,
             test_cfg=None)
train_cfg = None
test_cfg = None
