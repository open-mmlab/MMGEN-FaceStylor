_base_ = [
    '../_base_/models/agile_transfer.py', '../_base_/datasets/ffhq_flip.py',
    '../_base_/default_runtime.py'
]

# define dataset
# you must set `samples_per_gpu`
# `samples_per_gpu` and `imgs_root` need to be set.
imgs_root = 'data/toonify'
data = dict(samples_per_gpu=4,
            workers_per_gpu=4,
            train=dict(dataset=dict(imgs_root=imgs_root)),
            val=dict(imgs_root=imgs_root))

aug_kwargs = {
    'xflip': 1,
    'rotate90': 1,
    'xint': 1,
    'scale': 1,
    'rotate': 1,
    'aniso': 1,
    'xfrac': 1,
    'brightness': 1,
    'contrast': 1,
    'lumaflip': 1,
    'hue': 1,
    'saturation': 1
}

model = dict(
    lpips_lambda=0.5,
    freezeD=5,
    discriminator=dict(
        data_aug=dict(type='ADAAug', aug_pipeline=aug_kwargs, ada_kimg=100)))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=400, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(type='VisualizeUnconditionalSamples',
         output_dir='training_samples',
         interval=100),
    dict(type='ExponentialMovingAverageHook',
         module_keys=('generator_ema', ),
         interval=1,
         start_iter=1,
         interp_cfg=dict(momentum=0.999),
         priority='VERY_HIGH')
]
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
# 30000 images in celeba-hq
total_iters = 1600

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = True

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)
