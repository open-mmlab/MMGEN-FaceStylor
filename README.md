# MMGEN-FaceStylor
<a href="http://app.openmmlab.com/facestylor/"><img src="https://img.shields.io/badge/Play%20Now!-Demo-orange" height=22.5></a>
<a href="https://colab.research.google.com/drive/12ECMTWtP-MyZn3HetiFJ6udXBIX_C1Gb?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>

English | [简体中文](https://github.com/open-mmlab/MMGEN-FaceStylor/blob/master/README_CN.md)

## Introduction
This repo is an efficient toolkit for Face Stylization based on the paper "AgileGAN: Stylizing Portraits by Inversion-Consistent Transfer Learning". We note that since the training code of AgileGAN is not released yet, this repo merely adopts the pipeline from AgileGAN and combines other helpful practices in this literature.

This project is based on [MMCV](https://github.com/open-mmlab/mmcv) and [MMGEN](https://github.com/open-mmlab/mmgeneration), star and fork is welcomed 🤗!

<div align="center">
  <b> Results from FaceStylor trained by MMGEN</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144411672-77fe6bcd-8fe4-40e6-8e7b-903cbac9ed58.gif">
</div>

## Requirements
- CUDA 10.0 / CUDA 10.1
- Python 3
- PyTorch >= 1.6.0
- MMCV-Full >= 1.3.15
- MMGeneration >= 0.3.0

## Setup
### Step-1: Create an Environment
First, we should build a conda virtual environment and activate it.
```bash
conda create -n facestylor python=3.7 -y
conda activate facestylor
```
Suppose you have installed CUDA 10.1, you need to install the prebuilt PyTorch with CUDA 10.1.
```bash
conda install pytorch=1.6.0 cudatoolkit=10.1 torchvision -c pytorch
```

### Step-2: Install MMCV and MMGEN
We can run the following command to install MMCV.
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```
Of course, you can also refer to the MMCV [Docs](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) to install it.

Next, we should install MMGEN containing the basic generative models that will be used in this project.
```bash
# Clone the MMGeneration repository.
git clone https://github.com/open-mmlab/mmgeneration.git
cd mmgeneration
# Install build requirements and then install MMGeneration.
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
cd ..
```
### Step-3: Clone repo and prepare the data and weights
<!-- I'm not sure what the git address is -->
Now, we need to clone this repo and install dependencies.
```bash
git clone https://github.com/open-mmlab/MMGEN-FaceStylor.git
cd MMGEN-FaceStylor
pip install -r requirements.txt
```

For convenience, we suggest that you make these folders under `MMGEN-FaceStylor`.
```bash
mkdir data
mkdir work_dirs
mkdir work_dirs/experiments
mkdir work_dirs/pre-trained
```
For testing and training, you need to download some necessary [data](https://drive.google.com/drive/folders/1sksjD4awYwSAgibix83hVtx1sm4KOekm) provided by [AgileGAN](https://github.com/flyingbread-elon/AgileGAN) and put them under `data` folder. Or just run this:
```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AavRxpZJYeCrAOghgtthYqVB06y9QJd3' -O data/shape_predictor_68_face_landmarks.dat
```
Then, you can put or create the soft-link for your data under `data` folder, and store your experiments under `work_dirs/experiments`.


We also provide some pre-trained weights.

| Pre-trained Weights                  |
|---------------------------------------|
| [FFHQ-1024 StyleGAN2](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth)            |
| [FFHQ-256 StyleGAN2](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth)      |
|[IR-SE50 Model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmgen/agilegan/model_ir_se50.pth)|
| [Encoder for FFHQ-1024 StyleGAN2](https://download.openmmlab.com/mmgen/agilegan/agile_encoder_ffhq1024x1024_lr_1e-4_500kiter_20211201_112111-fb1312dc.pth) |
| [Encoder for FFHQ-256 StyleGAN2](https://download.openmmlab.com/mmgen/agilegan/agile_encoder_celebahq256x256_lr_1e-4_150k_20211104_134520-9cce67da.pth)  |
| [MetFace-Oil 1024 StyleGAN2](https://download.openmmlab.com/mmgen/agilegan/agile_transfer_metfaces-oil1024x1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134350-2b99cb9b.pth)      |
| [MetFace-Sketch 1024 StyleGAN2](https://download.openmmlab.com/mmgen/agilegan/agile_transfer_metfaces-sketch1024x1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134426-081af2a2.pth)   |
| [Toonify 1024 StyleGAN2](https://download.openmmlab.com/mmgen/agilegan/agile_transfer_toonify1024x1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_1600iter_20211104_134449-cb6785b6.pth)          |
|[Cartoon 256](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmgen/agilegan/agile_transfer_photo2cartoon256x256_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_1e-4_800_iter_20211201_140719-062c09fa.pth)|
|[Bitmoji 256](https://download.openmmlab.com/mmgen/agilegan/agile_transfer_bitmoji256x256_z_wolpips_freezeD3_ada_bs4x2_lr_1e-4_iter_1600_20211202_195819-9010a9fe.pth)|
|[Comic 256](https://download.openmmlab.com/mmgen/agilegan/agile_transfer_face2comics256x256_z_wolpips_freezeD3_ada_bs4x2_lr_1e-4_30kiter_best_fid_iter_15000_20211201_111145-4905b63a.pth)|
| More Styles on the Way!             |

## Play with MMGEN-FaceStylor
If you have followed the aforementioned steps, we can start to investigate FaceStylor!
### Quick Try
To quickly try our project, please run the command below
```bash
python demo/quick_try.py demo/src.png --style toonify
```
Then, you can check the result in `work_dirs/demos/agile_result.png`.

- If you want to play with your own photos, you can replace `demo/src.png` with your photo.
- If you want to switch to another style, change `toonify` with other styles. Now, supported styles include `toonify`, `oil`, `sketch`, `bitmoji`, `cartoon`, `comic`.

### Inversion
The inversion task will adopt a source image as input and return the most similar image that can be generated by the generator model.

For inversion, you can directly use `agilegan_demo` like this
```bash
python demo/agilegan_demo.py SOURCE_PATH CONFIG [--ckpt CKPT] [--device DEVICE] [--save-path SAVE_PATH]
```
Here, you should set `SOURCE_PATH` to your image path, `CONFIG` to the config file path, and `CKPT` to checkpoint path.

Take [Celebahq-Encoder](configs/agilegan/agile_encoder_celebahq_lr_1e-4_150k.py) as an example, you need to download the [weights](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmgen/agilegan/agile_encoder_celebahq1024x1024_lr_1e-4_150k_20211104_133124-a7e2fd7f.pth?versionId=CAEQHhiBgMDpiaGo5xciIDgzNTQ4YTQ2OWQ1OTQ0NmM4NWZiZjg2MTk0ZGEzMmFi) to `work_dirs/pre-trained/agile_encoder_celebahq1024x1024_lr_1e-4_150k.pth`, put your test image under `data` run
```bash
python demo/agilegan_demo.py demo/src.png configs/agilegan/agile_encoder_celebahq1024x1024_lr_1e-4_150k.py --ckpt work_dirs/pre-trained/agile_encoder_celebahq_lr_1e-4_150k.pth
```
You will find the result `work_dirs/demos/agile_result.png`.

### Stylization
Since the encoder and decoder of stylization can be trained from different configs, you're supposed to set their ckpts' path in config file.
Take [Metface-oil](configs/demo/agile_transfer_metface-oil1024x1024.py) as an example, you can see the first two lines in config file.
```python
encoder_ckpt_path = xxx
stylegan_weights = xxx
```
You should keep your actual weights path in line with your configs. Then run the same command without specifying `CKPT`.
```bash
python demo/agilegan_demo.py SOURCE_PATH CONFIG [--device DEVICE] [--save-path SAVE_PATH]
```


## Train
Here I will tell you how to fine-tune with your own datasets. With only 100-200 images and less than one hour,
you can train your own StyleGAN2. The only thing you need to do is to copy an
`agile_transfer` config, like this [one](configs/agilegan/agile_transfer_metfaces-oil1024x1024_zplus_lpips0.5_freezeD5_ada_bs4x2_lr_2e-3_1600iter.py). Then modify the `imgs_root` with your actual data root, choose one of the two commands below to train your own model.
```bash
# For distributed training
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS_NUMBER} \
    --work-dir ./work_dirs/experiments/experiments_name \
    [optional arguments]
# For slurm training
bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} \
    [optional arguments]
```

## Training Details
In this part, I will explain some training details, including ADA setting, layer freeze, and losses.
### ADA Setting
To use [adaptive discriminator augmentation](https://github.com/NVlabs/stylegan2-ada-pytorch) in your discriminator, you can use `ADAStyleGAN2Discriminator` as your discriminator, and adjust `ADAAug` setting as follows:
```python
model = dict(
    discriminator=dict(
                 type='ADAStyleGAN2Discriminator',
                 data_aug=dict(type='ADAAug',
                 aug_pipeline=aug_kwargs, # This and below arguments can be set by yourself.
                 update_interval=4,
                 augment_initial_p=0.,
                 ada_target=0.6,
                 ada_kimg=500,
                 use_slow_aug=False)))
```

### Layer Freeze Setting
In transfer learning, it's a routine to freeze some layers in models.
In GAN's literature, freezing the shallow layers of pre-trained generator and discriminator may help training convergence.
[FreezeD](https://github.com/sangwoomo/FreezeD) can be used for small data fine-tuning,
[FreezeG](https://github.com/bryandlee/FreezeG) can be used for pseudo translation.
```python
model = dict(
  freezeD=5, # set to -1 if not need
  freezeG=4 # set to -1 if not need
  )
```

### Losses Setting
In [AgileGAN](https://github.com/GuoxianSong/AgileGAN), to preserve the recognizable identity of the generated image, they introduce a similarity loss at the perceptual level. You can adjust the `lpips_lambda` as follows:
```python
model = dict(lpips_lambda=0.8)
```
Generally speaking, the larger `lpips_lambda` is, the better the recognizable identity can be kept.


## Datasets Link
To make it easier for you to train your own models, here are some links to publicly available datasets.
|Dataset Links|
|------|
|[MetFaces](https://github.com/NVlabs/metfaces-dataset)|
|[AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq)|
|[Toonify](https://mega.nz/file/HslSXS4a#7UBanJTjJqUl_2Z-JmAsreQYiJUKC-8UlZDR0rUsarw)|
|[photo2cartoon](https://www.kaggle.com/arnaud58/photo2cartoon)|
|[selfie2anime](https://www.kaggle.com/arnaud58/selfie2anime)|
|[face2comics v2](https://www.kaggle.com/defileroff/comic-faces-paired-synthetic-v2)|
|[High-Resolution Anime Face](https://www.kaggle.com/subinium/highresolution-anime-face-dataset-512x512)|
|[Bitmoji](https://www.kaggle.com/mostafamozafari/bitmoji-faces)|


## Applications
We also provide `LayerSwap` and `DNI` apps for the trade-off between the structure of the original image and the stylization degree.
To this end, you can adjust some parameters to get your desired result.
### LayerSwap
When [Layer Swapping](https://github.com/justinpinkney/toonify) is applied, the generated images have a higher similarity to the source image than AgileGAN's results.
<div align="center">
  <b> From Left to Right: Input, Layer-Swap with L = 4, 3, 2, xxx Output </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/140281887-b24f6805-90c9-4735-9d02-1b7bc44d288f.png" width="800"/>
</div>

Run this command line to with different `SWAP_LAYER`(1, 2, 3, 4, etc) :
```bash
python demo/quick_try.py demo/src.png --style toonify --swap-layer=SWAP_LAYER
```
and you can discover the result tends to be close to the source image.

We also provide a blending script to create and save the mixed weights.
```bash
python apps/blend_weights.py modelA modelB [--swap-layer SWAP_LAYER] [--show-input SHOW_INPUT] [--device DEVICE] [--save-path SAVE_PATH]
```

Here, `modelA` is the base model, where only the deep layers of its decoder will be replaced with `modelB`'s counterpart.

### DNI

<div align="center">
  <b> Deep Network Interpolation between L4 and AgileGAN output </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/140469139-8de3d1b2-e009-4acd-9754-cab24eaa59a3.png" width="800"/>
</div>

For more precise stylization control, you can try [DNI](https://github.com/xinntao/DNI) with following commands:
```bash
python apps/dni.py source_path modelA modelB [--intervals INTERVALS] [--device DEVICE] [--save-folder SAVE_FOLDER]
```
Here, `modelA` and `modelB` are supposed to be `PSPEncoderDecoder`(configs start with `agile_encoder`) with decoders of different stylization degrees. `INTERVALS` is supposed to be the interpolation numbers.

You can also try [applications](https://github.com/open-mmlab/mmgeneration/tree/master/apps) in MMGEN, like interpolation and SeFA.

### Interpolation

<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/145522062-8a4f1210-694a-42d5-8a12-9de6e844c293.gif">
</div>

Indeed, we have provided an application script to users. You can use apps/interpolate_sample.py with the following commands for unconditional models’ interpolation:
```bash
python apps/interpolate_sample.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--show-mode ${SHOW_MODE}] \
    [--endpoint ${ENDPOINT}] \
    [--interval ${INTERVAL}] \
    [--space ${SPACE}] \
    [--samples-path ${SAMPLES_PATH}] \
    [--batch-size ${BATCH_SIZE}] \
```
For more details, you can read related [Docs](https://mmgeneration.readthedocs.io/en/latest/tutorials/applications.html#interpolation).


## Galary
Toonify
***
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747013-70de4823-c393-40c1-bade-ff50b23c7f0e.png">
</div>

<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747056-d92f0cba-763a-4f8c-a847-c19db79e533b.png">
</div>

<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747096-591fbb78-7df2-4a3f-9b2d-308fd1acea76.png">
</div>

Oil
***
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747167-3ed7f8b3-d6c8-49b4-b1ce-e8b57758fdf4.png">
</div>
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747217-9d6dfa62-7b33-443b-9e98-653c3e8ebf96.png">
</div>
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747254-666a4c82-adcb-4f6c-86c6-ec17e22de398.png">
</div>

Cartoon
***
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747455-0a074884-c94d-43b8-8869-154be45a6fd0.png">
</div>
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747472-53d1a3cd-ae28-46ea-9cad-1cd71cec0d92.png">
</div>
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747474-c4cdb2d2-ab5f-4c98-9356-e0da6dbdc15b.png">
</div>

Comic
***
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747553-19c92568-edfb-4713-8ff3-9a4ae9497cd2.png">
</div>
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747564-37a8ddba-b605-4cb4-a0ad-3115308fc746.png">
</div>
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747748-5c5ea3e1-6eb1-46f9-9760-c8e205c21282.png">
</div>

Bitmoji
***
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747899-7c1633e2-8178-4a0f-888e-9002e0935bdd.png">
</div>
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747943-bcb4201f-f08c-4af6-ae72-6647be4b5ce2.png">
</div>
<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/144747931-ce65fb82-895f-49eb-a172-e1a0c4d1545b.png">
</div>

## Notions and TODOs
- For encoder, I experimented with vae-encoder but found no significant improvement for inversion. I follow the "encoding into z plus space" way as the author does. I will release the vae-encoder version later, but I only offer a vanilla encoder this time.
- For generator, I released vanilla stylegan2-generator, and `attribute-aware generator` will be released in next version.
- For training settings, the parameters have slight difference from the paper. And I also tried `ADA`, `freezeD` and other methods not mentioned in paper.
- More styles will be available in the next version.
- More applications will be available in the next version.
- Further code clean jobs.

## Acknowledgments
Codes reference:
- https://github.com/open-mmlab/mmcv
- https://github.com/open-mmlab/mmgeneration
- https://github.com/GuoxianSong/AgileGAN
- https://github.com/flyingbread-elon/AgileGAN
- https://github.com/eladrich/pixel2style2pixel
- https://github.com/happy-jihye/Cartoon-StyleGAN
- https://github.com/NVlabs/stylegan2-ada-pytorch
- https://github.com/sangwoomo/FreezeD
- https://github.com/bryandlee/FreezeG
- https://github.com/justinpinkney/toonify

Display photos from:
https://unsplash.com/t/people

Web demo powered by:
https://gradio.app/

## License
This project is released under the [Apache 2.0 license](https://github.com/open-mmlab/MMGEN-FaceStylor/blob/master/LICENSE). Some implementation in MMGEN-FaceStylor are with other licenses instead of Apache2.0. Please refer to [LICENSES.md](https://github.com/open-mmlab/MMGEN-FaceStylor/blob/master/LICENSE.md) for the careful check, if you are using our code for commercial matters.
