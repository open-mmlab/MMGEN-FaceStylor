import argparse
import os
import sys

import torch

# yapf: enable
# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa
import agilegan # isort:skip  # noqa

from mmgen.apis import init_model, sample_uncoditional_model  # isort:skip  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description='Blend decoders\' weights of'
                                     ' EncoderDecoders')
    parser.add_argument('modelA', help='Encoder config file path')
    parser.add_argument('modelB', help='Transfer config file path')
    parser.add_argument(
        '--swap-layer', type=int, default=4, help='swap layer')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/pre-trained/agile_transfer_blended.pth',
        help='path to save image transfer result')
    args = parser.parse_args()
    return args


def swap_weights(base_ckpt, swap_ckpt, start_index, end_index=8):
    for i in range(start_index, end_index):
        for layer in base_ckpt.keys():
            if f'convs.{2*i}' in layer or \
                    f'convs.{2*i + 1}' in layer or f'to_rgbs.{i}' in layer:
                base_ckpt[layer] = swap_ckpt[layer]
    return base_ckpt


def main():
    args = parse_args()
    # init models
    modelA = init_model(args.modelA, checkpoint=None, device=args.device)
    modelB = init_model(args.modelB, checkpoint=None, device=args.device)

    base_ckpt = modelA.decoder.state_dict()
    swap_ckpt = modelB.decoder.state_dict()

    start_index = args.swap_layer - 1
    end_index = modelA.decoder.log_size - 2

    modelA.decoder.load_state_dict(swap_weights(base_ckpt, swap_ckpt,
                                                start_index, end_index))

    torch.save(modelA.state_dict(), args.save_path)
    torch.save(modelA.decoder.state_dict(), 'work_dirs/pre-trained/'
               'swap_toonify.pth')


if __name__ == '__main__':
    main()
