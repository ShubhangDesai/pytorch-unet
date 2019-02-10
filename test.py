import torch
import torch.nn.functional as F

from unet import UNet
from data import *

import argparse, os, sys, csv, scipy.misc
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--img_dir', required=True, type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--depth', default=4, type=int)

    return parser

def save_images(X, pred, args, i):
    X, pred = X.cpu().data.numpy(), pred.cpu().data.numpy()

    for j in range(len(X)):
        n = args.batch_size*i+j
        scipy.misc.imsave('inference/' + args.name + '/' + str(n) + '_X.jpg', np.transpose(X[j], (1, 2, 0)))
        scipy.misc.imsave('inference/' + args.name + '/' + str(n) + '_pred.jpg', np.argmax(pred[j], axis=0))

def test(args):
    model = UNet(in_channels=args.in_channels, n_classes=args.num_classes, depth=args.depth, padding=True, up_mode='upsample').to(device)
    model.load_state_dict(torch.load(args.model_path))

    loader = get_test_loader(args.img_dir, args.batch_size)
    for i, (X, _) in enumerate(loader):
        pred = model(X.to(device)) 
        save_images(X, pred, args, i)

        sys.stdout.write('%i/%i' % (i+1, len(loader)))
        break


def prepare_dir(args):
    os.makedirs('inference/' + args.name)

if __name__ == '__main__':
    args = get_parser().parse_args()
    args.name = args.img_dir.split('/')[-1]

    prepare_dir(args)
    test(args)
