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

    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--img_dir', required=True, type=str)
    parser.add_argument('--mask_dir', required=True, type=str)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)

    return parser

def save_images(X, pred, args, e):
    X, pred = X.cpu().data.numpy(), pred.cpu().data.numpy()

    scipy.misc.imsave('exps/' + args.name + '/results/' + str(e) + '_X.jpg', np.transpose(X, (1, 2, 0)))
    scipy.misc.imsave('exps/' + args.name + '/results/' + str(e) + '_pred.jpg', np.argmax(pred, axis=0))

def graph_loss(losses, name):
    plt.figure()
    plt.plot(list(range(len(losses))), losses)
    plt.savefig('exps/' + name + '/loss.png')
    plt.close()

def train(args):
    model = UNet(in_channels=args.in_channels, n_classes=args.num_classes, depth=args.depth, padding=True, up_mode='upsample').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = get_loader(args.img_dir, args.mask_dir, args.batch_size)

    losses = []
    for e in range(args.num_epochs):
        total_loss = 0
        for i, (X, y) in enumerate(loader):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device).reshape(args.batch_size, -1).long()  # [N, in_channels, H, W], [N, HW]

            pred = model(X)  # [N, num_classes, H, W]
            loss = F.cross_entropy(pred.reshape(args.batch_size, args.num_classes, -1), y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            sys.stdout.write('\rEpoch %i: %i/%i' % (e+1, i+1, len(loader)))

        torch.save(model.state_dict(), 'exps/' + args.name + '/checkpoints/' + str(e))
        save_images(X[0], pred[0], args, e)

        losses.append(total_loss / len(loader))
        graph_loss(losses, args.name)

        sys.stdout.write('\rEpoch %i\n' % (e+1))

def prepare_dir(args):
    os.makedirs('exps/' + args.name)
    os.makedirs('exps/' + args.name + '/results')
    os.makedirs('exps/' + args.name + '/checkpoints')
    
    args_dict = vars(args)
    with open('exps/' + args.name + '/args.csv', 'w') as f:
        w = csv.DictWriter(f, args_dict.keys())
        w.writeheader()
        w.writerow(args_dict)

if __name__ == '__main__':
    args = get_parser().parse_args()
    prepare_dir(args)
    train(args)
