import argparse
import numpy as np
import torch
import torch.nn as nn

from models import GCN
from dataloader import DataLoader
from utils import train_model

def get_args():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--cuda', action='store_true', help='declare if running on GPU')
    parser.add_argument('--emb_size', type=int, default=64, help='dimension of the latent space')
    parser.add_argument('--hidden_size', type=int, default=256, help='size of the hidden layer')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--seed', type=int, default=7, help='random seed, -1 for not fixing it')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--dataset', type=str, default='cora')
    args = parser.parse_args()
    return args

def main(args):
    # config device
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    # fix random seeds
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    dl = DataLoader(args)

    model = GCN(dl.G, dl.features.size(1), args.hidden_size, args.emb_size, dl.n_class, args.dropout)
    model.to(args.device)
    model = train_model(args, dl, model)

if __name__ == "__main__":
    args = get_args()
    main(args)
