import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from torch.optim import Adam

import os
import pandas as pd
import argparse

class MLP(nn.Module):
    def __init__(self, input, output, n_layers, n_hidden):
        super(MLP, self).__init__()
        layers = [nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU()) for _ in range(n_hidden)] # hidden layers
        layer_first = nn.Sequential(
            nn.Linear(input, n_hidden),
            nn.ReLU(inplace=True)
        )
        layers.insert(0, layer_first) # initial layer
        layers.append(nn.Linear(n_hidden, output)) # final layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Coupling(nn.Module):
    def __init__(self, n_layers, n_hidden, D, d):
        super(Coupling, self).__init__()
        self.scale_block = MLP(d, D - d, n_layers, n_hidden)
        self.translation_block = MLP(d, D - d, n_layers, n_hidden)
        self.D = D
        self.d = d
        # permute outputs after applying blocks, and inverse
        # for inverse map
        self.perm = torch.randperm(D)
        self.inv_perm = torch.argsort(self.perm)

    def forward(self, x):
        # compute scale and translation for end partition
        scale = torch.exp(self.scale_block(x[:, :self.d]))
        translation = self.translation_block(x[:, :self.d])
        # logdet determinant as per (6) in RealNVP paper
        log_det = torch.sum(scale, axis=1)
        # apply scale and translation and combine
        y_start = x[:, :self.d]
        y_end = scale * x[:, self.d:] + translation
        y = torch.cat((y_start, y_end), 1)
        # permute before outputting
        return y[:, self.perm], log_det

    def inverse(self, y):
        y_perm = y[:, self.inv_perm]
        translation = self.translation_block(y_perm[:, :self.d])
        scale = self.scale_block(y_perm[:, :d])
        x_start = y_perm[:, :d]
        x_end = (y_perm[:, d:] - translation) * torch.exp(-scale)
        return torch.cat((x_start, x_end), 1)

class RealNVP(nn.Module):
    def __init__(self, n_couplings, n_layers, n_hidden, D, d):
        super(RealNVP, self).__init__()
        couplings = [Coupling(n_layers, n_hidden, D, d) for _ in range(n_couplings)]
        self.couplings = nn.ModuleList(couplings)
        self.n_couplings = n_couplings

    def forward(self, x):
        logdet = torch.zeros(x.shape[0], self.n_couplings)
        for i, coupling in enumerate(self.couplings):
            x, ld = coupling(x)
            logdet[:, i] = ld
        return x, logdet

    def inverse(self, y):
        for coupling in reversed(self.couplings):
            y = coupling.inverse(y)
        return y 

    def NLL(self, x):
        y, logdet = self.forward(x)
        ld = torch.logsumexp(torch.abs(logdet), dim=1)
        N = MultivariateNormal(torch.zeros(x.shape[1]), covariance_matrix=torch.eye(x.shape[1]))
        log_prob = N.log_prob(y) + ld     
        return -log_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--num-couplings', type=int, default=5,
                        help="coupling layers")
    parser.add_argument('-nl', '--num-layers', type=int, default=3,
                        help="number of hidden layers per coupling layer")
    parser.add_argument('-nh', '--num-hidden', type=int, default=256,
                        help="number of hidden units per hidden layer in coupling layer")
    parser.add_argument('-d', '--partition-index', type=int, default=8,
                        help="index to partition input to do scale/translation parameterization. \
                                Must be less than input dimension.")
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help="batch size used in training")
    parser.add_argument('-e', '--num-epochs', type=int, default=100,
                        help="number of training epochs")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5,
                        help="learning rate for optimizing")
    parser.add_argument('-wd', '--weight-decay', type=float, default=5e-5,
                        help="weight decay for optimizer")
    parser.add_argument('--file', type=str, default="FA2")
    args = parser.parse_args()

    # import data, remove headers
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv("{}/data/{}.csv".format(dir_path, args.file))
    data = data.to_numpy(dtype='float32')
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    D = data.shape[1]

    model = RealNVP(args.num_couplings, args.num_layers, args.num_hidden, D, args.partition_index)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # for each epoch, batch samples and fit flow
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        loss = 0
        for batch in loader:
            loss += torch.sum(model.NLL(batch))
        # eval gradients and take training step
        loss.backward()
        optimizer.step()
        print("epoch {} train loss {}".format(epoch, loss))