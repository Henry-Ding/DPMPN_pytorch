import argparse

import torch
import torch.optim as optim

import gre
import data
from model import Reasoner


def train_step(hparams, context, reasoner, batch):
    """ [Param] batch (2D numpy): B x 4
    """
    heads, tails, rels, di = batch.T
    heads = torch.from_numpy(heads).to(device=context['device'])
    rels = torch.from_numpy(rels).to(device=context['device'])
    di = torch.from_numpy(di).to(device=context['device'])

def eval_step():
    pass

def run(hparams, context, graph, dataset):
    reasoner = Reasoner(hparams, context, graph)
    optimizer = optim.Adam(reasoner.parameters(), lr=hparams.learning_rate)

    train_loader = dataset.get_train_loader()
    for epoch in range(1, hparams.n_epochs + 1):
        for train_batch, batch_size in train_loader(hparams.batch_size):
            train_step(reasoner, train_batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', type=str, default='datasets/FB237/graph')
    parser.add_argument('--dataset', default='FB237', choices=['FB237'])
    parser.add_argument('--n_dims', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=6)
    parser.add_argument('--max_attd_from', type=int, default=10)
    parser.add_argument('--max_attd_over', type=int, default=100)
    parser.add_argument('--removed_edges', default='ht_th', choices=['h_t', 'h_t_r_di', 'ht_th', 'ht_th_r'])
    args = parser.parse_args()

    graph = gre.LoadedGraph(args.graph_path, add_reverse=True)
    dataset = getattr(data, args.dataset)(graph)
    print(dataset.name)

    context = dict()
    context['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    context['activated_edges'] = dataset.train

    run(args, context, graph, dataset)
