import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
import numpy as np

def train_model(args, dl, model):
    device = args.device
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    # data
    features = dl.features.to(device)
    labels = dl.labels.to(device)
    # normalization (D^{1/2})
    degs = dl.G.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.to(device)
    norm = norm.unsqueeze(1)
    loss = nn.CrossEntropyLoss()

    best_vali_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        logits = model(features, norm)
        # losses
        l = loss(logits[dl.train_nid], labels[dl.train_nid])
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # validate (without dropout)
        model.eval()
        logits_eval = model(features, norm)
        vali_results = eval_node_cls(logits_eval[dl.val_nid], labels[dl.val_nid])
        print('Epoch [{:2}/{}]: loss: {:.4f}, vali acc: {:.4f}'.format(epoch+1, args.epochs, l.item(), vali_results['acc']))
        if vali_results['acc'] > best_vali_acc:
            best_vali_acc = vali_results['acc']
            test_results = eval_node_cls(logits_eval[dl.test_nid], labels[dl.test_nid])
            print('                 test acc: {:.4f}'.format(test_results['acc']))
    print('Final test results: acc: {:.4f}'.format(test_results['acc']))
    return model

def eval_node_cls(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels)
    acc = correct.item() / len(labels)
    results = {'acc': acc}
    return results

