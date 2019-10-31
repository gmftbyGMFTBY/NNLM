#!/bin/bash
# Author: GMFTBY
# Time 2019.10.21

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import random
import numpy as np
import argparse
import math, os
import pickle
from tqdm import tqdm
import ipdb

from NNLM import NNLM
from data_loader import *


def train(net, train_iter, optimizer, criterion, grad_clip):
    net.train()
    total_loss, batch_num = 0.0, 0
    
    pbar = tqdm(train_iter)
    
    for idx, batch in enumerate(pbar):
        w2idx, idx2w, batch, lengths, sbatch = batch
        vocab_size = len(w2idx)
        
        optimizer.zero_grad()
        # batch: [seq, batch], output: [seq, batch, vocab_size]
        output = net(batch, lengths, hidden=sbatch)
        # ignore the <eos>
        loss = criterion(output[:-2].view(-1, vocab_size),
                         batch[1:-1].contiguous().view(-1))
        
        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
        batch_num += 1
        
        pbar.set_description(f'training loss: {round(loss.item(), 4)}, max_seqlen: {batch.size(0)}')
    
    pbar.close()
        
    return round(total_loss / batch_num, 4)


def main(vocabp, datapath, batch_size, epoch, lr=1e-3, grad_clip=5, mode='train'):
    # init the vocab
    w2idx, idx2w = load_pickle(vocabp)
    dataset = load_pickle(datapath)
    
    # init the model
    net = NNLM(len(w2idx))
    net.cuda()
    print(f'[!] net:')
    print(net)
    
    if mode == 'pretrained':
        print('[!] load pretrained weighted')
        load_best_model('pretrained', net, 0, 20)
    
    # init the criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=w2idx['<pad>'])
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    pbar = tqdm(range(1, epoch + 1))
    best_loss = None
    for epoch in pbar:
        # load the dataset
        train_iter = get_batch(batch_size, w2idx, idx2w, dataset)
        loss = train(net, train_iter, optimizer, criterion, grad_clip)
        
        if best_loss is None or loss < best_loss:
            best_loss = loss
            patience = 0
        else:
            patience += 1
        
        # save all the checkpoints
        state = {'net': net.state_dict(), 'epoch': epoch}
        torch.save(state, f'./ckpt/{args.dataset}/epoch_{epoch}.pt')
        
        pbar.set_description(f'Epoch: {epoch}, loss: {loss}, ppl: {round(math.exp(loss), 4)}, patience: {patience}/{args.patience}')

        if patience > args.patience:
            print(f'Early Stop {args.patience} at epoch {epoch}')
            break

    pbar.close()
    print('[!] Done')
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NNLM Train script')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='xiaohuangji')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--mode', type=str, default='train')
    
    args = parser.parse_args()
    
    vocabpath = f'./data/vocab.pkl'
    datapath = f'./data/{args.dataset}/data.pkl'
    
    main(vocabpath, datapath, args.batch_size, args.epoch, 
         lr=args.lr, grad_clip=args.grad_clip, mode=args.mode)