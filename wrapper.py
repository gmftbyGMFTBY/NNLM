#!/bin/python
# Author: GMFTBY
# Time: 2019.10.21

import torch
import torch.nn as nn
import torch.nn.functional as F
from NNLM import NNLM
from data_loader import *
from bert_serving.client import BertClient
import pickle
import ipdb
import os


def load_best_model(dataset, net, min_threshold, max_threshold):
    path = f'./ckpt/{dataset}/'
    best_loss, best_file, best_epoch = np.inf, None, -1

    for file in os.listdir(path):
        _, epoch = file.split('_')
        epoch = int(epoch.split('.')[0])

        if min_threshold <= epoch <= max_threshold and epoch > best_epoch:
            best_file = file
            best_epoch = epoch

    if best_file:
        file_path = path + best_file
        print(f'[!] Load the model from {file_path}, threshold ({min_threshold}, {max_threshold})')
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception('[!] No saved model')


def init_model(dataset):
    # init the vocab
    with open('vocab.pkl', 'rb') as f:
        w2idx, idx2w = pickle.load(f)
        
    net = NNLM(len(w2idx))
    net.cuda()
    
    # load the weights
    load_best_model(dataset, net, 0, 20)
    
    return net, w2idx, idx2w


class MI:
    
    '''
    MI = log \frac{P(T|S)}{P(T)}
    '''
    
    def __init__(self, dataset):
        print('[!] make sure the bert-as-service is running')
        self.net, self.w2idx, self.idx2w = init_model(dataset)
        self.bc = BertClient()
        
    def get_MI(self, s, t):
        # s, t are the string
        s = torch.from_numpy(self.bc.encode(s)).unsqueeze(0)    # [1, 768]
        t = torch.tensor([self.w2idx[i] for i in t.strip().split()], dtype=torch.float)
        lengths = torch.tensor([len(t)], dtype=torch.long)
        if torch.cuda.is_available():
            s = s.cuda()
            t = t.cuda()
            lengths = lengths.cuda(s)
            
        # p_t/p_t_s: [seq, vocab_size]
        p_t = net(t, lengths).squeeze(1)[-1].item()
        p_t_s = net(t, lengths, hidden=s).squeeze(1)[-1].item()
        
        return p_t - p_t_s
    

if __name__ == "__main__":
    mi = MI('xiaohuangji')
    ipdb.set_trace()
    mi.get_MI()