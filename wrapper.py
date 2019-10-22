#!/bin/python
# Author: GMFTBY
# Time: 2019.10.21

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        raise Exception(f'[!] No saved model of {dataset}')


def init_model(dataset, epoch=20):
    # init the vocab
    with open('./data/vocab.pkl', 'rb') as f:
        w2idx, idx2w = pickle.load(f)
        
    net = NNLM(len(w2idx))
    net.cuda()
    
    # load the weights
    load_best_model(dataset, net, 0, epoch)
    
    return net, w2idx, idx2w


class tools:
    
    '''
    1. P(t_i|T) = P(t_i|t_0,t_1,...,t_{i-1}) = \Pi_{i=0}^[N_t] P(t_i)
    2. MI = log \frac{P(T|S)}{P(T)}
    '''
    
    def __init__(self, dataset, epoch=20):
        print('[!] make sure the bert-as-service is running')
        self.net, self.w2idx, self.idx2w = init_model(dataset, epoch=epoch)
        self.bc = BertClient()
        
    def str2tensor(self, s):
        # string must be tokenized
        # append <sos> and <eos>, with shape [seq]
        s = torch.tensor([self.w2idx['<sos>']] + [self.w2idx.get(i, self.w2idx['<unk>']) for i in s.strip().split()] + [self.w2idx['<eos>']], dtype=torch.long)
        if torch.cuda.is_available():
            s = s.cuda()
        return s
    
    def str2bert(self, s):
        print('[!] make sure the bert-as-service is running')
        s = torch.from_numpy(self.bc.encode([s]))    # [768]
        if torch.cuda.is_available():
            s = s.cuda()
        return s
    
    def tensor2length(self, s):
        # return the lengths of the tensor s [seq, batch]
        lengths, pad = [], self.w2idx['<pad>']
        for i in range(s.size(1)):
            i = s[:, i]
            lengths.append((i != pad).sum().item())
        if torch.cuda.is_available():
            lengths = torch.tensor(lengths, dtype=torch.long).cuda()
        return lengths
    
    def get_possibility(self, t, hidden=None):
        # t is the tokenized string without the source s
        t = self.str2tensor(t).unsqueeze(1)     # [seq, 1]
        l = self.tensor2length(t)
        p = self.net(t, l, hidden=hidden).squeeze(1)    # [seq, vocab_size] without the softmax
        p = F.softmax(p, dim=1)     # [seq, vocab_size]
        return p
    
    def get_next_word_possibility(self, t, word):
        # t is the tokenized string without the source s
        wordidx = self.w2idx.get(word, self.w2idx['<unk>'])
        p = self.get_possibility(t)[-2]    # [vocab_size]
        p = F.softmax(p)     # [vocab_size]
        return p[wordidx].cpu().item()
    
    def generate_next_word(self, t):
        p = self.get_possibility(t)[-2]    # [vocab_size]
        p = F.softmax(p)     # [vocab_size]
        return self.idx2w[torch.max(p, 0)[1].cpu().item()]
    
    def get_p_t(self, t, s=None):
        # get the possibility of the target sequence w/o source's limitation
        # FORMULATION 1
        wordsidx = [self.w2idx['<sos>']] + [self.w2idx.get(i, self.w2idx['<unk>']) for i in t.strip().split()] + [self.w2idx['<eos>']]
        wordsidx = wordsidx[1:]    # ignore the <sos>
        if s:
            s = self.str2bert(s).reshape(1, 1, -1)     # [1, 1, 768]
            p = self.get_possibility(t, hidden=s)
        else:
            p = self.get_possibility(t)    # [seq, vocab_size]
        p_t = [i[wordsidx[idx]].cpu().item() for idx, i in enumerate(p[:-2])]
        return np.prod(p_t)
  
    def get_sentence_MI(self, t, s):
        # s, t are the tokenized string
        p_t = self.get_p_t(t)
        p_t_s = self.get_p_t(t, s=s)
        return np.log(p_t_s / p_t)
    

if __name__ == "__main__":
    tool = tools('xiaohuangji')
    ipdb.set_trace()
    tool.get_sentence_MI('你 觉得 今天 天气 怎么样', '今天 天气')