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
    
    def __init__(self, dataset, epoch=20, maxlen=20):
        self.net, self.w2idx, self.idx2w = init_model(dataset, epoch=epoch)
        self.maxlen = maxlen
        
    def str2tensor(self, s):
        # string must be tokenized
        # append <sos> and <eos>, with shape [seq]
        # s: [b]
        # return: [maxlen, batch]
        maxlen = min(self.maxlen, max([len(i) for i in s]))
        d = []
        for line in s:
            if len(line) > maxlen:
                pause = [self.w2idx['<sos>']] + [self.w2idx.get(i, self.w2idx['<unk>']) for i in list(line.strip())][-maxlen:] + [self.w2idx['<eos>']]
            else:
                pause = [self.w2idx['<sos>']] + [self.w2idx.get(i, self.w2idx['<unk>']) for i in list(line.strip())] + [self.w2idx['<eos>']] + [self.w2idx['<pad>']] * (maxlen - len(line.strip()))
            d.append(torch.tensor(pause, dtype=torch.long))
        d = torch.stack(d).transpose(0, 1)
            
        if torch.cuda.is_available():
            d = d.cuda()
        return d
    
    def tensor2length(self, s):
        # return the lengths of the tensor s [seq, batch]
        # return: [batch]
        lengths, pad = [], self.w2idx['<pad>']
        for i in range(s.size(1)):
            i = s[:, i]
            lengths.append((i != pad).sum().item())
        if torch.cuda.is_available():
            lengths = torch.tensor(lengths, dtype=torch.long).cuda()
        return lengths
    
    def get_possibility(self, t):
        # t is the tokenized string without the source s, [b]
        t = self.str2tensor(t)     # [seq, batch]
        l = self.tensor2length(t)    # [batch]
        p = self.net(t, l)    # [seq, batch, vocab_size]
        # ignore the unk
        p[:, :, self.w2idx['<unk>']] = -np.inf
        p = F.softmax(p, dim=2)     # [seq, batch, vocab_size]
        return p, l
    
    def get_next_word_possibility(self, t, word):
        # t is the tokenized string without the source s
        wordidx = self.w2idx.get(word, self.w2idx['<unk>'])
        p = self.get_possibility(t)[-2]    # [vocab_size]
        # ignore the unk
        p[self.w2idx['<unk>']] = -np.inf
        p = F.softmax(p)     # [vocab_size]
        return p[wordidx].cpu().item()
    
    def generate_next_word(self, t):
        p = self.get_possibility(t)[-2]    # [vocab_size]
        # ignore the unk
        p[self.w2idx['<unk>']] = -np.inf
        p = F.softmax(p)     # [vocab_size]
        return self.idx2w[torch.max(p, 0)[1].cpu().item()]
    
    def get_p_t(self, t, s=None):
        # get the possibility of the target sequence w/o source's limitation
        # FORMULATION 1
        # t: [b], s: [b]
        # return [b]
        if s:
            t = [j + i for i, j in zip(t, s)]
            pp = [len(i) for i in s]
        else:
            pp = [0 for i in t]
            
        wordsidx = []
        for line in t:
            pause = [self.w2idx.get(i, self.w2idx['<unk>']) for i in list(line.strip())] + [self.w2idx['<eos>']]
            wordsidx.append(pause)    # ignore the pause
        p, l = self.get_possibility(t)    # [seq, batch, vocab_size], [batch]
        p = p.permute(1, 0, 2)    # [batch, seq, vocab_size]
        result = []    # [b]
        for idx_, (i, length, k) in enumerate(zip(p, l, pp)):
            a = [w[wordsidx[idx_][idx]].cpu().item() for idx, w in enumerate(i[k:-2-(len(i)-length)])]
            result.append(np.prod(a))
        return result
    
    def get_SRF(self, s, t):
        # :param s: [b]
        # :param t: [b]
        pt = np.log(self.get_p_t(t))          # [b]
        pst = np.log(self.get_p_t(s, s=t))    # [b]
        pts = np.log(self.get_p_t(t, s=s))    # [b]
        
        return 2 * pst / (pt + pts)    # [b]
    

if __name__ == "__main__":
    tool = tools('xiaohuangji')
    ipdb.set_trace()
    tool.get_sentence_MI('你 觉得 今天 天气 怎么样', '今天 天气')