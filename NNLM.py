#!/bin/bash
# Author: GMFTBY
# Time 2019.10.21

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

'''
Inspired by NNLM (bengio)
'''

class NNLM(nn.Module):
    
    '''
    Calculate the propability of the next word
    P(T|S): hidden state is the bert embedding of the S.
    P(T): hidden state is zero vector.
    '''
    
    def __init__(self, vocab_size, nh=1000, hidden=768):
        super(NNLM, self).__init__()
        
        self.e = nn.Embedding(vocab_size, nh)
        self.m = nn.GRU(nh, nh)
        self.l = nn.Linear(nh, vocab_size)
        self.trans_h = nn.Linear(hidden, nh)
        
        self.init_weight()
            
    def init_weight(self):
        # orthogonal init
        init.orthogonal_(self.m.weight_hh_l0)
        init.orthogonal_(self.m.weight_ih_l0)
        self.m.bias_ih_l0.data.fill_(0.0)
        self.m.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, inpt, lengths, hidden=None):
        # inpt: [seq, batch], lengths: [batch], hidden: [1, batch, hidden]
        inpt = self.e(inpt)    # [seq, batch, hidden]
        
        if hidden is not None:
            hidden = self.trans_h(hidden)    # [1, batch, nh]
        
        embedded = nn.utils.rnn.pack_padded_sequence(inpt, lengths, enforce_sorted=False)
        output, _ = self.m(embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)    # [seq, batch, hidden]
        
        # [seq, batch, vocab_size]
        return self.l(output)
        