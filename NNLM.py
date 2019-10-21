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
    
    def __init__(self, vocab_size, hidden=768):
        super(NNLM, self).__init__()
        
        self.e = nn.Embedding(vocab_size, hidden)
        self.m = nn.GRU(hidden, hidden)
        self.l = nn.Linear(hidden, vocab_size)
        
        self.init_weight()
            
    def init_weight(self):
        # orthogonal init
        init.orthogonal_(self.m.weight_hh_l0)
        init.orthogonal_(self.m.weight_ih_l0)
        self.m.bias_ih_l0.data.fill_(0.0)
        self.m.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, inpt, lengths, hidden=None):
        # inpt: [seq, batch]
        inpt = self.e(inpt)    # [seq, batch, hidden]
        
        embedded = nn.utils.rnn.pack_padded_sequence(inpt, lengths, enforce_sorted=False)
        output, _ = self.m(embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)    # [seq, batch, hidden]
        
        # [seq, batch, vocab_size]
        return F.log_softmax(self.l(output), dim=1)
        