#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.10.21


import torch
import numpy as np
import math
import pickle
import ipdb


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

    
def tgt_vocab_content(path, maxsize=25000):
    words, corpus = {}, []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            ws = line.split()
            corpus.append(ws)
            for w in ws:
                if words.get(w, None):
                    words[w] += 1
                else:
                    words[w] = 1
    words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:maxsize]
    words = [i for i, j in words]
    words.extend(['<sos>', '<eos>', '<unk>', '<pad>'])
    w2idx, idx2w = {i:idx for idx, i in enumerate(words)}, words
    
    # process the dataset
    dataset = []
    for line in corpus:
        line = [w2idx['<sos>']] + [w2idx[i] if w2idx.get(i, None) else w2idx['<unk>'] for i in line] + [w2idx['<eos>']]
        dataset.append(line)
    
    return w2idx, idx2w, dataset


def pad_sequence(pad, batch, bs):
    maxlen = max([len(batch[i]) for i in range(bs)])
    for i in range(bs):
        batch[i].extend([pad] * (maxlen - len(batch[i])))
    

def get_batch(spath, tpath, batch_size, w2idx, idx2w, dataset):
    # spath: pickle file, tpath: text file
    src = load_pickle(spath)
    
    assert len(src) == len(dataset)
    
    turns = [len(i) for i in dataset]
    turnidx = np.argsort(turns)
    dataset = [dataset[idx] for idx in turnidx]
    src = np.stack([src[idx] for idx in turnidx])
    
    # P(T)
    fidx, bidx = 0, 0
    while fidx < len(dataset):
        bidx = fidx + batch_size
        rbatch = dataset[fidx:bidx]    # [batch, lengths]
        
        lengths, batch = [], []
        maxlen = max([len(i) for i in rbatch])
        for i in rbatch:
            lengths.append(len(i))
            batch.append(i + [w2idx['<pad>']] * (maxlen - len(i)))
            
        if torch.cuda.is_available():
            batch = torch.tensor(batch, dtype=torch.long).transpose(0, 1).cuda()    # [seq, batch]
            lengths = torch.tensor(lengths, dtype=torch.long).cuda()
        
        fidx = bidx
        
        # batch: [seq, batch], lengths: [batch]
        yield w2idx, idx2w, batch, lengths, None
    
    
    # P(T|S)
    fidx, bidx = 0, 0
    while fidx < len(dataset):
        bidx = fidx + batch_size
        rbatch = dataset[fidx:bidx]    # [batch, lengths]
        sbatch = src[fidx:bidx]    # [batch, 768]
        
        lengths, batch = [], []
        maxlen = max([len(i) for i in rbatch])
        for i in rbatch:
            lengths.append(len(i))
            batch.append(i + [w2idx['<pad>']] * (maxlen - len(i)))
            
        if torch.cuda.is_available():
            batch = torch.tensor(batch, dtype=torch.long).transpose(0, 1).cuda()    # [seq, batch]
            lengths = torch.tensor(lengths, dtype=torch.long).cuda()
            sbatch = torch.from_numpy(sbatch).unsqueeze(0).cuda()
        
        fidx = bidx
        
        # batch: [seq, batch], lengths: [batch], sbatch: [batch, 768]
        yield w2idx, idx2w, batch, lengths, sbatch
    


if __name__ == "__main__":
    pass