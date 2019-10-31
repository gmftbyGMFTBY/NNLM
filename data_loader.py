#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.10.21


import torch
import numpy as np
import math
import pickle
import random
import ipdb
import argparse
from tqdm import tqdm
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


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

    
def tgt_vocab(path, savepath, maxsize=25000):
    words, corpus = {}, []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if not line:
                continue
            ws = list(line)
            corpus.append(ws)
            for w in ws:
                if words.get(w, None):
                    words[w] += 1
                else:
                    words[w] = 1
    print(f'[!] raw vocab size: {len(words)}')
    words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:maxsize]
    words = [i for i, j in words]
    words.extend(['<sos>', '<eos>', '<unk>', '<pad>'])
    w2idx, idx2w = {i:idx for idx, i in enumerate(words)}, words
    
    with open(savepath, 'wb') as f:
        pickle.dump([w2idx, idx2w], f)
    print(f'vocab file save into {savepath}, vocab size: {len(w2idx)}')


def tgt_content(vocabp, path, savepath, maxsize=1000000):
    # vocab
    w2idx, idx2w = load_pickle(vocabp)
    
    with open(path) as f:
        corpus = []
        for line in tqdm(f.readlines()):
            line = ''.join(line.strip().split())
            if line:
                ws = list(line)
                corpus.append(ws)
            
    maxsize = min(maxsize, len(corpus))
    corpus = random.sample(corpus, maxsize)
    
    # process the dataset
    dataset = []
    for line in tqdm(corpus):
        line = [w2idx['<sos>']] + [w2idx[i] if w2idx.get(i, None) else w2idx['<unk>'] for i in line] + [w2idx['<eos>']]
        dataset.append(line)
        
    with open(savepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'[!] dataset file save into {savepath}, dataset size: {len(dataset)}')


def pad_sequence(pad, batch, bs):
    maxlen = max([len(batch[i]) for i in range(bs)])
    for i in range(bs):
        batch[i].extend([pad] * (maxlen - len(batch[i])))
    
    
def get_batch(batch_size, w2idx, idx2w, dataset, max_len=50):
    turns = [len(i) for i in dataset]
    turnidx = np.argsort(turns)
    dataset = [dataset[idx] for idx in turnidx]
    
    # P(T)
    fidx, bidx = 0, 0
    while fidx < len(dataset):
        bidx = fidx + batch_size
        rbatch = dataset[fidx:bidx]    # [batch, lengths]
        
        lengths, batch = [], []
        maxlen = min(max_len, max([len(i) for i in rbatch]))
        for i in rbatch:
            if len(i) > maxlen:
                i = i[:maxlen-1] + [w2idx['<eos>']]
            lengths.append(len(i))
            batch.append(i + [w2idx['<pad>']] * (maxlen - len(i)))
            
        if torch.cuda.is_available():
            batch = torch.tensor(batch, dtype=torch.long).transpose(0, 1).cuda()    # [seq, batch]
            lengths = torch.tensor(lengths, dtype=torch.long).cuda()
        
        fidx = bidx
        
        if lengths[0] == 3:
            continue
        
        # batch: [seq, batch], lengths: [batch]
        yield w2idx, idx2w, batch, lengths, None
    

def get_batch_pair(spath, batch_size, w2idx, idx2w, dataset, max_len=50):
    # spath: pickle file, tpath: text file
    # fine-tuning with the small dataset, lower learning rate (1e-7)
    src = load_pickle(spath)
    
    assert len(src) == len(dataset)
    
    turns = [len(i) for i in dataset]
    turnidx = np.argsort(turns)
    dataset = [dataset[idx] for idx in turnidx]
    src = np.stack([src[idx] for idx in turnidx])
    
    # P(T|S)
    fidx, bidx = 0, 0
    while fidx < len(dataset):
        bidx = fidx + batch_size
        rbatch = dataset[fidx:bidx]    # [batch, lengths]
        sbatch = src[fidx:bidx]    # [batch, 768]
        
        lengths, batch = [], []
        maxlen = min(max_len, max([len(i) for i in rbatch]))
        for i in rbatch:
            if len(i) > maxlen:
                i = i[:maxlen-1] + [w2idx['<eos>']]
            lengths.append(len(i))
            batch.append(i + [w2idx['<pad>']] * (maxlen - len(i)))
            
        if torch.cuda.is_available():
            batch = torch.tensor(batch, dtype=torch.long).transpose(0, 1).cuda()    # [seq, batch]
            lengths = torch.tensor(lengths, dtype=torch.long).cuda()
            sbatch = torch.from_numpy(sbatch).unsqueeze(0).cuda()
        
        fidx = bidx
        
        if lengths[0] == 3:
            continue
        
        # batch: [seq, batch], lengths: [batch], sbatch: [batch, 768]
        yield w2idx, idx2w, batch, lengths, sbatch
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='utils function')
    parser.add_argument('--file', type=str, default='./data/pretrained/corpus.txt')
    parser.add_argument('--vocabp', type=str, default='./data/vocab.pkl')
    parser.add_argument('--datap', type=str, default='./data/pretrained/data.pkl')
    parser.add_argument('--maxsize', type=int, default=50000)
    
    args = parser.parse_args()
    
    if args.vocabp != 'none':
        tgt_vocab(args.file, args.vocabp, args.maxsize)
    tgt_content('./data/vocab.pkl', args.file, args.datap)