## NNLM
Neural network language model

This repo contains the improved NNLM for computing the contigent probability
such as $P(T)$, $P(T|S)$ and the Mutual Information.

### 1. Requirements:
1. [bert-as-service](https://github.com/hanxiao/bert-as-service)
2. Pytorch 1.0+
3. tqdm
4. ipdb

### 2. Refer
* [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/abs/1510.03055)
* [Neural Probabilistic Language Models](https://link.springer.com/chapter/10.1007%2F3-540-33486-6_6)

### 3. Training data
* src-train.txt: training source dataset, one sentence one line
* src-train.embed: processed BERT embedding of the utterances in src-train.txt with shape $[data_size, 768]$ (use bert-as-service).
* tgt-train.txt: training target dataset, one sentence one line, the response to the sentence in the src-train.txt

### 4. Usage

Train the model

```bash
# train the dataset on the 0th GPU
./run.sh train dataset_name 0
```

Calculate the score

```bash
# test the NNLM on the 0th GPU
./run.sh test dataset_name 0
```