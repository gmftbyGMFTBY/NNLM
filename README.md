## NNLM
Neural network language model

This repo contains the improved NNLM for computing the contigent probability
such as $P(T)$, $P(T|S)$ and the Mutual Information.

### 1. Requirements:
1. [bert-as-service](https://github.com/hanxiao/bert-as-service) 1.9.8+
2. Pytorch 1.0+
3. numpy
4. tqdm
5. ipdb
6. jieba
7. Python 3.6+

### 2. Refer
* [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/abs/1510.03055)
* [Neural Probabilistic Language Models](https://link.springer.com/chapter/10.1007%2F3-540-33486-6_6)

### 3. Training data
* src-train.txt: training source dataset, one sentence one line
* src-train.embed: processed BERT embedding of the utterances in src-train.txt with shape [data_size, 768] (use bert-as-service).
* tgt-train.txt: training target dataset, one sentence one line, the response to the sentence in the src-train.txt

### 4. Usage

Process the pretrained corpus
* Download the data from https://github.com/brightmart/nlp_chinese_corpus (baike_qa_2019)
* Unzip and get the json file (here we only use the train file)
* Generate the vocab and dataset
    ```bash
    cd data/pretrianed
    python process.py    # generate the corpus.txt
    cd -
    # vocab: ./data/vocab.pkl; data: ./data/pretrianed/data.pkl
    ./run.sh vocab pretrained 0
    ```

Train the model

```bash
# train the dataset on the 0th GPU
./run.sh train pretrained 0
```

More details can be found in `test.ipynb`