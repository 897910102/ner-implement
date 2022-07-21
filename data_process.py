import platform
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from collections import Counter
#导入参数
from Model_para import Hpara
hp = Hpara()
parser = hp.parser
para = parser.parse_args(args=[])

def _parse_data(file_path):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'


    string = file_path.read()
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    file_path.close()
    return data

def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i+1) for i, w in enumerate(vocab))
    np.save(para.word2idx_path,word2idx)
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1])+1 for w in s] for s in data]

    x = pad_sequences(x, maxlen,padding='post')  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen,padding='post')
    return x,y_chunk


def load_data():
    train_data = _parse_data(open(para.train_path))
    test_data = _parse_data(open(para.test_path))

    word_counts = Counter(row[0].lower() for sample in train_data for row in sample)
    vocab = [w for w,f in iter(word_counts.items())]
    chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
    train = _process_data(train_data, vocab, chunk_tags,maxlen=para.maxlen)
    test = _process_data(test_data, vocab, chunk_tags,maxlen=para.maxlen)
    return train, test, vocab, chunk_tags

if __name__=="__main__":
    train, test, vocab, chunk_tags = load_data()
    data_len = len(train[0])
    batch_num = (data_len+para.batch_size-1)//para.batch_size
    print('label_num:',len(chunk_tags))#去修改para参数
    print('vocab_num:',len(vocab)) #去修改para参数
    print('train_num:',data_len)
    print('batch_num:',batch_num)
    print(train[0][0:10])
    print(train[1][0:10])