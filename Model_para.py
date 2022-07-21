"""
Created on Wed Oct 14 20:24:48 2020

@author: Johnma
"""

import argparse

class Hpara():
    parser = argparse.ArgumentParser()#构建一个参数管理对象

#数据集参数
    parser.add_argument('--train_path',default='./data/train_data.data',type=str)
    parser.add_argument('--test_path',default='./data/test_data.data',type=str)

    parser.add_argument('--len_vocab',default=4743,type=int)
    parser.add_argument('--label_num',default=7,type=int)
    parser.add_argument('--maxlen',default=100,type=int)
    parser.add_argument('--data_num',default=50658,type=int)

    parser.add_argument('--word2idx_path',default='./data/word2idx.npy')
#模型参数
    parser.add_argument('--lstm_unit',default=128,type=int)
    parser.add_argument('--embed_dim',default=32,type=int)
    parser.add_argument('--savepath',default='./data/check_point/lala',type=str)

#训练参数
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--batch_num',default=1584,type=int)
    parser.add_argument('--epoch',default=1,type=int)
    parser.add_argument('--learning_rate',default=0.05,type=float)
    

