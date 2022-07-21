import os 
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from model import Model_Ner
import data_process
from Model_para import Hpara

from tensorflow.python.compiler.mlcompute import mlcompute


def batch_iter(x,y,batch_size = 16):
    data_len = len(x)
    index = np.random.permutation(np.arange(data_len))      #这个很重要
    num_batch = (data_len+para.batch_size)//para.batch_size
    x_shuff = x[index]
    y_shuff = y[index]

    for i in range(num_batch):
        start_index = i*batch_size
        end_index = min(start_index+batch_size,data_len)
        yield i,num_batch,x_shuff[start_index:end_index],y_shuff[start_index:end_index]

def train_step(x,y):
    with tf.GradientTape() as tape:
        loss,batch_pred_seq = model(x,y)
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return loss , batch_pred_seq

def start(train):
    #print(len(train[0]))   #查看多少条数据
    num_batch = (len(train[0])+para.batch_size)//para.batch_size
    for e in range(para.epoch):
        loss_epoch = 0
        with tqdm(total=num_batch) as pbas:
            for i,batch_num,x,y in batch_iter(train[0],train[1],para.batch_size):
                loss_step,pred_step = train_step(x,y)
                loss_epoch+=loss_step
                pbas.update(1)
        print('\n第 %d 个epoch的平均损失是%f\n'%(e+1,loss_epoch/num_batch))
    #model.summary()
    model.save_weights(para.savepath)



def val():
    model = Model_Ner(para)
    model.load_weights(para.savepath)
    predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
    word2idx = np.load(para.word2idx_path,allow_pickle=True).item()
    x = [word2idx.get(w[0].lower(), 1) for w in predict_text]
    length = len(x)
    x = pad_sequences([x], para.maxlen,padding='post')  # left padding
    out,lens = model(x,None,False)
    viterbi_path, _ = tfa.text.crf_decode(out, model.crf.transition_params,lens)
    return viterbi_path

if __name__=='__main__':
    hp = Hpara()
    parser = hp.parser
    para = parser.parse_args(args=[])
#    mlcompute.set_mlc_device(device_name='any')
    model = Model_Ner(para)
    optimizer=tf.keras.optimizers.Adam(lr=para.learning_rate)
    train, test, vocab, chunk_tags = data_process.load_data()
    train_z = [train[0][:1000],train[1][:1000]]#取前1000条数据实验
#    start(train_z)
    print(val())

