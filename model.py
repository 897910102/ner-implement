from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras import Model



class Crf_layer(layers.Layer):
    def __init__(self,para):
        super().__init__(self)
        self.label_nums=para.label_num+1
        #定义CRF的转移参数矩阵
        self.transition_params = tf.Variable(tf.random.uniform(shape=(self.label_nums, self.label_nums)))
        
    def call(self,inputs,targets,lens):
        '''
        inputs是经过attention层之后的输出，这里还要将输入的大小[batch_size,maxlen,attention_dim]调整为
        [batch_size,maxlen,label_nums]
        '''
        self.log_likelihood,self.tran_paras=tfa.text.crf_log_likelihood(inputs, 
                                                                        targets,
                                                                        lens,
                                                                        transition_params=self.transition_params)
        self.batch_pred_sequence,self.batch_viterbi_score=tfa.text.crf_decode(inputs,self.tran_paras,lens)
        self.loss=tf.reduce_sum(-self.log_likelihood)
        
        return self.loss,self.batch_pred_sequence


class Model_Ner(tf.keras.Model):
    def __init__(self,para):
        super().__init__(self)
        self.len_vocab = para.len_vocab+1
        self.lstm_unit = para.lstm_unit
        self.input_length = para.maxlen
        self.embed_dim = para.embed_dim
        self.label_nums=para.label_num+1

        self.embedding = tf.keras.layers.Embedding(input_dim=self.len_vocab,output_dim =  self.embed_dim,input_length = self.input_length)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_unit,return_sequences=True))
        self.dense=layers.Dense(self.label_nums,use_bias=False,trainable=True,kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.crf = Crf_layer(para)
        
    def call(self,inputs,label,training=True):
        train_x = tf.convert_to_tensor(inputs)
        self.lens = tf.reduce_sum(tf.sign(train_x),axis=-1)
        self.out = self.embedding(train_x)
        self.out = self.lstm(self.out)
        self.out=self.dense(self.out)#调整大小为[batch_size,maxlen,nums_label]
        
        if training:
            label = tf.convert_to_tensor(label)
            self.loss,self.batch_pred_seq = self.crf(self.out,label,self.lens)
            return self.loss,self.batch_pred_seq   
        else:        
            return self.out,self.lens