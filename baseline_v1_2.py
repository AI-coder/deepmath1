import tensorflow as tf
import os
import numpy as np
from data_loader_new import Data_loader
from data_statistics import data_static
from generate_hol_dataset import Node
os.environ['CUDA_VISIBLE_DEVICES']='2'
BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
SEQUENCE_NUM = 128
VOCAB_SIZE = 2033+1
MAX_DEPTH = 100
MAX_TREE_SIZE = 56+1
OUTPUT_LENTH = 256
HIDDEN_SIZE =256
DENSE1=128
DENSE2 = 64
DENSE3 = 2
LEARNING_RATE = 0.001
EPOCH = 2000
DATA_SIZE =2030000
TRAIN_PATH = r"train_processed"
TEST_PATH = r"test_processed"

class Model():
    def __init__(self,input_tensor,position_tensor,drop_rate):
        self.input_tensor = input_tensor
        self.position = position_tensor
        self.drop_rate = drop_rate
        self.built_model()
    def built_model(self):
        with tf.variable_scope("position"):
            weight_depth = tf.get_variable("weight_depth",shape=[MAX_DEPTH,OUTPUT_LENTH],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            weight_son  = tf.get_variable("weight_son",shape=[MAX_TREE_SIZE,OUTPUT_LENTH],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            gather = tf.gather(weight_son,self.position,axis=0)
            muti = tf.multiply(gather,weight_depth)
            self.position = tf.reduce_sum(muti,axis=2)

        with tf.variable_scope("feature_embedding"):
            weight = tf.get_variable("weight",shape=[VOCAB_SIZE,OUTPUT_LENTH],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding = tf.nn.embedding_lookup(weight,self.input_tensor)

        with tf.variable_scope("lstm"):
            lstm_input = tf.concat([self.position,self.embedding],axis=-1)
            x = tf.unstack(lstm_input,SEQUENCE_NUM,1)
            lstm_qx = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE,forget_bias =1.0)
            lstm_hx = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE,forget_bias = 1.0)
            output,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_qx,lstm_hx,x,dtype=tf.float32)
            # lstm_concat = tf.concat(output,axis=-1)
            # lstm_concat = tf.transpose(lstm_concat,[1,0,2])
            self.lstm_output = output[-1]

        with tf.variable_scope("fc_1"):
            fc = tf.layers.dense(self.lstm_output,DENSE1,activation=tf.nn.relu,)
            self.fc_1 = tf.layers.dropout(fc,self.drop_rate)#1-KEEPDROP

        with tf.variable_scope("fc_2"):
            fc = tf.layers.dense(self.fc_1,DENSE2,activation=tf.nn.relu)
            self.fc_2 = tf.layers.dropout(fc,self.drop_rate)

        with tf.variable_scope("softmax_layer"):
            self.output_tensor = tf.layers.dense(self.fc_2,DENSE3)



def train():
    Input = tf.placeholder(dtype=tf.int32,shape=[None,SEQUENCE_NUM])
    Position = tf.placeholder(dtype=tf.int32,shape=[None,SEQUENCE_NUM,MAX_DEPTH])
    Drop_rate = tf.placeholder(tf.float32)
    Output = tf.placeholder(dtype=tf.int32,shape=[None])
    model = Model(Input,Position,Drop_rate)
    logits = model.output_tensor
    correct = tf.nn.in_top_k(logits,Output,1)
    correct = tf.cast(correct,tf.float32)
    correct = tf.reduce_mean(correct,-1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Output,logits=logits),-1)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    saver = tf.train.Saver()
    test_data_loader = Data_loader(SEQUENCE_NUM,MAX_DEPTH,BATCH_SIZE,TEST_PATH,r'tf_data/test/')
    train_data_loader = Data_loader(SEQUENCE_NUM,MAX_DEPTH,BATCH_SIZE,TRAIN_PATH,r'tf_data/train/')
    test_ans,test_conj_f,test_conj_p,test_stat_f,test_stat_p = test_data_loader.read_and_decode()
    train_ans,train_conj_f,train_conj_p,train_stat_f,train_stat_p = train_data_loader.read_and_decode()
    label_batch, conj_f_batch, conj_p_batch, stat_f_batch, stat_p_batch = tf.train.shuffle_batch([train_ans,train_conj_f,train_conj_p,train_stat_f,train_stat_p],BATCH_SIZE,BATCH_SIZE*4,BATCH_SIZE*2,32)
    test_label_batch,test_conj_f_batch,test_conj_p_batch,test_stat_f_batch,test_stat_p_batch = tf.train.batch([test_ans,test_conj_f,test_conj_p,test_stat_f,test_stat_p],TEST_BATCH_SIZE,32)
    init =tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        for epoch in range(EPOCH):
            for mini_step in range(2040000//BATCH_SIZE):
                label_data,stat_f_data,stat_p_data = sess.run([label_batch,stat_f_batch,stat_p_batch])
                if mini_step==0:
                    print("successful")
                train_acuracy,train_loss,_ = sess.run([correct,loss,optimizer],feed_dict={Input:stat_f_data,Position:stat_p_data,Output:label_data,Drop_rate:0.5})
                if mini_step%1000==0:
                    print("epoch :{},mini_step:{},train_loss;{}，train_arrcuracy:{}".format(epoch,mini_step,train_loss,train_acuracy))
            sum_test_accuracy = 0
            sum_test_loss = 0
            for index in range(200000//TEST_BATCH_SIZE):
                test_label_data,test_stat_f_data,test_stat_p_data =sess.run([test_label_batch,test_stat_f_batch,test_stat_p_batch])
                test_accuracy,test_loss = sess.run([correct,loss],feed_dict={Input:test_stat_f_data,Position:test_stat_p_data,Output:test_label_data,Drop_rate:0.0})
                sum_test_loss+=test_loss
                sum_test_accuracy+=test_accuracy
            print("epoch:{},test_loss:{},test_accuracy;{}".format(epoch,sum_test_loss/index,sum_test_accuracy/index))
            saver.save(sess=sess,save_path='log/baseline_v1_2/baseline_v1_2.ckpt',global_step=1)
        coord.request_stop()
        coord.join(threads)







if __name__ == '__main__':
    train()








