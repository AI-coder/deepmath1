import tensorflow as tf
import numpy as np
from data_loader import Data_Loader
from generate_hol_dataset import Node
BATCH_SIZE = 8
SEQUENCE_NUM = 610
VOCAB_SIZE = 2033+1
MAX_DEPTH = 100
MAX_TREE_SIZE = 56+1
OUTPUT_LENTH = 128
HIDDEN_SIZE =128
DENSE1=128
DENSE2 = 64
DENSE3 = 2
KEEP_PROB = 0.6
LEARNING_RATE = 0.01
EPOCH = 200000
PATH = r"train_processed"

class Model(object):
    def __init__(self,input_tensor,position_tensor):
        self.input_tensor = input_tensor
        self.position_tensor = position_tensor
        self.position_layer()
        self.embed()
        self.lstm()
        self.dense1_layer1()
        self.dense_layer2()
        self.dense_layer3()
    def position_layer(self):
        with tf.variable_scope('position'):
            weight_depth = tf.get_variable('weight_depth',shape=[MAX_DEPTH,OUTPUT_LENTH],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.5))
            weight_son = tf.get_variable('weight_son',shape=[MAX_TREE_SIZE,OUTPUT_LENTH],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.5))
            gather = tf.gather(weight_son,self.position_tensor,axis=0)
            muti = tf.multiply(gather,weight_depth)
            self.position = tf.reduce_mean(muti,axis=2)

    def embed(self):
        with tf.variable_scope("embedding"):
            weights = tf.get_variable('weights',shape=[VOCAB_SIZE,OUTPUT_LENTH])
            self.embeding = tf.nn.embedding_lookup(weights,self.input_tensor)

    def lstm(self):
        with tf.variable_scope('lstm'):
            lstm_input = tf.concat([self.position, self.embeding], axis=2)  # 此时axis指定的为在该维数上相加
            print(lstm_input)
            lstm_qx = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0)
            lstm_hx = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0)
            output, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_qx, lstm_hx, lstm_input, dtype=tf.float32,                                                         time_major=False)
            qx_state = output_states[0]
            hx_state = output_states[1]
            state_h_concat = tf.concat([qx_state.h, hx_state.h], axis=1)
            self.lstm_output = state_h_concat
            print(state_h_concat.shape)
    def dense1_layer1(self):
        with tf.variable_scope("dense_layer1"):
            self.dense_l1 = tf.layers.dense(self.lstm_output,DENSE1,activation=tf.nn.relu)
            self.drop_out_l1 = tf.nn.dropout(self.dense_l1,KEEP_PROB)
    def dense_layer2(self):
        with tf.variable_scope("dense_l2"):
            self.dense_l2 = tf.layers.dense(self.drop_out_l1,DENSE2,activation=tf.nn.relu)
            self.drop_out_l2 = tf.nn.dropout(self.dense_l2,KEEP_PROB)
    def dense_layer3(self):
        with tf.variable_scope("dense_l3"):
            self.dense_l3 = tf.layers.dense(self.drop_out_l2,DENSE3)
            self.output_tensor = self.dense_l3

def train():
    tf.reset_default_graph()
    strps_hosts = "10.2.28.135:1681"
    strworker_hosts = "10.2.28.136:1681,10.2.28.137:1681,10.2.28.138:1681,10.2.28.139:1681,10.2.28.145:1681"
    strjob_name = "worker"
    task_index = 0
    ps_hosts = strps_hosts.split(',')
    worker_hosts = strworker_hosts.split(',')
    cluster_spec = tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})
    server = tf.train.Server(
        { 'ps':ps_hosts,'worker':worker_hosts},
    job_name=strjob_name,
    task_index = task_index)
    if strjob_name == 'ps':
        print("wait")
        server.join()
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster_spec)):
        global_step = tf.contrib.framework.get_or_create_global_step()
        Input = tf.placeholder(tf.int32, [None, SEQUENCE_NUM])
        Positon = tf.placeholder(dtype=tf.int32,shape=[None,SEQUENCE_NUM,MAX_DEPTH])
        Output = tf.placeholder(tf.int32, [None])
        model = Model(Input,Positon)
        logits = model.output_tensor
        correct = tf.nn.in_top_k(logits, Output, 1)
        correct = tf.cast(correct, tf.float32)
        correct = tf.reduce_mean(correct, axis=0)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Output, logits=logits))
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        saver = tf.train.Saver(max_to_keep=1)
        merged_summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
    sv = tf.train.Supervisor(is_chief=(task_index==0),
                         logdir="log/super/",
                         init_op=init,
                         summary_op=None,
                         saver=saver,
                         global_step=global_step,
                         save_model_secs=5)
    with sv.managed_session((server.target)) as sess:
        sess.run(init)
        index = 0
        for eopoch in range(EPOCH):
            data = Data_Loader(PATH, True,BATCH_SIZE)
            generate = data.get_batch( )
            for data in generate:
                features,position,label = data
                cost, _, predict, accuracy = sess.run([loss, optimizer, logits, correct],
                                                      feed_dict={Input: features,Positon:position, Output: label})
                print("minibatch_num:{} loss:{} accuracy:{}".format(index,cost,accuracy))
                index+=1

            if (eopoch + 1) % 1000 == 0:
                print("epoch:{} loss:{} accuracy:{} ".format(eopoch, cost, accuracy))
        sv.saver.save(sess, "log/mnist_with_summaries/" + "sv.cpk", global_step=epoch)
        print("successful")
    sv.stop()


if __name__ == '__main__':
    train()

