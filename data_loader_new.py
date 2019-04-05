import re
import pickle
import random
from generate_hol_dataset import Node
from data_statistics import data_static
import tensorflow as tf
import numpy as np
import os

class Data_loader(object):
    def __init__(self,sequence_num,max_lenth,batch_size,input_path,output_path):
        self.sequence_num = sequence_num
        self.max_lenth = max_lenth
        self.batch_size = batch_size
        self.input_path = input_path
        self.output_path = output_path

    def file_to_record(self):
        with open("data_class", 'rb') as f:
            static = pickle.load(f)
            dict = static.dict
        pos_pad = [0]*self.max_lenth
        files = os.listdir(self.input_path)
        for file in files:
            record_path = os.path.join(self.output_path,'{}.tfrecord'.format(file))
            writer = tf.python_io.TFRecordWriter(record_path)
            fpath = os.path.join(self.input_path,file)
            with open(fpath,'rb') as f:
                recorders = pickle.load(f)
                for recorder in recorders:
                    conj_feature = []
                    conj_position = []
                    if len(recorder[1])>self.sequence_num:
                        conj_lenth = self.sequence_num
                    else:
                        conj_lenth = len(recorder[1])
                    stat_feature = []
                    stat_position = []
                    if len(recorder[2])>self.sequence_num:
                        stat_lenth = self.sequence_num
                    else:
                        stat_lenth = len(recorder[2])
                    ans =recorder[0]
                    for conj_node in recorder[1]:
                        conj_feature.append(dict[conj_node.name]+1)
                        pos = conj_node.des
                        if len(pos)>self.max_lenth:
                            pos = pos[:self.max_lenth]
                        else:
                            while(len(pos)<self.max_lenth):
                                pos = pos + [-1]*(self.max_lenth-len(pos))
                        pos = np.array(pos,np.int64)+1
                        conj_position.append(pos)
                        if len(conj_feature)==self.sequence_num:
                            break
                    while len(conj_feature)<self.sequence_num:
                        conj_feature.append(0)
                        conj_position.append(pos_pad)
                    for stat_node in recorder[2]:
                        stat_feature.append(dict[stat_node.name]+1)
                        pos = stat_node.des
                        if len(pos)>self.max_lenth:
                            pos = pos[:self.max_lenth]
                        else:
                            while(len(pos)<self.max_lenth):
                                pos = pos + [-1]*(self.max_lenth-len(pos))
                        pos = np.array(pos,np.int64)+1
                        stat_position.append(pos)
                        if len(stat_feature)==self.sequence_num:
                            break
                    while len(stat_feature)<self.sequence_num:
                        stat_feature.append(0)
                        stat_position.append(pos_pad)
                    conj_position = np.reshape(conj_position,[-1])
                    stat_position = np.reshape(stat_position,[-1])
                    example = tf.train.Example(features = tf.train.Features(
                        feature={
                            'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [ans])),
                            'conj_feature':tf.train.Feature(int64_list = tf.train.Int64List(value = conj_feature)),
                            'conj_position':tf.train.Feature(int64_list = tf.train.Int64List(value = conj_position)),
                            'conj_lenth':tf.train.Feature(int64_list = tf.train.Int64List(value = [conj_lenth])),
                            'stat_feature':tf.train.Feature(int64_list = tf.train.Int64List(value = stat_feature)),
                            'stat_position':tf.train.Feature(int64_list = tf.train.Int64List(value = stat_position)),
                            'stat_lenth':tf.train.Feature(int64_list=tf.train.Int64List(value=[stat_lenth])),

                        }
                    ))
                    serialized = example.SerializeToString()
                    writer.write(serialized)
            writer.close()
        print("saved tfrecord file successful !!")



    def read_and_decode(self):
        reader = tf.TFRecordReader()
        files = os.listdir(self.output_path)
        file_names = [self.output_path+file for file in files]
        filename_queue = tf.train.string_input_producer(file_names, num_epochs=None)
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'conj_feature':tf.FixedLenFeature([self.sequence_num],tf.int64),
                                               'conj_position': tf.FixedLenFeature([self.max_lenth * self.sequence_num],tf.int64),
                                               'conj_lenth':tf.FixedLenFeature([],tf.int64),
                                               'stat_feature': tf.FixedLenFeature([self.sequence_num], tf.int64),
                                               'stat_position': tf.FixedLenFeature([self.sequence_num * self.max_lenth],tf.int64)
                                               'stat_lenth':tf.FixedLenFeature([],tf.int64)
                                           })
        label_d = features['label']
        label_d = tf.cast(label_d,tf.int32)
        conj_feature_d = features['conj_feature']
        conj_feature_d = tf.cast(conj_feature_d,tf.int32)
        conj_position_d = tf.reshape(features['conj_position'],[self.sequence_num,self.max_lenth])
        conj_position_d = tf.cast(conj_position_d,tf.int32)
        conj_lenth_d = tf.cast(features['conj_lenth'],tf.int32)
        stat_feature_d = features['stat_feature']
        stat_feature_d = tf.cast(stat_feature_d,tf.int32)
        stat_position_d = tf.reshape(features['stat_position'],[self.sequence_num,self.max_lenth])
        stat_position_d = tf.cast(stat_position_d,tf.int32)
        stat_lenth_d = tf.cast(features['stat_lenth'],tf.int32)
        return label_d,conj_feature_d,conj_position_d,conj_lenth_d,stat_feature_d,stat_position_d,stat_lenth_d

    def test(self):
        label_m,conj_feature_m,conj_position_m,stat_feature_m,stat_position_m = self.read_and_decode()
        label_batch,conj_f_batch,conj_p_batch,stat_f_batch,stat_p_batch = tf.train.shuffle_batch([label_m,conj_feature_m,conj_position_m,stat_feature_m,stat_position_m],4,32,16,4)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            tf.train.start_queue_runners(sess=sess)
            label,c_f= sess.run([label_batch,stat_f_batch])
            print(label)
            print(c_f)


if __name__ == '__main__':
    data_loader = Data_loader(50,10,56,r'test_data/',r'output.tfrecord')
    data_loader.file_to_record()
    data_loader.test()




