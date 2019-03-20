import tensorflow as tf
import numpy as np
import os
import re
import pickle
import random
from generate_hol_dataset import Node
MAX_DEPTH = 100
MAX_SEQUENCE = 610



# def name_one_hot_vector(name,dict):
#     vocab_size = len(dict)
#     one_hot = np.zeros(vocab_size)
#     one_hot[dict[name]] =1
#     return one_hot

class Data_Loader(object):
    '''
    path:the path of the processed_files
    shulffle:bool,if or not shuffle the dataset
    '''

    def __init__(self,path,shuffle,batchsize):
        self.path = path
        self.shuffle = shuffle
        self.batch_size = batchsize
        with open("data_class",'rb') as f:
            data = pickle.load(f)
        self.dict = data.dict

    def encoding(self,batch):
        features =[]
        positon = []
        ans = []
        pad = [0]*MAX_DEPTH
        for data in batch:
            ans.append(data[0])
            features.append([])
            positon.append([])
            for node in data[-1]:
                features[-1].append(self.dict[node.name]+1)
                pos = node.des
                while len(pos)<MAX_DEPTH:
                    pos.append(0)
                pos_array = np.array(pos,dtype=np.int32)+1
                positon[-1].append(pos_array)
            while len(positon[-1])<MAX_SEQUENCE:
                positon[-1].append(pad)
            while len(features[-1])<MAX_SEQUENCE:
                features[-1].append(0)
        return (features,positon,ans)

    def get_batch(self):
        files = os.listdir(self.path)
        file_index = 0
        remain_index = 0
        data_batch=[]
        while file_index <len(files):
            fpath = os.path.join(self.path,files[file_index])
            with open(fpath,'rb') as f:
                recorder = pickle.load(f)
            curren_lenth = len(data_batch)
            need_lenth = self.batch_size-curren_lenth
            # print(curren_lenth,remain_index,need_lenth,len(recorder))
            if remain_index + need_lenth < len(recorder):
                data_batch = data_batch + recorder[remain_index:remain_index + need_lenth]
                yield self.encoding(data_batch)
                # print("case1")
                # print(file_index)
                remain_index = remain_index + need_lenth
                data_batch = []
            elif remain_index + need_lenth == len(recorder):
                data_batch = data_batch + recorder[remain_index: remain_index + need_lenth]
                yield self.encoding(data_batch)
                # print("case2")
                # print(file_index)
                remain_index = 0
                data_batch = []
                file_index+=1
            elif remain_index + need_lenth > len(recorder):
                if file_index < len(files)-1:
                    # print("case3")
                    data_batch = data_batch+recorder
                    remain_index = 0
                    file_index +=1
                else:
                    data_batch = data_batch + recorder
                    yield self.encoding(data_batch)
                    # print("case4")
                    # print(file_index)








def get_info(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
        print(data)



if __name__ == '__main__':
    path = r'test_processed'
    loader = Data_Loader(path,True,1600,64)
    generate = loader.get_batch()
    for data in generate:
        print(data)





