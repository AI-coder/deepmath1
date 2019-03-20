import pickle
import os
class data_static():
    '''
    tong ji xiang guang shu ju
    '''
    def __init__(self,trian_path,test_path):
        self.trian_path = trian_path
        self.test_path = test_path
        self.max_depth = 0
        self.max_nodesize = 0
        self.vocb_size = None
        self.dict = {}
        self.data_size = 0 #数据的条数
        self.tree_size = 0 #树的叉数
        self.conj_node_size = []
        self.stats_node_size = []
        self.count_stmt()


    def __str__(self):
        return "max_depth:{} max_nodesize:{} vocab_size:{}".format(self.max_depth,self.max_nodesize,self.vocb_size)


    def count_stmt(self):
        max_node = None
        max_file = None
        vocabulary = set()
        trian_files = os.listdir(self.trian_path)
        trian_files.sort(key = lambda x: int(x[13:]))
        test_files = os.listdir(self.test_path)
        test_files.sort(key=lambda x: int(x[12:]))
        for i,fname in enumerate(trian_files+test_files):
            if i <len(trian_files):
                fpath = os.path.join(self.trian_path,fname)
            else:
                fpath = os.path.join(self.test_path,fname)
            print("Couting file {}/{} in {}".format(i+1,len(trian_files+test_files),fpath))
            with open(fpath,'rb') as f:
                recorder = pickle.load(f)
                if len(recorder)>0:
                    self.data_size = self.data_size + len(recorder)
                    self.conj_node_size.append(len(recorder[0][1]))
                    for data in recorder:
                        self.stats_node_size.append(len(data[2]))
                        lenth = max(len(data[1]),len(data[2]))
                        if self.max_nodesize<lenth:#单柯树求最大节点数
                            self.max_nodesize = lenth
                        for node in data[1]+data[2]:
                            vocabulary.add(node.name)
                            if len(node.des)!=0 and self.tree_size<max(node.des):#求树的最大叉数
                                self.tree_size = max(node.des)
                                max_node = node
                                max_file = fpath
                            if self.max_depth<node.depth:#求树的最大深度
                                self.max_depth = node.depth
        for index ,word in enumerate(vocabulary):
            self.dict[word] =index
        self.vocb_size = len(self.dict)

        print(max_node.des,max_file,max_node.name)





