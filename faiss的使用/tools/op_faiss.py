import faiss, os, json
import numpy as np
import cv2
from PIL import Image
from abc import ABC, abstractmethod


class OP_FAISS(ABC):
    def __init__(self, vector_dim=64, index_type='Flat', 
                 data_dir = None,
                 metric_type=faiss.METRIC_L2, test_mode=True):
        """_summary_
        Args:
            vector_dim:  向量维度
            metric_type: 向量度量类型
            index_type:  index 类型
                         'Flat': 暴力精确检索，全局最优，适合数十万级。
            
            data_dir:    图像对应文件夹的路径
            metric_type: 是否是测试模型， True: 使用人造向量数据
                                        False: 使用真实场景中的向量数据
        """
        print('coming.........')
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.metric_type = metric_type
        # self.test_model = test_mode
        self.vector_db = None
        self.vector_query = None
        # self.data_vector_func = data_vector_func # 该函数需要外部写好后，传入此处
        self.data_dir = data_dir
        self.support_img_format = [".jpg", ".png"]
        print(f'test_mode:{test_mode}')
        if test_mode:
            print("正处在测试模式下。。。。。。。==")
            self.prepare_vector_db()
        else:
            self.create_real_vector_db()
            
        self.build_vector_db_index()
    
    
    @abstractmethod
    def data_vector_func(self):
        pass
    
    
    def create_real_vector_db(self):   
        model = self.data_vector_func()
        feat_l = []
        label_d = {}
        # 假设图像的标签值 全部都为 '1'
        img_paths_l, labels = os.listdir(self.data_dir), len(os.listdir(self.data_dir)) * ['1']
        for img_no, i in enumerate(img_paths_l):
            img_path = os.path.join(self.data_dir, i)
            img = cv2.imread(img_path)
            label_d[i] = labels[img_no]
            # 还可以对图像做一些其他的处理，eg：resize等
            feat =  model(img.unqueeze(0)).squeeze().numpy()
            feat_l.append(feat)
        # ----------------------------------------******** self.vector_db 就是存在数据库中的向量 *******-------------------
        self.vector_db = np.stack(feat_l)
        with open('label.json','w') as f:
            json.dump(label_d, f)


        
    def prepare_vector_db(self, vector_db_num=1000, query_vector_num=100):
        """ 为测试做准备的，正式场景中并不使用
        vector_db_num: 创建该向量数据库时模拟创建向量的数量
        """
        # 1- 准备向量库向量
        np.random.seed(1234)
        self.vector_db = np.random.random((vector_db_num, self.vector_dim)).astype('float32')
        self.vector_db[:, 0] += np.arange(vector_db_num) / 1000.  # 将第一个维度的每个元素增加一个递增的值，使得每个向量在第一个维度上是唯一的。
        faiss.normalize_L2(self.vector_db)  # 将数据进行L2归一化处理
        print('self.vector_db=',self.vector_db.shape,'\n')
        
        # 2- 准备查询向量
        self.vector_query = np.random.random((query_vector_num, self.vector_dim)).astype('float32')
        self.vector_query[:, 0] += np.arange(query_vector_num) / 1000.  # 将第一个维度的每个元素增加一个递增的值，使得每个向量在第一个维度上是唯一的。
        faiss.normalize_L2(self.vector_query)  # 将数据进行L2归一化处理
        print('self.vector_query=',self.vector_query.shape,'\n')
        
        
    def build_vector_db_index(self):
        """
        构建向量库索引
        """
        self.index = faiss.index_factory(self.vector_dim,  # 向量维度
                                         self.index_type,  # 索引类型，用于指定索引的存储和搜索方式。例如，Flat 表示平面索引，IVF 表示倒排文件索引等。
                                         self.metric_type) # 度量类型，用于指定索引中使用的相似度度量方式。例如，faiss.METRIC_INNER_PRODUCT 表示使用内积作为度量，而 faiss.METRIC_L2 表示使用欧氏距离。
        print('index.is_trained=',self.index.is_trained) 
        """
        self.index.is_trained 的含义解释
        该属性用于表示索引是否已经经过训练（trained）。这个属性是一个布尔值，返回 True 或 False。
        ---True：表示索引已经训练完成。对于某些类型的索引，如基于聚类的索引（例如倒排文件索引 IVF），训练步骤是必要的，因为它们需要学习数据的聚类结构。
        ---False：False：表示索引尚未训练。对于不需要训练的索引，如平面索引（Flat），这个属性默认为 True，因为它们不需要额外的训练步骤，只需直接添加向量即可。
        ======================
        训练的目的：
        --训练索引的目的主要是为了优化搜索性能和准确性。对于需要训练的索引类型：
        1. 学习数据结构：训练过程中，索引会学习数据的分布特征，例如通过聚类算法找到数据的中心点。
        2. 构建辅助数据结构：训练可以构建一些辅助数据结构，如量化码本（用于量化索引）或划分边界（用于倒排文件索引）。
        3. 提高搜索效率：经过训练的索引在搜索时能够更快地定位到可能的匹配项，从而提高搜索速度。
        """
        if self.vector_db is not None:
            if not self.index.is_trained:
                self.index.train(self.vector_db)
            self.index.add(self.vector_db)
            print(f'index-num:{self.index.ntotal}') # 应该就是向量数据库中的向量个数
            return 
        print(f'create a new and null index......')
        
    def similar_search(self, top_k):
        """
        最关键的步骤，相似向量查询
        top_k: 要寻找相似度的前 k 名
        return:
                distance_l: 为每个待检索query最相似TopK的索引list
                top_k_index_l: 为其对应的距离
        """
        distance_l, top_k_index_l= self.index.search(self.vector_query, top_k)
        return distance_l, top_k_index_l
    
    def add(self, np_vector):
        """
        增加索引向量
        """
        # xa = np.random.random((10000, d)).astype('float32')
        # xa[:, 0] += np.arange(len(xa)) / 1000.                
        # faiss.normalize_L2(xa)
        # index.add(xa)
        if np_vector.shape[-1] !=self.vector_dim:
            print(f'索引向量的维度不满足要求!!!')
            return
        self.index.add(np_vector)
        
    
    def rm(self, idx):
        """
        删除索引向量
        """
        # index.remove_ids(np.arange(1000,1111))
        self.index.remove_ids(idx)
    
    def get_index_info(self):
        # 索引中的向量总数
        index_num = self.index.ntotal 
        # 索引中的维度
        vector_dim = self.index.d
        # 索引类型
        # index_type = self.index.index_type
        # 索引是否已经训练
        index_is_trained = self.index.is_trained
        # 索引辅助数据
        if isinstance(self.index, faiss.IndexIVF):
            print("聚类中心:", self.index.cluster_centers)
        print(f"index-num:{index_num}")
        # print(f"index-type:{index_type}")
        print(f'index-dim:{vector_dim}')
        print(f"index_is_trained:{index_is_trained}")
        
    
    def load_index(self, saved_index_path='image_vector.index'):
        """
        从文件中加载出index
        Args:
            saved_index_path (str, optional): _description_. Defaults to 'image_vector.index'.
        """
        self.index = faiss.read_index(saved_index_path)

        
        
        
    
    def save_index(self,save_index_path='image_vector.index'):
        """
        保存index索引保存起来
        """
        faiss.write_index(self.index, save_index_path)
    
if __name__ == '__main__':
    class my_faiss(OP_FAISS):

        def __init__(self, *args, **kwargs):
            super(my_faiss, self).__init__(*args, **kwargs)
        
        def data_vector_func(self):
            import torch
            from torchvision.models import resnet50
            # 加载预训练模型
            model = resnet50(pretrained=True)
            model.fc = torch.nn.Identity()  # 替换最后一层为恒等映射
            model.eval()
            return model