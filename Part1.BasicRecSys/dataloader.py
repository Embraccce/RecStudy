import collections
import numpy as np
import csv
import random

def read_triples(file_path="./dataset/ml-100k/rating_index.tsv",test_ratio = 0.1):
    """
    读取用户-物品-评分数据文件，并将数据分为训练集和测试集。
    
    输入:
    - file_path: str，数据文件路径，文件格式为制表符分隔的文本文件，每行格式如下:
        user_id    item_id    rating
        例如：
        1    101    1
        2    102    0
        ...

    输出:
    - user_set: list，用户集合的列表（去重后）
    - item_set: list，物品集合的列表（去重后）
    - train_set: list，训练集，包含 (user_id, item_id, rating) 的三元组
    - test_set: list，测试集，包含 (user_id, item_id, rating) 的三元组
    
    逻辑:
    1. 初始化用户集合、物品集合和三元组列表。
    2. 打开文件并逐行读取，解析出用户 ID、物品 ID 和评分，将它们加入相应的集合和列表。
    3. 随机采样一定比例的数据作为测试集，其余作为训练集。
    4. 返回用户集合、物品集合以及划分后的训练集和测试集。
    """
    # 初始化用户集合、物品集合和三元组列表
    user_set, item_set = set(), set()
    triples = []
    
    # 打开文件并逐行读取
    with open(file_path, 'r', newline='') as file:
        # 使用 csv 模块读取制表符分隔的文件
        reader = csv.reader(file, delimiter='\t')  
        for u, i, r in reader:
            # 将用户和物品 ID 转为整数并加入集合
            user_set.add(int(u))
            item_set.add(int(i))
            # 将三元组 (user_id, item_id, rating) 添加到列表
            triples.append((int(u), int(i), int(r)))

    # 随机采样一定比例的三元组作为测试集
    test_set = random.sample(triples, int(len(triples) * test_ratio))
    # 剩余的三元组作为训练集
    train_set = list(set(triples) - set(test_set))

    # 返回用户集合列表，物品集合列表，与用户-物品-评分三元组的训练集和测试集
    return list(user_set), list(item_set), train_set, test_set


class DataIter:
    """
    数据迭代器，用于将数据按批次加载，并支持数据打乱 (shuffle)。
    
    输入:
    - data: list，包含 (user_id, item_id, rating) 的三元组。
    - batch_size: int，单个批次的大小。

    方法:
    - __iter__: 初始化迭代器，并打乱数据。
    - __next__: 返回下一个批次的数据。
    """
    def __init__(self, data, batch_size=1024, shuffle=True):
        """
        初始化 DataIter。
        
        参数:
        - data: list，包含 (user_id, item_id, rating) 的三元组。
        - batch_size: int，单个批次的大小。
        - shuffle: bool，是否在每轮迭代时打乱数据。
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)
        self.index = 0  # 当前批次的起始索引

    def __iter__(self):
        """
        初始化迭代器。
        如果 shuffle 为 True，则在每轮迭代前打乱数据。
        """
        self.index = 0
        if self.shuffle:
            random.shuffle(self.data)  # 打乱数据集
        return self

    def __next__(self):
        """
        返回下一个批次的数据。
        如果没有更多数据，则抛出 StopIteration。
        """
        # 检查是否还有剩余数据
        if self.index >= self.n_samples:
            raise StopIteration
        
        # 切片获取当前批次的数据
        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        # 转换为 NumPy 数组，便于后续操作
        user_ids = np.array([x[0] for x in batch], dtype=np.int32)
        item_ids = np.array([x[1] for x in batch], dtype=np.int32)
        ratings = np.array([x[2] for x in batch], dtype=np.float32)

        return user_ids, item_ids, ratings
