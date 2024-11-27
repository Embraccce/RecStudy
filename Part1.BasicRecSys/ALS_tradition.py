import numpy as np
import dataloader
import random

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

class ALS():
    def __init__(self,n_users,n_items,n_factors=10):
        self.p = np.random.uniform(size=(n_users,n_factors))
        self.q = np.random.uniform(size=(n_items,n_factors))
        self.bu = np.random.uniform(size=(n_users, 1))
        self.bi = np.random.uniform(size=(n_items, 1))

    def forward(self,u,i): 
        return np.dot(self.p[u],self.q[i]) + self.bu[u] + self.bi[i] # r_ui
    
    def backward(self,r,r_pred,u,i,lr,lambd):
        loss = r - r_pred
        self.p[u] += lr*(loss*self.q[i] - lambd*self.p[u])
        self.q[i] += lr*(loss*self.p[u] - lambd*self.q[i])
        self.bu[u] += lr*(loss - lambd*self.bu[u])
        self.bi[i] += lr*(loss - lambd*self.bi[i])

def train(epochs=10,batch_size=1024,lr=0.01,lambd=0.1,factors_dim=64):
    # 加载数据
    user_set, item_set, train_set, test_set = dataloader.read_triples(file_path="./dataset/ml-100k/rating_index_5.tsv")

    # 初始化 ALS 模型
    als = ALS(n_users=len(user_set), n_items=len(item_set), n_factors=factors_dim)

    # 创建训练数据迭代器
    dataIter = DataIter(train_set, batch_size=batch_size)

    # 开始训练
    for epoch in range(epochs):
        total_loss = 0
        for user_ids, item_ids, ratings in dataIter:
            for u, i, r in zip(user_ids, item_ids, ratings):
                print(u,i,r)
        #         # 前向传播计算预测评分
        #         r_pred = als.forward(u, i)

        #         # 反向传播更新参数
        #         als.backward(r, r_pred, u, i, lr, lambd)

        #         # 累计损失
        #         total_loss += (r - r_pred) ** 2

        # # 打印当前轮次的平均损失
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_set)}")

    return als

    
if __name__ == "__main__":
    # 调用训练函数
    trained_model = train(epochs=10, batch_size=1024, lr=0.01, lambd=0.1, factors_dim=64)