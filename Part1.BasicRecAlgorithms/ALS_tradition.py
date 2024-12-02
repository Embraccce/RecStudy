import numpy as np
import dataloader
from dataloader import DataIter
import evaluate

class ALS():
    def __init__(self,n_users,n_items,n_factors=10):
        self.p = np.random.uniform(size=(n_users,n_factors))
        self.q = np.random.uniform(size=(n_items,n_factors))
        self.bu = np.random.uniform(size=(n_users, 1))
        self.bi = np.random.uniform(size=(n_items, 1))

    def forward(self,u,i):
        return np.sum(self.p[u]*self.q[i],axis=1,keepdims=True) + self.bu[u] + self.bi[i]    # r_ui
    
    def backward(self,r,r_pred,u,i,lr,lamda):
        loss = r - r_pred
        self.p[u] += lr*(loss*self.q[i] - lamda*self.p[u])
        self.q[i] += lr*(loss*self.p[u] - lamda*self.q[i])
        self.bu[u] += lr*(loss - lamda*self.bu[u])
        self.bi[i] += lr*(loss - lamda*self.bi[i])


def load_data(file_path,test_ratio):
    """加载数据集并返回相关集合"""
    user_set, item_set, train_set, test_set = dataloader.read_triples(file_path=file_path,test_ratio=test_ratio)
    return user_set, item_set, train_set, test_set

def initialize_model(user_count, item_count, factors_dim):
    """初始化 ALS 模型"""
    return ALS(n_users=user_count, n_items=item_count, n_factors=factors_dim)


def train_model(model, train_set, batch_size, epochs, lr, lamda):
    """训练 ALS 模型"""
    data_iter = DataIter(train_set, batch_size=batch_size)
    for epoch in range(epochs):
        for user_ids, item_ids, ratings in data_iter:
            u, i, r = user_ids, item_ids, ratings[:,None]
            # 前向传播计算预测评分
            r_pred = model.forward(u, i)

            # 反向传播更新参数
            model.backward(r, r_pred, u, i, lr, lamda)

        # 打印当前轮次的RMSE
        print(f"Epoch {epoch+1}, RMSE: {evaluate.RMSE(r, r_pred)}")
    return model

def evaluate_model(model, test_set):
    """
    评估 ALS 模型的性能
    
    参数:
    model: 已训练的 ALS 模型
    test_set: 测试数据集，格式为 (user_ids, item_ids, ratings)
    
    返回:
    dict: 包含 MSE、RMSE 和 MAE 的评估指标
    """

    test_set = np.array(test_set)
    u,i,r = test_set[:,0],test_set[:,1],test_set[:,2].reshape(-1,1)

    # 预测评分
    r_pred = model.forward(u, i)
    
    # 计算评估指标
    mse = evaluate.MSE(r, r_pred)
    rmse = evaluate.RMSE(r, r_pred)
    mae = evaluate.MAE(r, r_pred)
    
    # 返回评估指标
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}

    
if __name__ == "__main__":
    # 配置参数
    file_path = "./dataset/ml-100k/rating_index_5.tsv"
    factors_dim = 64
    batch_size = 1024
    epochs = 200
    lr = 0.01
    lamda = 0.1
    test_ratio = 0.1

    # 步骤 1: 加载数据
    user_set, item_set, train_set, test_set = load_data(file_path,test_ratio)

    # 步骤 2: 初始化模型
    model = initialize_model(len(user_set), len(item_set), factors_dim)

    # 步骤 3: 训练模型
    model = train_model(model, train_set, batch_size, epochs, lr, lamda)

    # 步骤 4: 评估模型性能
    res = evaluate_model(model, test_set)
    print(f"Evaluation Results - MSE: {res['MSE']}, RMSE: {res['RMSE']}, MAE: {res['MAE']}")



