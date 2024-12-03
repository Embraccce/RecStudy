import os
import sys
sys.path.append(os.getcwd())
from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import dataloader4ml100kOneHot
from torch.utils.data import DataLoader, TensorDataset

class POLY2(nn.Module):
    def __init__(self, n_features):
        """
        初始化 POLY2 类，定义模型的参数。
        
        参数:
        - n_features (int): 输入特征的数量。
        """
        
        super(POLY2, self).__init__()
        self.w0 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(1, 1))) # w0 是一个标量，用于偏置项
        self.w1 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(n_features, 1))) # w1 是一个形状为 [n_features, 1] 的矩阵，用于线性部分的权重
        self.w2 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(n_features, n_features))) # w2 是一个形状为 [n_features, n_features] 的矩阵，用于二次交叉项

    def crossLayer(self, x):
        """
        计算二次交叉项。
        
        参数:
        - x (Tensor): 输入张量，形状为 [batch_size, n_features]。
        
        返回:
        - cross_out (Tensor): 二次交叉项的输出，形状为 [batch_size, 1]。
        """
        x = x.unsqueeze(2) # 将输入张量扩展为 [batch_size, n_features, 1]
        x_transpose = x.transpose(1, 2) # 计算输入的转置，形状为 [batch_size, 1, n_features]
        x_cross = torch.bmm(x, x_transpose) # 计算每个样本的二次交叉矩阵乘法，结果形状为 [batch_size, n_features, n_features]
        cross_out = torch.sum(self.w2 * x_cross, dim=(1, 2)).reshape(-1, 1) # 对交叉结果和 w2 做 Hadamard 乘积，并求和
        
        return cross_out

    def forward(self, x):
        """
        前向传播逻辑。
        
        参数:
        - x (Tensor): 输入张量，形状为 [batch_size, n_features]。
        
        返回:
        - logits (Tensor): 模型的输出概率，形状为 [batch_size, 1]。
        """
        lr_out = self.w0 + torch.matmul(x, self.w1) # 计算线性模型的输出部分 lr_out = w0 + x * w1
        cross_out = self.crossLayer(x) # 计算二次交叉项的输出
        logits = torch.sigmoid(lr_out + cross_out) # 计算最终的输出，并应用 sigmoid 激活函数
        
        return logits
    

def load_data(rate_thr=3):
    """加载数据集并返回相关集合"""
    x_train, x_test, y_train, y_test = dataloader4ml100kOneHot.get_train_test_split()

    # 将评分转化为二分类标注（r > rate_thr -> 1, 否则 -> 0）
    y_train = (y_train > rate_thr).astype(int)
    y_test = (y_test > rate_thr).astype(int)

    # 转换为 Tensor 类型
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 变为列向量
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # 变为列向量


    return x_train, x_test, y_train, y_test


def initialize_model(n_features):
    """初始化 LR 模型"""
    return POLY2(n_features)


def train_model(model, train_set, batch_size=1024, epochs=20, lr=0.01, wd=5e-3):
    """训练 POLY2 模型"""
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCELoss()


    # 将训练集转换为 DataLoader
    train_data = TensorDataset(train_set[0], train_set[1])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 开始训练过程
    for epoch in range(epochs):
        model.train()  # 将模型设置为训练模式
        epoch_loss = 0.0  # 记录每个 epoch 的总损失

        for x, y in train_loader:
            optimizer.zero_grad()  # 梯度清零

            # 前向传播：计算预测结果
            y_pred = model(x)

            # 计算损失
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()  # 累加损失值

            # 反向传播：计算梯度
            loss.backward()

            # 更新参数
            optimizer.step()

        # 输出当前 epoch 的平均损失
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader)}")

    return model


def evaluate_model(model, test_set,threshold=0.5):
    """
    评估 POLY2 模型的性能，计算 precision, recall 和 accuracy。

    参数:
    - model: 训练好的 LR 模型
    - test_set: 测试集，包含特征和标签
    - threshold: 阈值，用于将预测分数转化为二分类标签

    返回:
    - metrics: 字典，包含 precision, recall 和 accuracy
    
    """
    # 将 test_set 拆分为特征和标签
    x_test, y_test = test_set

    # 使用模型进行预测，获取预测的概率值
    with torch.no_grad():
        logits = model(x_test).numpy()

    # 将预测概率转化为二分类标签，使用给定的阈值
    y_pred = (logits >= threshold).astype(int)

     # 计算 precision, recall, accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # 返回评估指标
    metrics = {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }
    
    return metrics


if __name__ == "__main__":
    # 配置参数
    batch_size = 1024
    epochs = 20
    lr = 0.01
    wd=5e-3
    rate_thr=3

    # 步骤 1: 加载数据
    x_train, x_test, y_train, y_test = load_data(rate_thr)

    # 步骤 2: 初始化模型
    model = initialize_model(n_features = x_train.shape[1])

    # 步骤 3: 训练模型
    model = train_model(model, (x_train,y_train), batch_size, epochs, lr, wd)

    # 步骤 4: 评估模型性能
    res = evaluate_model(model, (x_test, y_test))
    print(f"Evaluation Results - precision: {res['precision']}, recall: {res['recall']}, accuracy: {res['accuracy']}")