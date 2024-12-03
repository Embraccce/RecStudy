import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.dataloader4ml100kIndexs as dataloader4ml100kIndexs
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset


class Embbeding_CNN(nn.Module):
    """
    初始化Embbeding_CNN类。

    参数:
    - total_user_features: 用户特征总数。
    - total_item_features: 商品特征总数。
    - n_user_features: 每个用户具有的特征数量。
    - n_item_features: 每个商品具有的特征数量。
    - dim: 嵌入维度，默认为128。

    该构造函数初始化了用户和商品的特征嵌入，并构建了MLP层。
    """
    def __init__(self, total_user_features, total_item_features, n_user_features, n_item_features, dim=128):
        super(Embbeding_CNN,self).__init__()
        # 初始化用户特征嵌入，每个用户特征嵌入到dim维空间，确保嵌入向量的最大范数为1
        self.user_features = nn.Embedding(total_user_features, dim,max_norm=1)
        # 初始化商品特征嵌入，每个商品特征嵌入到dim维空间，确保嵌入向量的最大范数为1，并设置填充索引为0。
        self.item_features = nn.Embedding(total_item_features, dim,max_norm=1, padding_idx = 0)

        # 计算总特征数，包括用户特征和商品特征。
        total_features = n_user_features + n_item_features

        self.Conv = nn.Conv1d(in_channels = total_features, out_channels = 1, kernel_size = 3)

        # 初始化全连接层
        self.dense1 = self.dense_layer(dim - 2, dim // 2)
        self.dense2 = self.dense_layer(dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def dense_layer(self,in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh()
        )

    def forward(self, u, i):
        user_embs = self.user_features(u)
        item_embs = self.item_features(i)

        # item_emb_sum = torch.sum(item_embs, dim=1, keepdim=True)

        ui = torch.cat([user_embs, item_embs], dim=1)

        ui = self.Conv(ui)
        ui = ui.squeeze(1)

        ui = self.dense1(ui)
        ui = self.dense2(ui)

        ui = F.dropout(ui, training = self.training)
        logit = self.sigmoid(ui)

        return logit
    

def load_data(rate_thr=3):
    """加载数据集并返回相关集合"""
    x_train, x_test, y_train, y_test = dataloader4ml100kIndexs.get_train_test_split()

    user_features_train, item_features_train = x_train
    user_features_test, item_features_test = x_test

    # 将评分转化为二分类标注（r > rate_thr -> 1, 否则 -> 0）
    y_train = (y_train > rate_thr).astype(int)
    y_test = (y_test > rate_thr).astype(int)

    # 转换为 Tensor 类型
    user_features_train = torch.tensor(user_features_train, dtype=torch.long)
    item_features_train = torch.tensor(item_features_train, dtype=torch.long)
    user_features_test = torch.tensor(user_features_test, dtype=torch.long)
    item_features_test = torch.tensor(item_features_test, dtype=torch.long)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 变为列向量
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # 变为列向量


    return (user_features_train, item_features_train), (user_features_test, item_features_test), y_train, y_test


def initialize_model(total_user_features, total_item_features, n_user_features, n_item_features, n_factors):
    """初始化 Embbeding_CNN 模型"""
    return Embbeding_CNN(total_user_features, total_item_features, n_user_features, n_item_features, n_factors)


def train_model(model, train_set, batch_size=1024, epochs=20, lr=0.01, wd=5e-3):
    (user_features_train, item_features_train), y_train = train_set

    """训练 Embbeding_CNN 模型"""
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCELoss()


    # 将训练集转换为 DataLoader
    train_data = TensorDataset(user_features_train, item_features_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 开始训练过程
    for epoch in range(epochs):
        model.train()  # 将模型设置为训练模式
        epoch_loss = 0.0  # 记录每个 epoch 的总损失

        for u, i, r in train_loader:
            optimizer.zero_grad()  # 梯度清零

            # 前向传播：计算预测结果
            r_pred = model(u, i)

            # 计算损失
            loss = criterion(r_pred, r)
            epoch_loss += loss.item()  # 累加损失值

            # 反向传播：计算梯度
            loss.backward()

            # 更新参数
            optimizer.step()

        # 输出当前 epoch 的平均损失
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader)}")

    return model


def evaluate_model(model, test_set, threshold=0.5):
    """
    评估 Embbeding_MLP 模型的性能，计算 precision, recall 和 accuracy。

    参数:
    - model: 训练好的 Embbeding_MLP 模型
    - test_set: 测试集，包含特征和标签
    - threshold: 阈值，用于将预测分数转化为二分类标签

    返回:
    - metrics: 字典，包含 precision, recall 和 accuracy
    
    """
    # 将 test_set 拆分为特征和标签
    (user_features_test, item_features_test), y_test = test_set

    # 使用模型进行预测，获取预测的概率值
    with torch.no_grad():
        logits = model(user_features_test, item_features_test).numpy()

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
    epochs = 10
    lr = 0.01
    wd = 5e-3 # 权重衰减率
    rate_thr = 3 # 评分阈值
    n_factors = 128 # 隐因子的维度

    # 步骤 1: 加载数据
    x_train, x_test, y_train, y_test = load_data(rate_thr)

    # 步骤 2: 初始化模型
    model = initialize_model(total_user_features=x_train[0].max().item() + 1, 
                             total_item_features=x_train[1].max().item() + 1, 
                             n_user_features=x_train[0].shape[1],
                             n_item_features=x_train[1].shape[1], # 这里没有把类别embbeding相加，也可以相加，那么这里就只有1个类别特征
                             n_factors=n_factors)
    

    # 步骤 3: 训练模型
    model = train_model(model, (x_train,y_train), batch_size, epochs, lr, wd)

    # 步骤 4: 评估模型性能
    res = evaluate_model(model, (x_test, y_test))
    print(f"Evaluation Results - precision: {res['precision']}, recall: {res['recall']}, accuracy: {res['accuracy']}")
