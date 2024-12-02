"""
直接将隐向量当作特征的embedding，通过端到端的训练方式来训练FM（Factorization Machine）模型。
"""
import torch
import torch.nn as nn
import numpy as np
import dataloader4ml100kIndexs
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset



class FM(nn.Module):
    def __init__(self, total_user_features, total_item_features, n_factors=128):
        """
        初始化 FM 类，定义模型的参数。
        
        参数:
        - total_user_features (int): 用户特征的总数（为了构建查找表，为不同的离散属性构建不同的Embbeding，这里是不同特征取值的总和）。
        - total_item_features (int): 物品特征的总数（为了构建查找表，这里是电影类别（1-19）的数量加上类别0）。
        - n_factors (int): 隐因子的维度，默认为128。
        """
        super(FM, self).__init__()
        # 用户嵌入层，将用户特征映射到隐因子空间
        self.user_embedding = nn.Embedding(total_user_features, n_factors, max_norm = 1)
        # 物品嵌入层，将物品特征映射到隐因子空间
        self.item_embedding = nn.Embedding(total_item_features, n_factors, max_norm = 1, padding_idx = 0)

    def FMcross(self, feature_embs):
        """
        计算 FM 模型中的二阶交互项（cross term）。FM 模型通过二阶交互项来捕捉特征之间的相互作用。
        
        参数:
        - feature_embs (Tensor): 输入特征的嵌入向量，形状为 [batch_size, n_features, n_factors]。
        
        返回:
        - output (Tensor): 计算得到的二阶交互项的输出，形状为 [batch_size, 1]。
        """

        # feature_embs: [batch_size, n_features, n_factors]
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2 # [batch_size, n_factors]
        sum_of_square = torch.sum(feature_embs ** 2, dim=1) # [batch_size, n_factors]
        output = square_of_sum - sum_of_square # [batch_size, n_factors]
        output = 0.5 * torch.sum(output, dim=1, keepdim=True) # [batch_size, 1]

        return output
    
    def get_all_feature_embs(self, u, i):
        """
        获取所有特征的嵌入向量，其中用户特征与物品特征的嵌入拼接。
        
        参数:
        - u: 用户的 ID（批量）
        - i: 物品的 ID（批量）
        
        返回:
        - all_feature_embs: 拼接后的所有特征嵌入
        """
        # 用户嵌入
        user_emb = self.user_embedding(u) # [batch_size, n_user_features, n_factors]
        
        # 物品特征嵌入（对物品特征进行嵌入并求和）
        item_embs = self.item_embedding(i) # [batch_size, total_item_features, n_factors]
        
        # 对物品的特征进行求和处理
        item_emb_sum = torch.sum(item_embs, dim=1, keepdim=True) # [batch_size, n_item_features, n_factors], 这里其实只有一项特征, n_item_features = 1
        
        # 将用户嵌入与物品嵌入拼接在一起，形成最终的特征嵌入
        all_feature_embs = torch.cat([user_emb, item_emb_sum], dim=1)
        
        return all_feature_embs
    

    def forward(self, u, i):
        all_feature_embs = self.get_all_feature_embs(u, i)
        out = self.FMcross(all_feature_embs)
        logits = torch.sigmoid(out)
        return logits
    

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


def initialize_model(total_user_features, total_item_features, n_factors):
    """初始化 FM 模型"""
    return FM(total_user_features, total_item_features, n_factors)


def train_model(model, train_set, batch_size=1024, epochs=20, lr=0.01, wd=5e-3):
    (user_features_train, item_features_train), y_train = train_set

    """训练 FM 模型"""
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


def evaluate_model(model, test_set,threshold=0.5):
    """
    评估 FM 模型的性能，计算 precision, recall 和 accuracy。

    参数:
    - model: 训练好的 LR 模型
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
    epochs = 20
    lr = 0.01
    wd = 5e-3 # 权重衰减率
    rate_thr = 3 # 评分阈值
    n_factors = 128 # 隐因子的维度

    # 步骤 1: 加载数据
    x_train, x_test, y_train, y_test = load_data(rate_thr)

    # 步骤 2: 初始化模型
    model = initialize_model(total_user_features=x_train[0].max().item() + 1, total_item_features=x_train[1].max().item() + 1, n_factors=n_factors)

    # 步骤 3: 训练模型
    model = train_model(model, (x_train,y_train), batch_size, epochs, lr, wd)

    # 步骤 4: 评估模型性能
    res = evaluate_model(model, (x_test, y_test))
    print(f"Evaluation Results - precision: {res['precision']}, recall: {res['recall']}, accuracy: {res['accuracy']}")