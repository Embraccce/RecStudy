from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import dataloader4ml100kOneHot
from torch.utils.data import DataLoader, TensorDataset


class LR(nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.b = nn.init.xavier_normal_(nn.Parameter(torch.empty(1,1)))
        self.w = nn.init.xavier_normal_(nn.Parameter(torch.empty(n_features,1)))
    def forward(self, x):        
        logits = torch.sigmoid(torch.matmul(x,self.w)+self.b)
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
    return LR(n_features)


def train_model(model, train_set, batch_size=1024, epochs=20, lr=0.01, wd=5e-3):
    """训练 LR 模型"""
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
    评估 LR 模型的性能，计算 precision, recall 和 accuracy。

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
    



