import numpy as np

def MSE(y_true, y_pred):
    """
    计算均方误差（Mean Squared Error，MSE）

    参数:
    y_true (list or array-like): 实际值序列
    y_pred (list or array-like): 预测值序列

    返回:
    float: 均方误差值
    """
    # 将输入转换为数组并计算差值的平方
    # 使用np.mean计算均值，得到均方误差
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def RMSE(y_true, y_pred):
    """
    计算均方根误差（Root Mean Squared Error，RMSE）

    参数:
    y_true: array-like，真实值数组
    y_pred: array-like，预测值数组

    返回:
    float，均方根误差值
    """
    # 调用MSE函数计算均方误差，然后取平方根得到均方根误差
    return np.sqrt(MSE(y_true, y_pred))

def MAE(y_true, y_pred):
    """
    计算真实值和预测值之间的平均绝对误差（MAE）。

    参数:
    y_true: 真实值数组，表示实际发生的数据。
    y_pred: 预测值数组，表示模型预测的数据。

    返回:
    返回真实值和预测值之间的平均绝对误差。
    """
    # 将输入的真实值和预测值转换为numpy数组，以便进行向量化操作
    # 计算真实值和预测值之间的差值，并取绝对值来计算绝对误差
    # 对所有样本的绝对误差求平均，得到平均绝对误差（MAE）
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def recall4Set(test_set, pred_set):
    """
    计算召回率 (Recall)。
    
    输入:
    - test_set: set，测试集的正样本集合
    - pred_set: set，预测集的正样本集合

    输出:
    - float，召回率（Recall），范围 [0, 1]

    公式:
    Recall = |pred_set ∩ test_set| / |test_set|
    其中，|pred_set ∩ test_set| 表示预测为正且实际为正的样本数量（TP）。
    """
    # 计算交集中的正样本数量，并除以测试集中正样本的总数
    return len(pred_set & test_set) / len(test_set)

def percision4Set(test_pos_set, test_neg_set, pred_set):
    """
    计算精确率 (Precision)。
    
    输入:
    - test_pos_set: set，测试集的正样本集合
    - test_neg_set: set，测试集的负样本集合
    - pred_set: set，预测集的正样本集合

    输出:
    - float，精确率（Precision），范围 [0, 1]

    公式:
    Precision = TP / (TP + FP)
    其中:
    - TP（True Positives）: 预测为正且实际为正的样本数量
    - FP（False Positives）: 预测为正但实际为负的样本数量
    """
    # 计算TP：预测为正且实际为正的样本数量
    TP = len(pred_set & test_pos_set)
    # 计算FP：预测为正但实际为负的样本数量
    FP = len(pred_set & test_neg_set)

    p = TP / (TP + FP) if TP + FP > 0 else None
    # p = TP/len(pred_set) #若对模型严格一点可这么去算精确度

    return p