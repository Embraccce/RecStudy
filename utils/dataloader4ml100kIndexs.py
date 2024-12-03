import numpy as np
import pandas as pd

def load_ml100k_user_item_indices(user_path, item_path):
    """
    加载并处理 ml-100k 数据集的用户和物品特征，各自转换为连续不重复的索引。

    参数:
    - user_path: 用户特征文件路径
    - item_path: 物品特征文件路径

    返回:
    - user_indices: 用户特征索引数组
    - item_indices: 物品特征索引数组
    """
    # 加载用户数据，定义列名
    user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_data = pd.read_csv(user_path, sep='|', names=user_columns, engine='python')
    
    # 对年龄进行分组：每10岁为一组
    user_data['age_group'] = user_data['age'] // 10  

    # 初始化索引计数器，用于生成连续不重复的特征索引
    current_index = 0

    # 用户特征索引数组
    user_indices = []

    # 处理每一列特征：'age_group', 'gender', 'occupation'
    for col in ['age_group', 'gender', 'occupation']:
        # 获取该列的唯一值，并排序（确保特征值有序）
        unique_values = user_data[col].unique()

        # 为该列的每个唯一值分配一个连续的索引
        col_feature_map = {value: current_index + idx for idx, value in enumerate(sorted(unique_values))}
        
        # 更新当前索引计数器
        current_index += len(unique_values)

        # 将该列的特征值映射为对应的索引，并添加到 user_indices
        user_indices.append(user_data[col].map(col_feature_map).values)

    # 将所有特征的索引数组组合成一个二维数组
    user_indices = np.vstack(user_indices).T

    # 加载物品数据，定义列名，包括电影的特征 'feature_1' 到 'feature_19'
    item_columns = ['item_id', 'title', 'release_date', 'video_release_date', 
                    'IMDb_url'] + [f'feature_{i}' for i in range(1, 20)]
    item_data = pd.read_csv(item_path, sep='|', names=item_columns, engine='python', encoding='ISO-8859-1')

    # 提取特征列的 OneHot 编码矩阵
    feature_matrix = item_data[[f'feature_{i}' for i in range(1, 20)]].to_numpy()

    # 创建一个与 feature_matrix 行数一致的特征矩阵，序列是 [1, 2, 3, ..., 19]
    item_indices = np.tile(np.arange(1, 20), (feature_matrix.shape[0], 1)) # 第一个参数为y轴扩大的倍数，第二个参数为x轴扩大的倍数

    # 将OneHot 编码矩阵乘以 item_indices，得到每个特征对应的索引，未出现则用 0 填充，方便后续的 Embedding 层（Embedding 层设置padding_idx = 0）
    item_indices = feature_matrix * item_indices

    return user_indices, item_indices



def process_ml100k_ratings_data_with_indices(ratings_path, user_indices, item_indices):
    """
    根据评分数据组合用户和物品特征索引，分别返回用户特征索引和物品特征索引。

    参数:
    - ratings_path: 评分文件路径
    - user_indices: 用户特征索引数组
    - item_indices: 物品特征索引数组

    返回:
    - user_features: 用户特征索引数组
    - item_features: 物品特征索引数组
    - labels: 评分数据作为标签
    """
    # 加载评分数据
    ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_data = pd.read_csv(ratings_path, sep='\t', names=ratings_columns, engine='python')

    # 根据用户和物品 ID 提取特征索引
    user_ids = ratings_data['user_id'] - 1  # 将 ID 转为索引，从 0 开始
    item_ids = ratings_data['item_id'] - 1
    labels = ratings_data['rating'].values  # 获取评分

    # 提取用户特征索引和物品特征索引
    user_features = user_indices[user_ids]
    item_features = item_indices[item_ids]

    return (user_features, item_features), labels


def get_train_test_split(
    train_ratings_path='./dataset/ml-100k-orginal/ua.base', 
    test_ratings_path='./dataset/ml-100k-orginal/ua.test', 
    user_path="./dataset/ml-100k-orginal/u.user", 
    item_path='./dataset/ml-100k-orginal/u.item'
):
    """
    加载训练集和测试集并返回训练和测试数据。

    参数:
    - train_ratings_path: 训练集路径
    - test_ratings_path: 测试集路径
    - user_path: 用户数据路径
    - item_path: 物品数据路径

    返回:
    - x_train: 训练特征索引,包含用户特征索引和物品特征索引,是一个形如(user_features, item_features)的元组
    - x_test: 测试特征索引,包含用户特征索引和物品特征索引,是一个形如(user_features, item_features)的元组
    - y_train: 训练标签
    - y_test: 测试标签
    """
    # 加载用户和物品特征索引
    user_indices, item_indices = load_ml100k_user_item_indices(user_path, item_path)

    # 加载训练集
    x_train, y_train = process_ml100k_ratings_data_with_indices(train_ratings_path, user_indices, item_indices)

    # 加载测试集
    x_test, y_test = process_ml100k_ratings_data_with_indices(test_ratings_path, user_indices, item_indices)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    # 文件路径
    user_file = "./dataset/ml-100k-orginal/u.user"
    item_file = './dataset/ml-100k-orginal/u.item'
    train_ratings_file = './dataset/ml-100k-orginal/ua.base'
    test_ratings_file = './dataset/ml-100k-orginal/ua.test'

    # 获取训练集和测试集
    x_train, x_test, y_train, y_test = get_train_test_split(
        train_ratings_file, test_ratings_file, user_file, item_file
    )

    user_features_train, item_features_train = x_train
    user_features_test, item_features_test = x_test
    

    # 输出数据形状
    print("user_features_train shape:", user_features_train.shape)
    print("item_features_train shape:", item_features_train.shape)
    print("user_features_test shape:", user_features_test.shape)
    print("item_features_test shape:", item_features_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)