import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_ml100k_user_item_features(user_path, item_path):
    """
    加载并处理 ml-100k 数据集的用户和物品特征。

    参数:
    - user_path: u.user 文件路径
    - item_path: u.item 文件路径

    返回:
    - user_features: 编码后的用户特征
    - item_features: 物品特征
    """
    # 加载用户数据
    user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_data = pd.read_csv(user_path, sep='|', names=user_columns, engine='python')
    user_data['age_group'] = user_data['age'] // 10  # 对年龄按每10岁分组


    # One-hot 编码用户特征
    user_encoder = OneHotEncoder()
    user_features = user_encoder.fit_transform(
        user_data[['age_group', 'gender', 'occupation']].astype(str)
    ).toarray()


    # 加载物品数据
    item_columns = ['item_id', 'title', 'release_date', 'video_release_date',
                    'IMDb_url'] + [f'feature_{i}' for i in range(19)]
    item_data = pd.read_csv(item_path, sep='|', names=item_columns, engine='python', encoding='ISO-8859-1')


    # 直接使用物品特征作为特征输入（无需再进行 One-hot 编码）
    item_features = item_data[[f'feature_{i}' for i in range(19)]].to_numpy()

    return user_features, item_features



def process_ml100k_ratings_data(ratings_path, user_features, item_features):
    """
    根据评分数据组合用户和物品特征。

    参数:
    - ratings_path: 评分文件路径
    - user_features: 预处理好的用户特征
    - item_features: 预处理好的物品特征

    返回:
    - features: 编码后的用户和物品组合特征
    - labels: 评分数据作为标签
    """
    # 加载评分数据
    ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_data = pd.read_csv(ratings_path, sep='\t', names=ratings_columns, engine='python')

    # 根据用户和物品 ID 提取特征
    user_ids = ratings_data['user_id'] - 1  # 将 ID 转为索引，从 0 开始
    item_ids = ratings_data['item_id'] - 1
    labels = ratings_data['rating'].values  # 获取评分

    # 合并用户和物品的特征
    features = np.hstack([user_features[user_ids], item_features[item_ids]])

    return features, labels


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
    - x_train: 训练特征
    - x_test: 测试特征
    - y_train: 训练标签
    - y_test: 测试标签
    """
    # 加载用户和物品特征
    user_features, item_features = load_ml100k_user_item_features(user_path, item_path)

    # 加载训练集
    x_train, y_train = process_ml100k_ratings_data(train_ratings_path, user_features, item_features)

    # 加载测试集
    x_test, y_test = process_ml100k_ratings_data(test_ratings_path, user_features, item_features)

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

    # 输出数据形状
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
