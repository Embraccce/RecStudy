import pandas as pd
import numpy as np

# 读取数据文件
ratings_df = pd.read_csv('./dataset/ml-latest-small/ratings.csv')

# 按照用户ID和时间戳排序数据
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
ratings_df = ratings_df.sort_values(by=['userId', 'timestamp'])

# 初始化一个列表，用于保存生成的序列
sequences = []

# 遍历每个用户，生成对应的序列
for user_id, group in ratings_df.groupby('userId'):
    # 将该用户的电影ID和评分按时间顺序排序
    movie_ids = group['movieId'].values
    ratings = group['rating'].values
    
    # 初始化一个列表用于保存该用户的最近喜欢的物品序列
    sequence = []

    # 遍历该用户的所有电影，生成序列
    for i in range(len(movie_ids)):
        # 获取最近5个电影ID（前5个）
        current_movie = movie_ids[i]
        
        # 判断用户是否点击该电影（评分大于等于4为点击）
        label = 1 if ratings[i] >= 4 else 0
        # 如果sequence中已经有5个历史物品ID，准备生成序列
        if len(sequence) == 5:
            # 生成序列，将当前电影和评分（标签）加入
            sequences.append(sequence + [current_movie, label])
            # 保留最近的5个物品ID
            sequence = sequence[1:]  
        if label == 1:
            sequence.append(current_movie)

# 将结果保存为CSV文件
sequences_df = pd.DataFrame(sequences)

# 设置列名
sequences_df.columns = ['movie1', 'movie2', 'movie3', 'movie4', 'movie5', 'movie6', 'label']

# 去重，删除重复的序列
sequences_df = sequences_df.drop_duplicates()

# 将结果保存到文件中
sequences_df.to_csv('./dataset/ml-latest-small/sequences.csv', index=False)
