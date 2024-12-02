import collections
import basicSim
import dataloader
import evaluate
from tqdm import tqdm
import numpy as np


def getSet(triples):
    """
    根据用户-物品-评分的三元组，生成用户和物品的多种映射集合。
    
    输入:
    - triples: list of tuples，格式为 (user, item, rating)，表示用户对物品的评分。
        - user: int 或 str，用户的唯一标识符
        - item: int 或 str，物品的唯一标识符
        - rating: int，评分，通常为 1（正反馈）或 0（负反馈）

    输出:
    - user_pos_items: dict，{用户: 正反馈物品集合}
        - 用户与其正反馈（rating = 1）的物品之间的映射。
    - item_users: dict，{物品: 用户集合}
        - 每个物品被评分过的用户集合。
    - user_neg_items: dict，{用户: 负反馈物品集合}
        - 用户与其负反馈（rating ≠ 1）的物品之间的映射。
    - user_all_items: dict，{用户: 全部物品集合}
        - 用户评分过的所有物品集合，无论是正反馈还是负反馈。
    
    逻辑:
    1. 初始化四个 `defaultdict`，分别存储正反馈、负反馈、所有评分过的物品，以及物品被评分的用户。
    2. 遍历输入的三元组列表：
        - 将物品添加到 `user_all_items` 中，表示用户评分过的所有物品。
        - 将用户添加到 `item_users` 中，表示物品被评分过的用户集合。
        - 根据评分（rating），将物品归入正反馈集合或负反馈集合。
    3. 返回四个字典。
    """
    # 用户的正反馈物品集合
    user_pos_items = collections.defaultdict(set)
    # 用户的负反馈物品集合
    user_neg_items = collections.defaultdict(set)
    # 用户评分过的所有物品集合
    user_all_items = collections.defaultdict(set)
    # 物品被评分的用户集合
    item_users = collections.defaultdict(set)
    
    # 遍历每个三元组 (user, item, rating)
    for u, i, r in triples:
        # 添加到用户的所有物品集合
        user_all_items[u].add(i)
        
        if r == 1:  # 正反馈
            user_pos_items[u].add(i)
            item_users[i].add(u)  # 这里重复添加无影响，因为是集合
        else:  # 负反馈
            user_neg_items[u].add(i)
    
    # 返回用户和物品之间的四种映射关系
    return user_pos_items, item_users, user_neg_items, user_all_items


def knn4set_itemCF(trainset, k, sim_method):
    """
    计算每个物品在给定训练集中基于相似度的 K 相似物品。

    参数：
        trainset (dict): 一个字典，键是物品 ID，值是该物品的用户集合。
        k (int): 每个物品的近邻数量。
        sim_method (function): 一个相似度计算方法，用于计算两个物品之间的相似度。该方法接受两个集合作为输入，返回一个相似度值。

    返回：
        dict: 一个字典，其中键是物品 ID，值是与该物品最相似的 K 个物品的 ID 列表。

    说明：
        1. 对于训练集中的每个物品 i1，遍历所有其他物品 i2 以计算相似度。
        2. 只对拥有共同用户的物品对（i1 和 i2）进行相似度计算。
        3. 忽略 i1 自身以及没有共同用户的物品。
        4. 计算相似度后，将每个 i1 的所有其他物品按相似度排序并选取前 K 个物品作为其近邻。
    """
    sims = collections.defaultdict(list)
    for i1 in tqdm(trainset):
        ilist = []
        for i2 in trainset:
            if i1 == i2 or len(trainset[i1] & trainset[i2]) == 0:
                continue
            sim = sim_method(trainset[i1], trainset[i2])
            ilist.append((i2,sim))
        # 选择 K 个最相似的物品
        sims[i1] = [i[0] for i in sorted(ilist, key=lambda x:x[1],reverse=True)[:k]]
    return sims


def get_recommodations_by_itemCF(item_sims, user_o_set):
    """
    基于物品协同过滤（ItemCF）推荐物品。

    参数：
        item_sims (dict): 物品相似度字典，其中键是物品 ID，值是与该物品相似的物品 ID 列表。
        user_o_set (dict): 每个用户已拥有物品的集合，键是用户 ID，值是该用户已拥有的物品集合。

    返回：
        dict: 用户推荐字典，其中键是用户 ID，值是推荐给该用户的物品集合。

    功能：
        该函数遍历每个用户及其已拥有的物品，为每个物品找到相似物品，并将用户尚未拥有的相似物品推荐给用户。
    """
    
    recommodations = collections.defaultdict(set)
    for u in user_o_set:
        for item in user_o_set[u]:
            if item in item_sims:
                recommodations[u] |= set(item_sims[item]) - user_o_set[u]
    return recommodations


def trainItemCF(item_users_train, sim_method, user_all_items, k=5):
    """
    训练Item-Based协同过滤模型并生成推荐结果。

    该函数首先使用给定的物品-用户交互数据计算物品间的相似度，然后基于这些相似度信息和用户的历史行为，
    为每个用户生成一个推荐物品列表。

    参数:
    item_users_train (dict): 物品-用户训练数据集，格式为{item_id: set(user_ids)}。
    sim_method (str): 计算物品相似度的方法，例如'cosine'、'jaccard'等。
    user_all_items (dict): 用户历史交互过的所有物品，格式为{user_id: set(item_ids)}。
    k (int): 为每个物品找到的最相似物品的数量，默认值为5。

    返回:
    dict: 用户的推荐物品列表，格式为{user_id: [recommended_item_ids]}。
    """
    # 计算物品之间的相似度
    item_sims = knn4set_itemCF(item_users_train, k, sim_method)
    
    # 根据物品相似度和用户历史行为生成推荐
    recomedations = get_recommodations_by_itemCF(item_sims, user_all_items)
    
    return recomedations


def evaluate_itemCF(user_pos_item_test, user_neg_item_test, pred_set):
    """
    使用测试集评估用户协同过滤 (ItemCF) 模型的性能。
    
    参数：
        user_pos_item_test: dict，测试集中每个用户的正反馈物品集合。
        user_neg_item_test: dict，测试集中每个用户的负反馈物品集合。
        pred_set: dict，模型为每个用户预测的推荐物品集合。

    返回：
        dict: 包含召回率 (Recall) 和精确率 (Precision) 的评估结果。
    """
    recall_scores = []
    precision_scores = []
    
    # 遍历每个用户
    for user in user_pos_item_test:
        # 获取该用户的测试集正样本和负样本
        test_pos_set = user_pos_item_test[user]
        test_neg_set = user_neg_item_test.get(user, set())
        
        # 获取模型为该用户的预测推荐集合
        pred_items = pred_set.get(user, set())
        
        # 计算召回率
        recall = evaluate.recall4Set(test_pos_set, pred_items)
        recall_scores.append(recall)
        
        # 计算精确率
        precision = evaluate.percision4Set(test_pos_set, test_neg_set, pred_items)
        precision_scores.append(precision)
    
    # 计算所有用户的平均召回率和精确率
    avg_recall = np.mean(recall_scores)
    avg_precision = np.mean(np.array([x for x in precision_scores if x is not None]))

    # 返回评估结果
    return {
        "Average Recall": avg_recall,
        "Average Precision": avg_precision
    }

if __name__ == '__main__':
    # 1.读取数据
    _, _, train_set, test_set = dataloader.read_triples(test_ratio=0.1)
    _, item_users_train, _, user_all_items = getSet(train_set)
    user_pos_item_test ,_,user_neg_item_test,_ = getSet(test_set)

    # 2.训练模型
    recomedations_by_itemCF = trainItemCF( item_users_train, basicSim.cos4set, user_all_items, k=5 )

    # 3.模型评估
    evaluation_results = evaluate_itemCF(
        user_pos_item_test=user_pos_item_test,
        user_neg_item_test=user_neg_item_test,
        pred_set=recomedations_by_itemCF
    )

    # 4.打印评估结果
    print("ItemCF Evaluation Results:")
    print(f"Average Recall: {evaluation_results['Average Recall']:.4f}")
    print(f"Average Precision: {evaluation_results['Average Precision']:.4f}")