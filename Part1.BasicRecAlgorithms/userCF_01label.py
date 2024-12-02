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


def knn4set_userCF(trainset,k,sim_method):
    """
    计算每个商品在给定训练集中基于相似度的 K 相似用户。

    参数：
        trainset (dict): 一个字典，键是用户 ID，值是该用户的物品集合。
        k (int): 每个用户的近邻数量。
        sim_method (function): 一个相似度计算方法，用于计算两个用户之间的相似度。该方法接受两个集合作为输入，返回一个相似度值。

    返回：
        dict: 一个字典，其中键是用户 ID，值是与该用户最相似的 K 个用户的 ID 列表。

    说明：
        1. 对于训练集中的每个用户 u1，遍历所有其他用户 u2 以计算相似度。
        2. 只对拥有共同物品的用户对（u1 和 u2）进行相似度计算。
        3. 忽略 u1 自身以及没有共同物品的用户。
        4. 计算相似度后，将每个 u1 的所有其他用户按相似度排序并选取前 K 个用户作为其近邻。
    """
    sims = {}
    for u1 in tqdm(trainset):
        ulist = []
        for u2 in trainset:
            if u1 == u2 or len(trainset[u1] & trainset[u2]) == 0:
                continue
            sim = sim_method(trainset[u1],trainset[u2])
            ulist.append((u2,sim))
        sims[u1] = [i[0] for i in sorted(ulist, key=lambda x:x[1],reverse=True)[:k]]
    return sims


def get_recommodations_by_userCF(user_sims, user_o_set):
    """
    基于用户协同过滤（userCF）推荐物品。
    
    参数：
        user_sims (dict): 用户相似度字典，其中键是用户 u，值是与该用户相似的用户列表。
        user_o_set (dict): 每个用户已拥有物品的集合，键是用户 u，值是该用户已拥有的物品集合。
        
    返回：
        dict: 用户推荐字典，其中键是用户 u，值是推荐给该用户的物品集合。
    
    功能：
        该函数遍历每个用户 u 的相似用户列表，并从相似用户中找到用户 u 尚未拥有的物品，
        将这些物品推荐给用户 u。
    """
    recommodations = collections.defaultdict(set)
    for u in user_sims:
        for sim_u in user_sims[u]:
            recommodations[u] |= (user_o_set[sim_u] - user_o_set[u])
    return recommodations


def trainUserCF( user_items_train, sim_method, user_all_items, k = 5 ):
    user_sims = knn4set_userCF( user_items_train, k, sim_method )
    recomedations = get_recommodations_by_userCF( user_sims, user_all_items )
    return recomedations


def evaluate_userCF(user_pos_item_test, user_neg_item_test, pred_set):
    """
    使用测试集评估用户协同过滤 (UserCF) 模型的性能。
    
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


if __name__ == "__main__":
    # 1.读取数据
    _, _, train_set, test_set = dataloader.read_triples(test_ratio=0.1)
    user_items_train, _, _, user_all_items = getSet(train_set)
    user_pos_item_test ,_,user_neg_item_test,_ = getSet(test_set)

    # 2.训练模型
    recomedations_by_userCF = trainUserCF( user_items_train, basicSim.cos4set, user_all_items, k=5 )

    # 3.模型评估
    evaluation_results = evaluate_userCF(
        user_pos_item_test=user_pos_item_test,
        user_neg_item_test=user_neg_item_test,
        pred_set=recomedations_by_userCF
    )

    # 4.打印评估结果
    print("UserCF Evaluation Results:")
    print(f"Average Recall: {evaluation_results['Average Recall']:.4f}")
    print(f"Average Precision: {evaluation_results['Average Precision']:.4f}")