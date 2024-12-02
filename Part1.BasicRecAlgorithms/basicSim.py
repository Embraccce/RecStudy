import numpy as np
def CN(set1, set2):
    """
    计算两个集合的交集元素数量。

    此函数接收两个集合作为参数，返回这两个集合交集的元素数量。
    交集是指两个集合中都包含的元素集合。

    参数:
    set1 -- 第一个集合
    set2 -- 第二个集合

    返回值:
    交集元素的数量
    """
    return len(set1 & set2)


def Jaccard(set1, set2):
    """
    计算两个集合的Jaccard相似度。
    
    Jaccard相似度是用来衡量两个集合交集大小与并集大小的比例，用于表示两个集合的相似度。
    
    参数:
    set1 -- 第一个集合
    set2 -- 第二个集合
    
    返回值:
    两个集合的Jaccard相似度
    """
    # 计算并返回两个集合的交集长度除以并集长度
    return len(set1 & set2) / len(set1 | set2)


def cos4vector(v1, v2):
    """
    计算两个向量之间的夹角余弦值。
    
    该函数用于接收两个向量v1和v2，计算它们之间的夹角余弦值。
    夹角余弦值是通过点积和向量的模长来计算的，能够反映两个向量的相似度。
    
    参数:
    v1: 一个向量，表示第一个输入向量。
    v2: 一个向量，表示第二个输入向量。
    
    返回值:
    返回两个向量之间的夹角余弦值。
    """
    # 使用numpy库的dot函数计算两个向量的点积
    # 使用numpy库的linalg.norm函数计算两个向量的模长
    # 将点积除以模长的乘积，得到夹角余弦值
    return (np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cos4set(set1, set2):
    """
    计算两个集合的余弦相似度。
    
    余弦相似度是通过计算两个集合交集的大小与两个集合大小的几何平均值的比值来衡量集合的相似度。
    这个函数的目的是为了提供一个简单的度量两个集合在元素组成上的相似程度的方法。
    
    参数:
    set1 -- 第一个集合
    set2 -- 第二个集合
    
    返回:
    返回两个集合的余弦相似度，返回值是一个浮点数，范围在0到1之间，值越接近1表示两个集合越相似。
    """
    # 计算两个集合的交集并取其长度
    intersection_len = len(set1 & set2)
    # 计算两个集合大小的几何平均值
    geometric_mean = (len(set1) * len(set2)) ** 0.5
    # 计算并返回余弦相似度
    return intersection_len / geometric_mean

def pearson(v1, v2):
    """
    计算两个向量v1和v2的皮尔逊相关系数。
    
    参数:
    v1: 一维向量，表示第一个变量的观测值。
    v2: 一维向量，表示第二个变量的观测值。
    
    返回值:
    返回皮尔逊相关系数，其值介于-1到1之间，表示两个变量的线性相关程度。
    """

    # 计算向量v1的均值
    v1_mean = np.mean(v1)
    # 计算向量v2的均值
    v2_mean = np.mean(v2)
    # 计算并返回皮尔逊相关系数
    return np.dot(v1 - v1_mean, v2 - v2_mean) / (np.linalg.norm(v1 - v1_mean) * np.linalg.norm(v2 - v2_mean))


def pearsonSimple(v1, v2):
    """
    计算两个向量的皮尔逊相关系数的简单方法。

    参数:
    v1: 一维数组，表示第一个向量。
    v2: 一维数组，表示第二个向量。

    返回值:
    返回两个向量的皮尔逊相关系数，该系数反映了两个向量的线性相关程度。
    """
    # 将向量v1减去其均值，以便计算差值的标准化
    v1 -= np.mean(v1)
    # 将向量v2减去其均值，达到标准化的目的
    v2 -= np.mean(v2)
    # 使用标准化后的向量，调用cos4vector函数计算并返回简化皮尔逊相关系数
    return cos4vector(v1, v2)