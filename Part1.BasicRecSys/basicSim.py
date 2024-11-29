import numpy as np
def CN(set1,set2):
    return len(set1&set2)


def Jaccard(set1,set2):
    return len(set1&set2)/len(set1|set2)


def cos4vector(v1,v2):
    return (np.dot(v1,v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cos4set(set1,set2):
    return len(set1&set2)/(len(set1)*len(set2))**0.5

def pearson(v1,v2):
    v1_mean = np.mean(v1)
    v2_mean = np.mean(v2)
    return np.dot(v1-v1_mean,v2-v2_mean) / (np.linalg.norm(v1-v1_mean) * np.linalg.norm(v2-v2_mean))


def pearsonSimple(v1,v2):
    v1 -= np.mean(v1)
    v2 -= np.mean(v2)
    return cos4vector(v1,v2)