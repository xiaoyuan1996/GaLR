# coding:utf-8
# Author:Zhiqiang Yuan
"""导入一些包"""
import os
import time, random
import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""                       打印一些东西                                    """
"""----------------------------------------------------------------------"""


# 打印列表按照竖行的形式
def print_list(list):
    print("++++++++++++++++++++++++++++++++++++++++++++")
    for l in list:
        print(l)
    print("++++++++++++++++++++++++++++++++++++++++++++")


# 打印字典按照竖行的形式
def print_dict(dict):
    print("++++++++++++++++++++++++++++++++++++++++++++")
    for k, v in dict.items():
        print("key:", k, "   value:", v)
    print("++++++++++++++++++++++++++++++++++++++++++++")


# 打印一些东西，加入标识符
def print_with_log(info):
    print("++++++++++++++++++++++++++++++++++++++++++++")
    print(info)
    print("++++++++++++++++++++++++++++++++++++++++++++")


# 打印标识符
def print_log():
    print("++++++++++++++++++++++++++++++++++++++++++++")


"""                           文件存储                                    """
"""----------------------------------------------------------------------"""


# 保存结果到json文件
def save_to_json(info, filename, encoding='UTF-8'):
    with open(filename, "w", encoding=encoding) as f:
        json.dump(info, f, indent=2, separators=(',', ':'))


# 从json文件中读取
def load_from_json(filename):
    with open(filename, encoding='utf-8') as f:
        info = json.load(f)
    return info


# 储存为npy文件
def save_to_npy(info, filename):
    np.save(filename, info, allow_pickle=True)


# 从npy中读取
def load_from_npy(filename):
    info = np.load(filename, allow_pickle=True)
    return info


# 保存结果到txt文件
def log_to_txt(contexts=None, filename="save.txt", mark=False, encoding='UTF-8', add_n=False):
    f = open(filename, "a", encoding=encoding)
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    elif isinstance(contexts, dict):
        tmp = ""
        for c in contexts.keys():
            tmp += str(c) + " | " + str(contexts[c]) + "\n"
        contexts = tmp
        f.write(contexts)
    else:
        if isinstance(contexts, list):
            tmp = ""
            for c in contexts:
                if add_n:
                    tmp += str(c) + "\n"
                else:
                    tmp += str(c)
            contexts = tmp
        else:
            contexts = contexts + "\n"
        f.write(contexts)

    f.close()


# 从txt中读取行
def load_from_txt(filename, encoding="utf-8"):
    f = open(filename, 'r', encoding=encoding)
    contexts = f.readlines()
    return contexts


"""                           字典变换                                   """
"""----------------------------------------------------------------------"""


# 键值互换
def dict_k_v_exchange(dict):
    tmp = {}
    for key, value in dict.items():
        tmp[value] = key
    return tmp


# 2维数组转字典
def d2array_to_dict(d2array):
    # Input: N x 2 list
    # Output: dict
    dict = {}
    for item in d2array:
        if item[0] not in dict.keys():
            dict[item[0]] = [item[1]]
        else:
            dict[item[0]].append(item[1])
    return dict


"""                             绘图                                     """
"""----------------------------------------------------------------------"""


# 绘制3D图像
def visual_3d_points(list, color=True):
    """
    :param list: N x (dim +1)
    N 为点的数量
    dim 为 输入数据的维度
    1 为类别， 即可视化的颜色  当且仅当color为True时
    """
    list = np.array(list)
    if color:
        data = list[:, :4]
        label = list[:, -1]
    else:
        data = list
        label = None

    # PCA降维
    pca = PCA(n_components=3, whiten=True).fit(data)
    data = pca.transform(data)

    # 定义坐标轴
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    if label is not None:
        color = label
    else:
        color = "blue"
    ax1.scatter3D(np.transpose(data)[0], np.transpose(data)[1], np.transpose(data)[2], c=color)  # 绘制散点图

    plt.show()


"""                           实用工具                                    """
"""----------------------------------------------------------------------"""


# 计算数组中元素出现的个数
def count_list(lens):
    dict = {}
    for key in lens:
        dict[key] = dict.get(key, 0) + 1
    dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    print_list(dict)
    return dict


# list 加法 w1、w2为权重
def list_add(list1, list2, w1=1, w2=1):
    return [l1 * w1 + l2 * w2 for (l1, l2) in zip(list1, list2)]
