"""模式识别与机器学习
决策树-iris数据集分类-ID3算法"""
import pandas as pd
from math import log2
from sklearn.model_selection import train_test_split
import numpy as np
from src import treePlot
from src.mytest import cal_acc
from src.config import *


def cal_entropy(data):
    """信息熵计算
    :param data:数组形式的数据，为Label数组
    :return 信息熵"""
    num = len(data)
    label_num = {}
    probability = {}  # 存储每个类的概率
    for a in data:
        if a not in label_num.keys():
            label_num[a] = 0
            probability[a] = 0
        label_num[a] += 1
    for a in set(data):
        probability[a] = label_num[a]/num
    entropy = 0
    for p in probability.values():
        entropy = entropy - p * log2(p)
    return entropy


def cal_df_entropy(data):
    """
    计算Dataframe格式的信息熵
    :param data:数据集，Dataframe形式
    :return:entropy 熵
    """
    array = data['Label']
    entropy = cal_entropy(array)
    return entropy


def cut_data(data, standard):
    """
    根据分离值切割数据
    :param data:Dataframe格式的数据
    :param standard:分离标准，包括特征名和特征值
    :return 分离后2个数据集
    """
    small_data = data.loc[data[standard['name']] <= standard['value']]
    big_data = data.loc[data[standard['name']] > standard['value']]
    return small_data, big_data


def cal_label_num(data):
    """
    统计三种类别的个数
    :param data: DataFrame格式数据集
    :return 数量最多的种类编号
    """
    num = [0, 0, 0]
    for i in range(3):
        num[i] = len(data[data['Label'] == i])
    return num.index(max(num))


def judge_stop(min_sample, mat1, mat2):
    """判断是否停止分类过程
    :param min_sample: 临界值
    :param mat1:数据集1
    :param mat2:数据集2
    :return 是否停止分类"""
    if (np.shape(mat1)[0] < min_sample) or (np.shape(mat2)[0] < min_sample):
        return True
    else:
        return False


def get_average(list):
    sum = 0
    for item in list:
        sum += item
    return sum/len(list)


def make_decision(data, limit):
    """根据数据集和阈值完成决策节点生成
    :param data: Dataframe格式数据
    :param limit: 阈值,包括信息增益和数量两种阈值"""
    chosen_f = {'name': 0, 'value': 0}
    f_names = ['A', 'B', 'C', 'D']  # 特征名字
    min_entropy = np.inf  # 最小信息熵
    # 分类结果全是一样的，返回样本数
    if len(set(data['Label'].tolist())) == 1:
        chosen_f = {'name': 'None', 'value': cal_label_num(data)}
        return chosen_f
    entropy = cal_df_entropy(data)  # 计算误差值
    for f_name in f_names:  # 遍历特征
        for f_v in data[f_name].tolist():  # 遍历特征所有值  #  平均值法：f_v_mean = Get_Average(f_vs)
            now_f = {'name': f_name, 'value': f_v}  # 当前分类标准
            s_mat, b_mat = cut_data(data, now_f)  # 分离数据集
            if judge_stop(limit[1], s_mat, b_mat) is True:
                continue
            tem_entropy = cal_df_entropy(s_mat) + cal_df_entropy(b_mat)  # 计算以当下分割方式的信息熵
            if tem_entropy < min_entropy:
                min_entropy = tem_entropy
                chosen_f = now_f
    # 信息增益和阈值比较，IG过小则返回样本数
    if (entropy - min_entropy) < limit[0]:
        chosen_f = {'name': 'None', 'value': cal_label_num(data)}
        return chosen_f
    s_mat, b_mat = cut_data(data, chosen_f)
    # 分类数小于阈值，认为达到分类要求，返回样本数
    if judge_stop(limit[1], s_mat, b_mat) is True:
        chosen_f = {'name': 'None', 'value': cal_label_num(data)}
        return chosen_f
    # 得到选择的特征名和特征值
    return chosen_f


def tree_builder(data, limit):
    """
    建立决策树
    :param data:Dataframe格式数据
    :param limit:阈值
    :return 字典格式的树
    """
    chosen_f = make_decision(data, limit)
    # IG或分类数小于阈值，返回分类出的样本数
    if chosen_f['name'] == 'None':
        return chosen_f['value']
    sub_tree = {}
    s_mat, b_mat = cut_data(data, chosen_f)
    # 建立决策树当前点
    sub_tree[chosen_f['name']] = {}
    # 建立决策树两个分支
    sub_tree[chosen_f['name']]['<=' + str(chosen_f['value']) + ' : ' + str(len(s_mat))] = tree_builder(s_mat, limit)
    sub_tree[chosen_f['name']]['>' + str(chosen_f['value']) + ' : ' + str(len(b_mat))] = tree_builder(b_mat, limit)
    return sub_tree


if __name__ == '__main__':
    iris_data = pd.read_csv("../data/iris.data")
    iris_data.sample(frac=1.0, replace=True)  # 打乱样本
    # 划分数据集
    train_data, test_data = train_test_split(iris_data, test_size=test_size, random_state=rand)
    print('Num of Trainset:', len(train_data))
    print('Num of Testset:', len(test_data))
    # 设定超参数
    print('minimum information gain:', lim[0])
    print('minimum number of samples:', lim[1])
    # 开始训练
    mytree = tree_builder(train_data, limit=lim)
    print('-'*8, 'Training Begin', '-'*8)
    print('My Decision Tree: ')
    print(mytree)
    print('-'*8, 'Training Done', '-'*8)
    # 画出决策树
    treePlot.createPlot(mytree)
    # 测试效果
    print('Train accuracy:', cal_acc(train_data))
    print('Test accuracy:', cal_acc(test_data))
