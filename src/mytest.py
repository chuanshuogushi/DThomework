from sklearn.model_selection import train_test_split
import pandas as pd
from src.config import *


def predict(data):
    """预测结果，输入dataframe，输出修改后的dataframe，新增一列predict"""
    # 在第五列插入新列，作为预测结果
    col_name = data.columns.tolist()
    col_name.insert(5, 'Predict')
    ndata = data.reindex(columns=col_name)
    if lim == (0.01, 10):
        for i in range(len(ndata)):
            if ndata.iloc[i, 2] <= 1.9:
                ndata.iloc[i, 5] = 0
            elif ndata.iloc[i, 3] <= 1.6:
                ndata.iloc[i, 5] = 1
            else:
                ndata.iloc[i, 5] = 2
    if lim == (0.01, 5):
        for i in range(len(ndata)):
            if ndata.iloc[i, 2] <= 1.9:
                ndata.iloc[i, 5] = 0
            elif ndata.iloc[i, 3] <= 1.6 and ndata.iloc[i, 2] <= 4.9:
                ndata.iloc[i, 5] = 1
            else:
                ndata.iloc[i, 5] = 2
    if lim == (0.01, 1):
        for i in range(len(ndata)):
            if ndata.iloc[i, 2] <= 1.9:
                ndata.iloc[i, 5] = 0
            elif ndata.iloc[i, 3] <= 1.6:
                if ndata.iloc[i, 2] <= 4.9:
                    ndata.iloc[i, 5] = 1
                else:
                    if ndata.iloc[i, 0] > 6:
                        ndata.iloc[i, 5] = 2
                    elif ndata.iloc[i, 1] <= 2.2:
                        ndata.iloc[i, 5] = 2
                    else:
                        ndata.iloc[i, 5] = 1
            else:
                if ndata.iloc[i, 2] <= 4.8:
                    if ndata.iloc[i, 1] <= 3:
                        ndata.iloc[i, 5] = 2
                    else:
                        ndata.iloc[i, 5] = 1
                else:
                    ndata.iloc[i, 5] = 2
    return ndata


def cal_acc(data):
    """计算准确率，输入dataframe，输出准确率"""
    y_test_pre = predict(data)
    y_test = data['Label']
    num_test = len(data)
    acc = sum(y_test_pre.iloc[:, 5] == y_test)/num_test
    return acc


if __name__ == '__main__':
    # assert rand == 2
    # assert
    dataset = pd.read_csv("../data/iris.data")
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=rand)
    print('Train accuracy:', cal_acc(train_data))
    print('Test accuracy:', cal_acc(test_data))

