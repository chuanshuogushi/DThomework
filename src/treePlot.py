# _*_ coding: UTF-8 _*_

import matplotlib.pyplot as plt

"""绘决策树的函数"""
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 定义分支点的样式
leafNode = dict(boxstyle="round4", fc="0.8")  # 定义叶节点的样式
arrow_args = dict(arrowstyle="<-")  # 定义箭头标识样式


# 计算树的叶子节点数量
def getNumLeafs(my_tree):
    num_leafs = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += getNumLeafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


# 计算树的最大深度
def getTreeDepth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + getTreeDepth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


# 画出节点
def plotNode(node_txt, center_pt, parent_pt, node_type):
    createPlot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                            xytext=center_pt, textcoords='axes fraction', va="center", ha="center",
                            bbox=node_type, arrowprops=arrow_args)


# 标箭头上的文字
def plotMidText(cntr_pt, parent_pt, txt_string):
    lens = len(txt_string)
    x_mid = (parent_pt[0] + cntr_pt[0]) / 2.0 - lens * 0.002
    y_mid = (parent_pt[1] + cntr_pt[1]) / 2.0
    createPlot.ax1.text(x_mid, y_mid, txt_string)


def plotTree(my_tree, parent_pt, node_txt):
    num_leafs = getNumLeafs(my_tree)
    depth = getTreeDepth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (plotTree.x0ff +
              (1.0 + float(num_leafs)) / 2.0 / plotTree.totalW, plotTree.y0ff)
    plotMidText(cntr_pt, parent_pt, node_txt)
    plotNode(first_str, cntr_pt, parent_pt, decisionNode)
    second_dict = my_tree[first_str]
    plotTree.y0ff = plotTree.y0ff - 1.0 / plotTree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plotTree(second_dict[key], cntr_pt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0 / plotTree.totalW
            plotNode(second_dict[key],
                     (plotTree.x0ff, plotTree.y0ff), cntr_pt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff)
                        , cntr_pt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0 / plotTree.totalD


def createPlot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(intree))
    plotTree.totalD = float(getTreeDepth(intree))
    plotTree.x0ff = -0.5 / plotTree.totalW
    plotTree.y0ff = 1.0
    plotTree(intree, (0.5, 1.0), '')
    plt.title('My Decision Tree', x=0.1)
    # plt.text(0.1, 1, "write anything")
    plt.show()


def retrieveTree(i):
    # 预先设置树的信息
    list_of_tree = [{'no surfacing': {0: 'no', 1: {'flipper': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flipper': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
                  {'a1': {0: 'b1', 1: {'b2': {0: {'c1': {0: 'd1', 1: 'd2'}}, 1: 'c2'}}, 2: 'b3'}}]
    return list_of_tree[i]
