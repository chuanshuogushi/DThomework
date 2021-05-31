test_size = 0.2  # 测试集所占比例
lim = (0.01, 5)  # 最小信息增益与最小样本数=1,5,10
rand = 1  # 随机数种子
assert test_size == 0.2
assert lim[0] == 0.01
assert rand == 1
assert lim[1] == 1 or 5 or 10
