import numpy as np
import matplotlib.pyplot as plt

# 设定每个策略在迭代0, 10, 20, 30, 40, 50时的模拟数据

# 生成正态分布数据的函数
def generate_data(mean, std, iterations, size=10):
    return np.random.normal(mean, std, (iterations, size))

# 设置每10次迭代的平均值和标准差
iterations = [0, 10, 20, 30, 40, 50]
mean_max = np.linspace(0, 230, num=6)
std_max = [15] * 6  # 假设MAX的数据波动较小

mean_ego = np.linspace(0, 200, num=6)
std_ego = [20] * 6  # 假设EGO的数据波动稍大

mean_rnd = np.linspace(0, 70, num=6)
std_rnd = [10] * 6  # 假设RND的数据波动小

mean_stdev = np.linspace(0, 100, num=6)
std_stdev = [12] * 6  # 假设STDEV的数据波动中等

# 生成数据
data_max = [generate_data(m, s, 1) for m, s in zip(mean_max, std_max)]
data_ego = [generate_data(m, s, 1) for m, s in zip(mean_ego, std_ego)]
data_rnd = [generate_data(m, s, 1) for m, s in zip(mean_rnd, std_rnd)]
data_stdev = [generate_data(m, s, 1) for m, s in zip(mean_stdev, std_stdev)]

# 转化为一维数组以便于绘图
data_max = [item.flatten() for item in data_max]
data_ego = [item.flatten() for item in data_ego]
data_rnd = [item.flatten() for item in data_rnd]
data_stdev = [item.flatten() for item in data_stdev]

# 绘制箱线图
fig, ax = plt.subplots()
bp = ax.boxplot(data_max + data_ego + data_rnd + data_stdev, patch_artist=True, labels=[
    "MAX 0", "MAX 10", "MAX 20", "MAX 30", "MAX 40", "MAX 50",
    "EGO 0", "EGO 10", "EGO 20", "EGO 30", "EGO 40", "EGO 50",
    "RND 0", "RND 10", "RND 20", "RND 30", "RND 40", "RND 50",
    "STDEV 0", "STDEV 10", "STDEV 20", "STDEV 30", "STDEV 40", "STDEV 50"
], medianprops=dict(color="black"), showmeans=True)

# 设置图形属性
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Simulated Data Distribution at Iterations for Each Strategy')
ax.set_ylabel('Number of Structures')
plt.show()

