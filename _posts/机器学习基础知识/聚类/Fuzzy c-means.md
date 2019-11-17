---
layout: post
title: 聚类算法--模糊 C 均值聚类
date: 2019-10-10
tag: 聚类 机器学习
---



### Fuzzy c-means clustering（模糊 C 均值聚类）

模糊逻辑原理可以用来对多维数据进行聚类，为每个聚类中心中的每一个点指定一个 0-100% 的分数？对比传统的强硬的给每个点指定一个清晰的、准确的标签相比，FCM 更强大。

Fuzzy c-means 通过 `skfuzzy.cmeans` 完成，并且可以根据输出的模型对新的数据进行预测 `skfuzzy.cmeans_predict` 

 

Fuzzy c-means clustering
====
安装：`pip install -U scikit-fuzzy` (无法安装可能是网络原因)

关键函数解释：
`skfuzzy.cluster.cmeans(data, c, m, error, maxiter, init=None, seed=None)`
- 输入：
    - data: 聚类数据 (S, N)  N 个样本 S 维特征
    - c: 聚类的数目
    - m: 模糊化参数
    - error: 算法停止时的误差
    - maxiter: 最大迭代次数
    - init: 初始化矩阵
    - seed: 随机种子，主要用于调试
- 输出：
    - cntr : (S, c) 聚类中心
    - u : 2d array, (S, N) ：最终的模糊 c 矩阵
    - u0 : 2d array, (S, N) ：Initial guess at fuzzy c-partitioned matrix (either provided init or random guess used if init was not provided).
    - d : 2d array, (S, N)：最终的欧氏距离矩阵
    - jm : 1d array, length P：目标函数的历史记录值
    - p : int：运行的迭代次数
    - fpc : float：最终的模糊划分系数，可以根据此值来选择合适的聚类数目，越大越好
- 说明：
    - 算法的主要参数是 c 和 m ,即分类是数目和模糊化参数


```python
# 生成数据
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# Define three cluster centers
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i))

# Visualize the test data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
             color=colors[label])
ax0.set_title('Test data: 200 points x3 clusters.')

```


```python
# 
# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()
```


```python
# 
fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")
```


```python
# Regenerate fuzzy model with 3 cluster centers - note that center ordering
# is random in this clustering algorithm, so the centers may change places
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
    alldata, 3, 2, error=0.005, maxiter=1000)

# Show 3-cluster model
fig2, ax2 = plt.subplots()
ax2.set_title('Trained model')
for j in range(3):
    ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
             alldata[1, u_orig.argmax(axis=0) == j], 'o',
             label='series ' + str(j))
ax2.legend()
```


```python
# Generate uniformly sampled data spread across the range [0, 10] in x and y
newdata = np.random.uniform(0, 1, (1100, 2)) * 10

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.005, maxiter=1000)

# Plot the classified uniform data. Note for visualization the maximum
# membership value has been taken at each point (i.e. these are hardened,
# not fuzzy results visualized) but the full fuzzy result is the output
# from cmeans_predict.
cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization

fig3, ax3 = plt.subplots()
ax3.set_title('Random points classifed according to known centers')
for j in range(3):
    ax3.plot(newdata[cluster_membership == j, 0],
             newdata[cluster_membership == j, 1], 'o',
             label='series ' + str(j))
ax3.legend()

plt.show()
```


参考资料：

- https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html

- https://blog.csdn.net/zjsghww/article/details/50922168

  