2015.4.10 

### 标题

VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION 

针对大尺度图片识别的深度卷积网络

### 摘要

主要研究是卷积网络深度对大尺度图片识别的精度的影响。使用小的卷积核（3x3）的架构，对增加网络深度进行全面的评估，表面将深度增加到 16-19 层，可以显著改进先前的网络结构。该研究主要基于 ImageNet Challenge 2014 竞赛的提交结果，VGG 取得了定位第一和分类第二的成绩。同时该论文展示了该模型推广到其他数据集的效果，取得了当前最好的结果，该论文开放了两个版本的 VGG 模型（VGG16 和 VGG19）

### 基本内容

VGG 不同层网络的输入都是 224×224×3 

VGG16 有 13 层卷积 3 个全连接层，全部采用 3×3 卷积核，步长为 1 和 2×2 最大池化核，步长为 2

![image](https://ws1.sinaimg.cn/large/acbcfa39gy1g6nbg69c3sj20ld0l4ada.jpg)

![image](https://ws3.sinaimg.cn/large/acbcfa39gy1g6nbreymgxj20o903uwf0.jpg)

### 结论

文章的工作是评估非常深的网络用于大尺度图像分类的影响，结果表明，深度有利于图像分类的精度，再次证明了深度在视觉领域的重要性

![image](https://wx1.sinaimg.cn/large/acbcfa39gy1g6nb5an74qj20m84azap4.jpg)