系统：WIN7

显卡型号：GeForce GTX 660 https://developer.nvidia.com/cuda-gpus

Visual Studio: https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_windows/ 先安装 VS 再安装 CUDA

CUDA

cuDNN

教程：https://www.twblogs.net/a/5c8df8bfbd9eee35cd6b0ea3

[码农的晋级之路，如何组装一台AI深度学习的工作站？](https://www.jd.com/phb/zhishi/cedb1514a670b829.html)

[深度学习工作站搭建全过程](https://blog.csdn.net/u011636440/article/details/72802689)

[配置网站](https://pcpartpicker.com/)



### 硬件配置

1. GPU：性价比、显存、散热

- 性价比：使用16bit的RTX 2070或者RTX 2080 Ti性价比更高，32bit GTX 1070、GTX 1080或者1080 Ti也是不错的选择
- 显存
  - 如果想在研究中追求最高成绩：显存>=11 GB；
  - 在研究中搜寻有趣新架构：显存>=8 GB；
  - 其他研究：8GB；
  - Kaggle竞赛：4~8GB；
  - 创业公司：8GB（取决于具体应用的模型大小）
  - 公司：打造原型8GB，训练不小于11GB
- 散热：多块 GPU 注意散热

2. 内存：时钟频率、容量

- 时钟频率：内存频率和数据转移到显存的速度无关，提高频率最多只能有3%的性能提升，你还是把钱花在其他地方吧！
- 内存容量：内存大小不会影响深度学习性能，但是它可能会影响你执行 GPU 代码的效率。内存容量大一点，CPU 就可以不通过磁盘，直接和 GPU 交换数据。
- 内存 ≥ 显存最大的那块 GPU 的 RAM；内存不用太大，用多少买多少

3. CPU：更需要关注的是CPU和主板组合支持同时运行的GPU数量。

- PCIe通道：只有 GPU 数量较大时，这个参数影响才会比较重要，比较少时（少于 4 个）不用过分关注此参数
- CPU 核心数：
  - CPU 作用：（1）启动 GPU 函数调用（2）执行 CPU 函数。
  - CPU 在数据预处理中的作用：不同预处理策略，配置不一
    - 第一种是在训练时进行预处理：高性能的多核 CPU 能显著提高效率。建议每个 GPU 至少有 4 个线程，即为每个 GPU 分配两个 CPU 核心。
    - 第二种是在训练之前进行预处理：不需要非常好的 CPU。建议每个 GPU 至少有 2 个线程，即为每个 GPU 分配一个 CPU 核心。
  - 频率要大于 2GHz，CPU 要能支持你的 GPU 数量

4. 硬盘/固态硬盘（SSD）：固态硬盘推荐，SSD 程序启动和响应速度更快，大文件的预处理更是要快得多。顶配：***NVMe SSD***

5. 电源装置（PSU）：将电脑 CPU 和 GPU 的功率相加，再额外加上 10% 的功率算作其他组件的耗能，就得到了功率的峰值。
   - 举个例子，如果你有 4 个 GPU，每个功率为250瓦，还有一个功率为 150 瓦的 CPU，则需电源提供 4×250+150+100=1250 瓦的电量。
   - 即使一个 PSU 达到了所需瓦数，也可能没有足够的 PCIe 8-pin 或 6-pin 的接头，所以买的时候还要确保电源上有***足够多的接头接 GPU***。
   - 能效等级高的电源可以节省电
6. CPU 和 GPU 的冷却：散热不好会降低性能
   - 风冷散热：单个 GPU 可以使用，多个 GPU 靠空气散热效果不佳
     - 鼓风式的风扇将热空气从机箱背面推出，让凉空气进来：多个 GPU 彼此相邻时使用，低成本配置
     - 非鼓风式的风扇是在 GPU 中吸入空气达到冷却效果：
   - 水冷散热：成本比风冷高，适用于多个 GPU 相邻的情况，安静

### 结论

主板卡槽要够，不要互相阻挡

机箱长度要足够，特别是水冷时，空间要更大

为了效率显示器可以多配

### 他人配置

![](https://img-blog.csdn.net/20180705145135598?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0Mzc0MjEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)