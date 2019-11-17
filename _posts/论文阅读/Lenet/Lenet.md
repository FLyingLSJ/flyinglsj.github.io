```python
"""
Lenet Model
卷积的卷积核都为 5×5 步长 stride=1 
输入是 32×32 
-> 6@28*28（卷积C1）   参数：5×5×6+6 =156
-> 6@14*14（池化S2）   参数：偏移量参数  2×6
-> 16@10*10（卷积C3）  参数：5×5×6×16+16 = 2416  # 这里与原始的 LeNet 网络有区别
-> 16@5*5（池化S4）    参数：偏移量参数  2×16
-> 120@1*1（卷积C5）当然，这里也可以认为是全连接层   参数：5×5×16×120+120 = 48120
-> 84（全连接F6） 这个 84 的选取有个背景：与 ASCII 码表示的 7×12 的位图大小相等  参数：120×84
-> 10(输出类别数)  参数：84×10
"""

class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False): # 可以适用单通道和三通道的图像
        """
        num_classes: 分类的数量
        grayscale：是否为灰度图
        """
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes
        
        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3
        
        # 卷积神经网络
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2)   # 原始的模型使用的是 平均池化
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84), 
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x) # 输出 16*5*5 特征图
        x = torch.flatten(x, 1) # 展平 （1， 16*5*5）
        logits = self.classifier(x) # 输出 10
        probas = F.softmax(logits, dim=1)
        return logits, probas
        
        
    
num_classes = 10  # 分类数目
grayscale = True  # 是否为灰度图
data = torch.rand((1, 1, 32, 32))
model = LeNet5(num_classes, grayscale)
logits, probas = model(data)
logits, probas
```

---
```python
(tensor([[-0.1061, -0.0565, -0.0419,  0.0500,  0.0125,  0.0068, -0.1552,  0.0803,
          -0.0061, -0.1017]], grad_fn=<AddmmBackward>),
 tensor([[0.0926, 0.0973, 0.0988, 0.1082, 0.1043, 0.1037, 0.0882, 0.1116, 0.1024,
          0.0930]], grad_fn=<SoftmaxBackward>))
```
---




![1](https://images.zsxq.com/FiVJhS9_IvJY1rE1wCYWhitzlodQ?e=1906272000&token=kIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zD:DxX6zR_4P3qgpGQF0zLtCpTDWiY=)

![img](https://images.zsxq.com/FhXlTrCdkcWbDHbDgeP6wkcanFNQ?e=1906272000&token=kIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zD:xvRrGreEyTlTIOhL49D-NvMyTOw=)

![	](https://images.zsxq.com/lneP_v2aIvuogzQ1N2d2IP4uhREs?e=1906272000&token=kIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zD:Mp9FApRRYtrmmEu8epqJ52NzHMo=)	