[TOC]

### 问题

请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

[牛客网](https://www.nowcoder.com/practice/4060ac7e3e404ad1a894ef3e17650423?tpId=13&tqId=11155&tPage=1&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

### 解题思路

##### 基本思路

解算法题之前不要马上就开始写算法，要先理清一下思路

- 能否用图表的形式形象化
- 能否将其分解使其简单化
- 能否用举例使问题具体化

其次，设想几个测试案例，依据

- 边界条件
- 特殊输入（空值，空指针）
- 错误处理

本题中，我们可以使用题目的案例进行具体分析

并且设想几个测试用例

##### 具体思路

- 遍历整个字符串，遇到空格直接进行替换，暴力破解
- 测试用例
  - 空字符串
  - 空格在字符串开头、中间、末尾
  - 全是空格
  - 输入为一个空格
  - 连续两个空格
  - 不包含空格的字符串
- 寻找规律，用案例进行分析

### 解法

- 暴力破解

```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        if len(s) == 0:
            return ""
        return s.replace(" ", "%20")
```

- 寻找规律

![image](https://ws4.sinaimg.cn/large/acbcfa39ly1g78brswc78j20k006yt9y.jpg)

![image](https://wx4.sinaimg.cn/large/acbcfa39gy1g78bm4oqloj20k1049t9j.jpg)

我们考虑从往前进行处理，即从字符串的尾部开始，遇到空格再进行处理

```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        originalLength = len(s)
        if originalLength == 0: # 若字符串为空，则直接返回
            return "" 
        numberOfBlank = 0 
        for i in s:
            if i == " ":
                numberOfBlank += 1 # 统计空格的数量
                
        newLength = originalLength + numberOfBlank * 2 # 新字符串的长度
        s = list(s) # 因为在 python 中，字符串定义好了以后就不可以再更改了，故将其转成列表形式
        s = s+[None]*(numberOfBlank*2) 
        indexOfNew = newLength - 1 # 新字符串的索引
        indexOfOriginal = originalLength - 1  # 旧字符串的索引

        while (indexOfOriginal>= 0):
            if s[indexOfOriginal] == " ": # 遇到空格 
                for i in ["0", "2", "%"]:
                    s[indexOfNew] = i
                    indexOfNew -= 1 
                 
            else: # 非空格
                s[indexOfNew] = s[indexOfOriginal]
                indexOfNew -= 1 
                
            indexOfOriginal -= 1 
        return "".join(s)
                
             
def Test(result, excepted):
    if result == excepted:
        print("Passed. \n")
    else:
        print("Failed. \n")

def Test1(): # 空字符串
    print("Test1:")
    s = ""
    result = soluton.replaceSpace(s)
    Test(result, "")
    
 
def Test2(): # 空格在开头
    print("Test2:")
    s = " hello"
    result = soluton.replaceSpace(s)
    Test(result, "%20hello")

def Test3(): # 空格在中间
    print("Test3:")
    s = "hello world"
    result = soluton.replaceSpace(s)
    Test(result, "hello%20world")     
    
def Test4(): # 空格在末尾
    print("Test4:")
    s = "hello "
    result = soluton.replaceSpace(s)
    Test(result, "hello%20")

def Test5(): # 全是空格
    print("Test5:")
    s = "     "
    result = soluton.replaceSpace(s)
    Test(result, "%20%20%20%20%20")

def Test6(): # 输入为一个空格
    print("Test6:")
    s = " "
    result = soluton.replaceSpace(s)
    Test(result, "%20")

def Test7(): # 输入为连续两个空格
    print("Test7:")
    s = "hello  world"
    result = soluton.replaceSpace(s)
    Test(result, "hello%20%20world")    

def Test8(): # 不包含空格的字符串
    print("Test8:")
    s = "helloworld"
    result = soluton.replaceSpace(s)
    Test(result, "helloworld")      
    
soluton = Solution()
Test1()
Test2()
Test3()
Test4()
Test5()
Test6()
Test7()
Test8()
```



参考：

- (https://www.cnblogs.com/yanmk/p/9172064.html)
- https://github.com/zhedahht/CodingInterviewChinese2/blob/master/05_ReplaceSpaces/ReplaceSpaces.cpp

