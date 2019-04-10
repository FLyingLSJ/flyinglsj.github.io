[参考](http://wiki.jikexueyuan.com/project/explore-python/Standard-Modules/argparse.html)

#### argparse 使用

- 创建 ArgumentParser() 对象
- 调用 add_argument() 方法添加参数
- 使用 parse_args() 解析添加的参数

```
# -*- coding: utf-8 -*-
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1000, type=int, help="train epochs")
args = parser.parse_args()

print(args.integer)
```

