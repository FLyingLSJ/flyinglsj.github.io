

```python
from pathlib import Path
now_path = Path.cwd()
home_path = Path.home()
now_path, home_path
```




    (WindowsPath('F:/jupyter/201907错误图片分析'), WindowsPath('C:/Users/Administrator'))



使用 `/` 实现对路径的拼接


```python
DIR_PATH = Path("f:/jupyter/") / '100'
DIR_PATH
```




    WindowsPath('f:/jupyter/100')



不需要往 open 里面输入文件路径


```python
DIR_PATH = Path("f:/jupyter/") / 'test.txt'
with DIR_PATH.open("r", encoding='utf-8') as f:
    data_1 = f.read()
data_2 = DIR_PATH.read_text(encoding='utf-8') # 作用同上
    
data_1, data_2
```




    ('pathlib 测试', 'pathlib 测试')



- .read_text(): 找到对应的路径然后打开文件，读成str格式。等同open操作文件的"r"格式。
- .read_bytes(): 读取字节流的方式。等同open操作文件的"rb"格式。
- .write_text(): 文件的写的操作，等同open操作文件的"w"格式。
- .write_bytes(): 文件的写的操作，等同open操作文件的"wb"格式。


```python
py_path = Path("test.ipynb")
py_path.resolve() # 返回文件的绝对路径
```




    WindowsPath('F:/jupyter/201907错误图片分析/test.ipynb')



文件路径的不同部分的选择
----
- .name: 可以获取文件的名字，包含拓展名。
- .parent: 返回上级文件夹的名字
- .stem: 获取文件名不包含拓展名
- .suffix: 获取文件的拓展名
- .anchor: 类似盘符的一个东西,


```python
now_path = Path.cwd() / "test.txt"
print("name",now_path.name)
print("stem",now_path.stem)
print("suffix",now_path.suffix)
print("parent",now_path.parent)
print("anchor",now_path.anchor)
```

    name test.txt
    stem test
    suffix .txt
    parent F:\jupyter\201907错误图片分析
    anchor F:\


移动和删除文件
---
- .replace() 移动文件可以指定移动后的文件名，如果文件存在则会覆盖。为避免文件可能被覆盖，最简单的方法是在替换之前测试目标是否存在。


```python
# 移动文件，并且指定目标文件的文件名
# 适合单个文件的操作
destination = Path.cwd() / "target.txt"
source = pathlib.Path.cwd() / "test.txt"
if not destination.exists():  # 判断目标文件是否存在，不存在才覆盖
    source.replace(destination)
```


```python
import pathlib

destination = pathlib.Path.cwd() / "target_2.txt" # 若目标文件存在，会出错
source = pathlib.Path.cwd() / "target.txt"
with destination.open(mode='xb') as fid:
    # xb表示文件不存在才操作
    fid.write(source.read_bytes())
```


```python
import pathlib
source = pathlib.Path.cwd() / "target.txt"
source.replace(source.with_suffix(".py"))  # 修改后缀并移动文件，即重命名
```

删除文件


```python
import pathlib

source = pathlib.Path.cwd() / "target.py"
source.unlink()
```

其他功能
----



```python
import pathlib
from collections import Counter
now_path = pathlib.Path.cwd()
gen = (i.suffix for i in now_path.iterdir())
print(Counter(gen))
```

    Counter({'': 5, '.jpg': 4, '.rar': 4, '.txt': 1, '.ipynb': 1})



```python
import pathlib
from  collections import Counter
gen =(p.suffix for p in pathlib.Path.cwd().glob('*.jpg'))
print(Counter(gen))
```

    Counter({'.jpg': 4})



```python
import pathlib
from collections import Counter


def tree(directory):
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(
            path.relative_to(directory).parts)  # 返回path相对于directory的路径。
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')


now_path = pathlib.Path.cwd()

if __name__ == '__main__':
    tree(now_path)
```

    + F:\jupyter\201907错误图片分析
        + .ipynb_checkpoints
            + test-checkpoint.ipynb
        + 27-1-1.jpg
        + 27-2-1.jpg
        + 27-3-1.jpg
        + 27-偏红原图
            + imagedata.jpg
            + result.jpg
        + 27-偏红原图.rar
        + 27-图片中其余色号对比图
            + 3163-1.jpg
            + 3163-2.jpg
            + 4863-1.jpg
            + 4863-2.jpg
            + 7154-1.jpg
            + 7154-2.jpg
        + 27-图片中其余色号对比图.rar
        + 27-对比图
            + 27-1-1.jpg
            + 27-1-2.jpg
            + 27-2-1.jpg
            + 27-2-2.jpg
            + 27-3-1.jpg
            + 27-3-2.jpg
        + 27-对比图.rar
        + result.jpg
        + target_2.txt
        + test.ipynb
        + 识别问题图片-16
            + 识别问题图片
                + A11564194226725443-c.jpg
                + A11564194226725443.jpg
                + A11564194298628936-c.jpg
                + A11564194298628936.jpg
        + 识别问题图片-16.rar



```python
import pathlib
now_path = pathlib.Path.cwd()
if __name__ == '__main__':
    print((now_path.parts)) # 
```

    ('F:\\', 'jupyter', '201907错误图片分析')

参考资料：

- https://mp.weixin.qq.com/s/1StudIEbdUFQPyxl117pxQ
- <https://docs.python.org/3/library/pathlib.html#basic-use>