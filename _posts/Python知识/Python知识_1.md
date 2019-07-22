- 变量定义：大小写字母、数字、下划线且开头不能使用数字，大小写敏感
- type isinstance 判断数据类型
- 列表索引 [start, stop, step] 且索引的最后位置不在返回的结果中
- 元组不可改变
- 字典：哈希映射、键值对、键唯一
- 创建集合 set 有两种方法：set() 和  {1, 2} 但是要创建空集合只能使用 set() 因为 {} 代表字典
- 文件读取尽量使用 with open 防止出错
  - **read** 打开 & **读取**
    –`r`：打开指定文件，只用于 `reading`。文件的指针在开头。python 的默认模式。若无指定文件则报错
    –·`rb`：以二进制执行的 `r`；
  - **write** 打开 & **覆盖**
    – `w`：打开指定文件，只用于 `writing`。如果文件存在，则先删除已有数据，如果不存在，则创建；
    – `wb`：以二进制执行的 `w`；
  - **append** 打开 & **添加**
    – `a`：打开指定文件，用于 `appending`。如果文件存在，指针放在结尾，如果文件不存在，则创建；
    –`ab`：以二进制执行的 `a`；
  - `+`
    – `r+` / `rb+`：`reading` & `writing`。在 `r` / `rb+` 的基础上多了 `writing`。
    – `w+` / `wb+`：`writing` & `reading`。在 `w+` / `wb+` 的基础上多了 `reading`。
    – `a+` / `ab+`：`appending` & `reading`。在 `a+` / `ab+` 的基础上多了 `reading`。
- <https://matplotlib.org/gallery.html>



