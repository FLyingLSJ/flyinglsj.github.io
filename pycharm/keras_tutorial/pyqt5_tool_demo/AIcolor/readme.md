- pyqt5_demo.py: 基础版本，连接、注册按钮成功实现
- pyqt5_demo_v1.py : 修改了 pyqt5_demo.py 存在的 bug(服务器连接不上的问题)
- pyqt5_demo_v2.py ：增加了断开连接按键，增加了服务端异常时的处理情况，但效果不是很好
- pyqt5_demo_v3.py：增加按键灰色
- pyqt5_demo_v4.py：增加连接成功以后，地址输入框不可编辑
- pyqt5_demo_v5.py: 增加获取图片文件夹函数，开始工作按钮
- pyqt5_demo_v6.py: 增加数据接收，但是存在异常，准备在下一版本中使用进程解决 
- pyqt5_demo_v7.py: 把连接和注册事件写在一个函数里，添加了获取模型的函数（未解决上一版本存在的并发问题）
- pyqt5_demo_v8.py: 桌面改成终端，成功
- pyqt5_demo_v9.py: 更改了模型读取的位置，预测速度更快，接下来主要调整各种变量的读取，如图片的格式，图片的位置（完成）、模型的位置（完成），模型的输入图片大小等，压缩成 exe 的时候，外部文件处理问题
- pyqt5_demo_v10.py: 从配置文件读取模型，图片类型，模型大小未设置
- pyqt5_demo_v11.py: 优化程序，将部分功能函数化，加上发回服务端的功能，exe 转化问题，加上服务端关闭后重新连接功能                    

以上基于第一版通信协议

以下基于第二版通信协议
AIColor_v1：修改完通信协议的第一版，基本实现
AIColor_v2：第二版，模型自动检测模型的输入


- 
## py2exe 相关
1. https://pypi.org/project/auto-py-to-exe/
2. https://zhuanlan.zhihu.com/p/38659588


pyinstaller [opts] yourprogram.py 
参数含义：

-F	指定打包后只生成一个 exe 格式的文件 (建议写上这个参数)

-D	–onedir 创建一个目录，包含 exe 文件，但会依赖很多文件（默认选项）

-c	–console, –nowindowed 使用控制台，无界面 (默认)

-w	–windowed, –noconsole 使用窗口，无控制台

-p	添加搜索路径，让其找到对应的库。

-i	改变生成程序的 icon 图标 (比如给女朋友写的程序，换个好看的图标，默认的很丑)

