这个包不能接收太长的数据

[TOC]

### Websocket Server

一个无需外部依赖的 Python 小型的 websocket server 

- 支持 python2、python3
- 简洁简单的 API
- 多客户端
- 无需依赖

此项目不支持如 SSL 等高级特性。该项目的重点主要是使运行websocket 服务器来进行原型设计、测试或为应用程序生成GUI变得容易。

### Installation

1. 直接下载 websocket_server.py 文件放在所需要的项目即可

2. ```
   pip install git+https://github.com/Pithikos/python-websocket-server 
   ```

3. ```
   pip install websocket-server 可能不是最新的版本
   ```

### API

`WebsocketServer` 的属性和方法

```
server = WebsocketServer(13254, host='127.0.0.1', loglevel=logging.INFO)
```

- 方法

  | 方法                      | 描述                                                       | 输入            | 返回 |
  | ------------------------- | ---------------------------------------------------------- | --------------- | :--: |
  | set_fn_new_client()       | 设置一个回调函数，该函数将为连接到我们的每个新客户机调用   | 函数            |  无  |
  | set_fn_client_left()      | 设置一个回调函数，该函数将为每个与我们断开连接的客户机调用 | 函数            |  无  |
  | set_fn_message_received() | 设置当客户端发送消息时将调用的回调函数                     | 函数            |  无  |
  | send_message()            | 向特定的客户端发送消息。消息是一个简单的字符串。           | client, message |  无  |
  | send_message_to_all()     | 向所有连接的客户机发送消息。消息是一个简单的字符串。       | message         |  无  |

- 回调函数

|                         | 描述             |          参数           |
| ----------------------- | ---------------- | :---------------------: |
| set_fn_new_client       | 新的客户端连接   |     client, server      |
| set_fn_client_left      | 客户端断开       |     client, server      |
| set_fn_message_received | 当客户端发送消息 | client, server, message |





参考资料：

- https://pypi.org/project/websocket-server/
- https://github.com/Pithikos/python-websocket-server

