---
layout: post
title: Github无法push问题
date: 2019-5-20
tag: Github
---

Github push  出现以下这种情况：

原因：DNS 被污染了

解决：在 hosts 文件`C:/windows/system32/drivers/etc/hosts`中添加 

```
192.30.253.112 github.com
```

保存后在终端`ipconfig / flushdns` 运行刷新配置

![](https://ws1.sinaimg.cn/large/acbcfa39ly1g381h1c4k6j20re0i6wgr.jpg)

参考资料：https://rovo98.github.io/posts/7e3029b3/

Ip地址查询：https://www.ipaddress.com/