---
layout: post
title: DBC
date: 2019-11-09
tag: 服务器 DBC
---



1. 在网站上创建一个钱包账号，创建以后会生成一个钱包地址，将钱包地址发送到群里面，会有工作人员会为你的账号进行充值。**（有个密钥务必保存下来，丢了，谁也没办法）**

2. 充值成功后，需要绑定邮箱

3. 开始租用设备

4. 收到一条邮件通知，上面有服务器的账号和密码和 notebook 的地址

   

   根据邮件的信息登录到远程服务器，我是使用的是 Xshell

   ![1571485264887](https://tva4.sinaimg.cn/large/acbcfa39gy1g928s8efn3j20vg08xaas.jpg)

![1571485227517](https://tvax1.sinaimg.cn/large/acbcfa39gy1g928s97kw3j20ic092aae.jpg)

5. 建议一开始时，新建 python 虚拟环境（曾经遇到坑，python 版本不对代码运行错误）

```bash
# 安装必要的包
apt install lrzsz # 压缩包命令，下载文件乐可以使用
unzip -o test.zip -d tmp/  # 

ls -l | grep "^-" | wc -l # 查看当前路径下有多少文件

conda create -n your_env_name python=X.X（2.7、3.6等) 
# 命令创建python版本为X.X、名字为your_env_name的虚拟环境。your_env_name文件可以在Anaconda安装目录envs文件下找到。
 source activate your_env_name(虚拟环境名称) #　激活虚拟环境
 conda deactivate # 退出环境
 
 conda remove -n your_env_name(虚拟环境名称) --all # 删除虚拟环境
```

