---
layout: post
title: Linux 教程（1）
date: 2018-12-31 
tag: Linux


---

# Linux 简介

## 学习路线

第1阶段： linux环境下的基本操作命令， 包括 文件操作命令(rm mkdir chmod, chown) 编
辑工具使用（vi vim） linux用户管理(useradd userdel usermod)等
第2阶段： linux的各种配置（环境变量配置，网络配置，服务配置）
第3阶段： linux下如何搭建对应语言的开发环境（大数据， JavaEE, Python等）
第4阶段： 能编写shell脚本，对Linux服务器进行维护。
第5阶段： 能进行安全设置， 防止攻击，保障服务器正常运行，能对系统调优。
第6阶段： 深入理解Linux系统（对内核有研究），熟练掌握大型网站应用架构组成、并熟
悉各个环节的部署和维护方法 

## 基础篇

1. Linux 与 Unix 的关系

![](http://ww1.sinaimg.cn/large/acbcfa39ly1fyvw41gs1wj20r00epqa0.jpg)



![2](http://ww1.sinaimg.cn/large/acbcfa39ly1fyvw41iiskj20nl0fb10z.jpg)

2. 安装 VM 与 CentOS 

[安装包及安装教程](https://pan.baidu.com/s/1PKB2vTrtkmGgToNB7dzjZw) 提取码：`dlhv`

这部分有个问题要说一下。在虚拟机网络设置时，有几种模式可以选择，分别是

`桥接模式`：相当于把虚拟机当作一台本地电脑使用，它会占据本地网络的一个 IP

`NAT 模式`：我们设置时使用的是这个，可以避免 IP 冲突，可以访问外网

`仅主机模式`：不会造成 IP 冲突，但是不能访问外网

![](http://ww1.sinaimg.cn/large/acbcfa39ly1fyvx37ga1uj20jo089q3p.jpg)

3. 安装 vmtools 

vmtools 安装后，可以让我们在windows下更好的管理vm虚拟机

- 可以直接粘贴命令在windows 和 centos系统之间
  - 进入centos
  - 点击vm菜单的->install vmware tools
  - centos会出现一个vm的安装包
  - 点击右键解压, 得到一个安装文件
  - 进入该vm解压的目录 ，该文件在 /root/桌面/vmware-tools-distrib/下
  - 安装 ./vmware-install.pl
  - 全部使用默认设置即可
  - 需要reboot重新启动即可生效
- 可以设置windows和centos的共享文件夹 
  - 在物理机新建一个文件夹
  - 虚拟机-设置

![](http://ww1.sinaimg.cn/large/acbcfa39gy1fyvxcx5i8zj20qj0a4gmt.jpg)



参考资料：尚硅谷 Linux 课程