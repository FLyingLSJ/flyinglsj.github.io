---
layout: post
title: 用 Jekyll 搭建博客
date: 2018-12-31 
tag: 博客


---

# 安装 Ruby

[文件下载地址](https://rubyinstaller.org/downloads/) [参考菜鸟教程](http://www.runoob.com/ruby/ruby-installation-windows.html)
主要注意的问题是 Ruby 版本与下面要安装的 Devkit 的版本要适应。

尽量选择 WITHOUT DEVKIT ，因为我在安装的时候，安装了 WITH DEVKIT 版本的好像在博客搭建过程中使用不了。

Ruby 下载界面

DEVKIT 下载界面

# 安装 DEVKIT 
解压到 C:/Devkit （随意，你知道的地方就行）
进行初始化
在Deckit解压目录运行cmd，执行ruby dk.rb init，初始化成功后，Devkit目录下将出现config.yml文件，打开，最后有一句- D:\Ruby22-x64（这里是Ruby的安装目录），如没有，则手动添加。回到cmd窗口，继续执行ruby dk.rb install，顺利结束后，则Devkit配置完成。


1.	安装 Jekyll
   因为我们是在 Github 上搭建博客，要显示效果，你需要 Push 到 Github 上才能显示效果。Jekyll 主要的功能就是可以在本地实时显示你的博客，仅此而已。
   安装：
   用 Ruby 安装 Jekyll 使用命令
   gem install jekyll 安装 jekyll 用 jekyll -v 查看是否安装完成

2.	安装 bundler
   $ gem install bundler
   $ bundle install

3.	安装其他
   安装 Pygments
   安装 Python
   安装 ‘Easy Install’
   安装 Pygments
   因为其他环境在我的电脑之前已经搭建了，这边可以参考：https://blog.csdn.net/rainloving/article/details/45745491

4.	启动 Jekyll 
   以上都安装完毕后，就可以启动 Jekyll 
   Jekyll 的官方文档中介绍了一个建议 blog （https://jekyllrb.com/docs/）搭建的程序代码：
   jekyll new myblog # 新建一个博客
   cd myblog # 定位到此目录
   jekyll serve  # 启动 Jekyll 
    这里讲一下这里可能出现的问题：
   首先是运行 jekyll serve 时会出现以下错误

这个错误是因为 Jekyll 默认以 4000 端口打开博客，但是有后台程序被占用了。
解决办法：输入命令 netstat -ano 查看计算机端口被占用情况

或者直接查看 4000 端口 使用netstat - aon | findstr "400"


发现是 chrome 占用了，我们可以关闭这个服务，当然也可以在启动jekyll服务的时候指定端口号，如下：
jekyll serve --port 3000 #  建议使用这个

然后在浏览器中输入 http://127.0.0.1:300 就可以运行博客了。
Jekyll serve –-port 3000 –-watch # 注意是两个横杠， --watch 的意思是你修改你的博客后刷新一下网页也就跟着更新。

 

7.	Github Page 配置
   注册 Github 账号
   建仓库，用来存博客内容
   参考这里：https://zhuanlan.zhihu.com/p/28321740
   注意的地方： 
8.	选择模板：

你可以 fork 我使用的模板。或者到原作者的 Github fork 也行https://github.com/leopardpan/leopardpan.github.io/
9.	更新与使用
   下载一个 github 界面版，在 https://zhuanlan.zhihu.com/p/28321740 也介绍了。

10.	博客文件目录简单介绍 
11.	其他问题
    当你把 blog push 到 Github 上时，有可能会出现以下问题

这是因为 Jekyll 和 ffi 与 Github page 版本不匹配的原因，这是你要对你的本地环境进行更新。输入代码进行更新即可



参考资料：
Windows 上安装 Jekyll https://blog.csdn.net/rainloving/article/details/45745491
​	及以上所借鉴的文章