virtualenv 工具简介

[virtualenv](http://pypi.python.org/pypi/virtualenv) 是一个创建隔绝的Python环境的 工具。virtualenv创建一个包含所有必要的可执行文件的文件夹，用来使用Python工程所需的包。可以将不同项目的 python 环境进行隔绝

```bash
pip install virtualenv # 安装
virtualenv --version # 查看安装情况
mkdir my_project_folder # 创建一个新工程
cd my_project_folder # 
virtualenv venv # 创建一个虚拟环境

# 要开始使用虚拟环境，其需要被激活：
source venv/bin/activate 
```

```bash
conda create -n test python==3.6 # test 是环境名
conda activate test 

```



当前虚拟环境的名字会显示在提示符左侧（比如说 `(venv)您的电脑:您的工程 用户名$） 以让您知道它是激活的。从现在起，任何您使用 pip 安装的包将会放在 `venv文件夹中， 与全局安装的 Python 隔绝开。

```bash
# 如果您在虚拟环境中暂时完成了工作，则可以停用它，这将会回到系统默认的Python解释器，包括已安装的库也会回到默认的。
deactivate 

# 要删除一个虚拟环境，只需删除它的文件夹。（要这么做请执行 rm -rf venv ）


```

注意事项：

为了保持您的环境的一致性，“冷冻住（freeze）”环境包当前的状态是个好主意。要这么做，请运行：

```bash
pip freeze > requirements.txt # 每次安装一个包以后都需要运行一下
```

这将会创建一个 `requirements.txt` 文件，其中包含了当前环境中所有包及 各自的版本的简单列表。您可以使用 `pip list` 在不产生requirements文件的情况下， 查看已安装包的列表。这将会使另一个不同的开发者（或者是您，如果您需要重新创建这样的环境） 在以后安装相同版本的相同包变得容易。

```bash
pip install -r requirements.txt # 可以创建相同版本的包
```

这能帮助确保安装、部署和开发者之间的一致性。





参考：

https://pythonguidecn.readthedocs.io/zh/latest/dev/virtualenvs.html

https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-uswgi-and-nginx-on-ubuntu-18-04