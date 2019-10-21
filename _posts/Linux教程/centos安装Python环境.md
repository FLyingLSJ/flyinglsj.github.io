
开源软件包管理器工具 yum，可以轻松地安装和更新以及删除计算机上的软件包。
```bash
sudo yum -y update  # 更新 yum
#  -y 用于警告系统我们知道我们正在进行更改，从而防止终端提示我们进行确认。
```

**yum-utils**，它是扩展和补充yum的实用程序和插件的集合

```bash
sudo yum -y install yum-utils
```

安装 CentOS 开发工具，该工具用于允许您从源代码构建和编译软件

```bash
sudo yum -y groupinstall development
```

我们希望安装最新的 Python 3+ 稳定版本，所以我们需要安装 **IUS**，它表示 Inline with Upstream Stable。IUS 是一个社区项目，为某些较新版本的精选软件提供 Red Hat 软件包管理器（RPM）软件包。

```bash
sudo yum -y install https://centos7.iuscommunity.org/ius-release.rpm
```

安装 python3

```bash
sudo yum -y install python36u # 安装

python3.6 -V # 查看版本
sudo yum -y install python36u-pip # 安装 pip
sudo yum -y install python36u-devel # 安装IUS软件包python36u-devel，该软件包为我们提供了Python 3开发所需的库和头文件：
```

使用 python 虚拟环境进行开发（不同项目之间进行隔离）

```bash
mkdir environments  # 创建一个目录，用来放置一个项目
cd environments  # 进入到目录中
python3.6 -m venv my_env  # 创建一个 python 虚拟环境 my_env
source my_env/bin/activate # 激活环境
# 在虚拟环境中，你可以使用命令 python 来代替 python3.6，而是 pip 不是 pip3.6 。如果您在环境之外的计算机上使用 Python 3，则需要专门使用 python3.6 and pip3.6 命令。

deactivate # 退出环境

# 使用 pip 安装时，可以使用以下命令，他可以记住每次安装的包，
pip freeze > requirements.txt # 每次安装一个包以后都需要运行一下
# 但项目移植到其他地方，为了使其有相同的环境，使用下面的命令安装所需要的包
pip install -r requirements.txt # 可以创建相同版本的包 


# 要删除一个虚拟环境，只需删除它的文件夹。（要这么做请执行 rm -rf venv ）
```





参考：

- https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-centos-7