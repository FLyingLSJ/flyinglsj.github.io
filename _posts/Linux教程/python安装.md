- ubuntu 安装 python

  ```css
  执行更新
  # sudo apt update
  # sudo apt upgrade -y
  安装编译 Python 源程序所需的包
  # sudo apt install build-essential -y
  # sudo apt install libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev -y
  
  下载 Python 3.7源程序压缩包
  # wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz
  ```

  ```css
安装 pip
  pip -V # 版本查看
sudo apt-get install python3-pip
  
  安装 opencv
  sudo apt-get install python-opencv
  ```
  
  
  
  参考资料：
  
  > https://dzone.com/articles/install-python-370-on-ubuntu-1804debian-95
  >
  > https://blog.csdn.net/qq_35933777/article/details/84325856