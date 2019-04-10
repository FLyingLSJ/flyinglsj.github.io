```
#!pip install kaggle
from google.colab import drive
import json

def get_googledrive():
  # 登录 Drive
  # !ls "/content/drive/My Drive/" # 显示 GoogleDrive 的文件内容
  return drive.mount('/content/drive/')  

# 登入 Drive 并写入 Token
# ! /drive/My Drive
get_googledrive()


base_dir = "./drive/My Drive/Kears-Python-tutorial/dog_vs_cat/small_datasets/"
```

```
!ls
!tar zxvf filename.tar.gz
```

