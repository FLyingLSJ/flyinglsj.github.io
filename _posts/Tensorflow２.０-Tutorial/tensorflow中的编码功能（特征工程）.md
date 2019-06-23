tensorflow 从 csv 读取文件，并将其转化为张量的方法：

- 数值类型

```python
from tensorflow import feature_column
from tensorflow.keras import layers


dataframe = pd.read_csv('*csv')

# 定义一个对象，数据将从其中产生
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy() 
    labels = dataframe.pop('target') # 将 target 列取出来
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)) # 将 dataframe 转化成字典
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5 
train_ds = df_to_dataset(train, batch_size=batch_size)

example_batch = next(iter(train_ds))[0] 

################# 数值类型直接转为张量
# 得到的是一个字典：
# {'age': <tf.Tensor: id=260, shape=(5,), dtype=int32, numpy=array([44, 57, 61, 67, 40])>, 'sex': <tf.Tensor: id=268, shape=(5,), dtype=int32, numpy=array([1, 1, 1, 1, 1])>, ...}
age = feature_column.numeric_column("age") # 将 age 列进行转换数据列
feature_layer = layers.DenseFeatures(age)
feature_layer(example_batch).numpy() 
# 结果：
'''
[[44.]
 [57.]
 [61.]
 [67.]
 [40.]]
'''
```

```python
######################## 数值类型进行分桶，相当于对数据进行分块后用二进制位来进行编码
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_layer = layers.DenseFeatures(age_buckets)
feature_layer(example_batch).numpy() 
'''运行结果
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
'''
```

- 非数值类型：读取的 csv 文件中，有些数据列可能不是数值类型，所以要对其进行编码

```python
# ------------------------- 方法一：one-hot 编码
# 例如：thal 这一列特征中包含有三个属性，所以对其进行编码
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = feature_column.indicator_column(thal)
feature_layer = layers.DenseFeatures(thal_one_hot)
feature_layer(example_batch).numpy() 
'''
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
'''

# --------------------- 方法二：Embedding 编码
# 对于某个列包含有多个特征，不适合使用 one-hot 进行编码，故使用 Embedding，其值不再是二进制位，而是可以是任何值（应该是 -1～1 的值），可以编码成任意维度的特征列，这个维度是一个参数，可以调节。
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_layer = layers.DenseFeatures(thal_embedding)
feature_layer(example_batch).numpy() 
''' 结果
[[-0.11711289 -0.4844503  -0.2236189  -0.16654855  0.11912168  0.34882942
   0.45681918 -0.20051774]
 [-0.13363995 -0.36279342  0.23082688  0.0015385   0.61947876 -0.44539914
   0.5930291  -0.17928404]
 [-0.11711289 -0.4844503  -0.2236189  -0.16654855  0.11912168  0.34882942
   0.45681918 -0.20051774]
 [-0.34317735  0.24289864  0.5409457  -0.24941932  0.02070024  0.13791308
   0.04231372 -0.09060758]
 [-0.11711289 -0.4844503  -0.2236189  -0.16654855  0.11912168  0.34882942
   0.45681918 -0.20051774]]
'''

# ------------------------- 方法三 hashed 编码，分桶
# 原理不是很懂
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print((feature_layer(example_batch).numpy()))
    
thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=15)
demo(feature_column.indicator_column(thal_hashed))    
'''结果
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
'''



```



参考资料：<https://blog.csdn.net/cjopengler/article/details/78161748> 这个资料的版本不是　２.０　的
