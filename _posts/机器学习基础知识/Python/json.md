#### 如何保存训练后得到的 acc loss 等数据

```
# 保存训练数据
with open('history.json', 'w') as f:
    json.dump(history.history, f)
```

```
# 读取训练数据并绘制
with open("history.json", 'r') as f:
    history_load = f.readlines()

history_load = json.loads(history_load[0])
acc = history_load['acc']
val_acc = history_load['val_acc']
loss = history_load['loss']
val_loss = history_load['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure() # 在另一个图像绘制
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
```

