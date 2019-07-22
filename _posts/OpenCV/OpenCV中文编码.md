- OpenCV 读取中文文件名

```python
test_sample_file="云A526EG.jpg"
image_origin = cv2.imdecode(np.fromfile(test_sample_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
print(image_origin.shape[0])

```

