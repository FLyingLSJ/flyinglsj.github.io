import json
import websocket
import time
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd
import sys


ImageData_path = 'ImageData'
Model_path = 'ModelPath'
flag = True
header = {
    "CommandHandler": "AIColor",
    "CommandName": "RegistComm",
    "ParamsJson": ""}  # 注册信息需要为 Json 格式
#header = dict(header)
header = json.dumps(header)
img_type = 'jpg'
path = "./"
model_path = "F:/jupyter/pycharm/keras_tutorial/pyqt5_tool_demo/ALcolor/ModelPath/1555812970.2107625.h5"
model = ''

try:
    import thread
except ImportError:
    import _thread as thread


def on_message(ws, message):
    global img_type, model_path, path
    print(time.ctime())
    message = json.loads(message)
    img_name = message['msg']['Msg']['ImgFile'] # 得到的是图片的文件名，不包括路径和后缀
    img_path = path + img_name + '.' +img_type # 图片集路径 + 图片名 + . + 后缀
    if os.path.exists(img_path) == False: # 判断文件是否存在
        print("图片文件不存在，请检查图片名或图片类型") # 通信协议上要约定若文件不存在时的情况
    img_array = img_processing(img_path) # 将图片转换成张量
    result = img_predict(model, img_array) # 对图片进行预测
    # 发送回去服务端 待写。。。。。
    #############
    pass

    print(np.argmax(result))
    print(time.ctime())
    if message['msg']['Msg']['Wood'] == 'end': # 通信协议加上结束运行的参数
        ws.close()



def on_error(ws, error):
    print(error)


def on_close(ws):
    print("### closed ###")


def on_open(ws):
    def run(*args):
        ws.send(header)
        print("thread terminating...")

    thread.start_new_thread(run, ())



def guide_txt():
    """
    信息提示语
    :return:
    """
    global add, img_type, path, model

    # 服务器开启提示语
    print("AI COLOR 服务")
    print("请确认服务端是否开启 yes/no:")
    print("1: yes", "2: no", sep='\n')
    start_flag = input()
    while 1:
        if start_flag == '1':
            break
        else:
            print("请开启后重试")
            start_flag = input()
    print("请输入服务器地址: ip:port") # 从配置文件中读取，若有新输入，保存
    print("历史记录：", "1: xx", "2: xx", sep='\n')
    add = '127.0.0.1:2019'#input()
    add = 'ws://' + add

    # 选择图片类型
    print("数据集文件名为 ImageData：")
    print("请选择本次识别图像的类型")
    configure_content = pd.read_csv("./configure.csv") # 配置文件内容
    img_type_dict = dict()
    for i, content in enumerate(list(configure_content['img_type'])):
        print(i, ":", content)
        img_type_dict[i] = content
    while 1:
        try:
            img_type_flag = int(eval(input("请选择")))
            break
        except:
            print("请输入数字")
    while 1:
        try:
            if img_type_flag in list(img_type_dict.keys()):
                # print(img_type_flag)
                img_type = img_type_dict[img_type_flag]
                print("所选图片类型为：%s" % img_type)
                break
            else:
                print("请重新选择")
                img_type_flag = int(eval(input("请选择")))
        except:
            print("重试")


    # 读取图片集路径
    now_path = os.getcwd()
    a = 1
    n = len(now_path.split("\\"))
    while not (ImageData_path in os.listdir(now_path)):
        #print("1:", now_path)
        a += 1
        n = len(now_path.split("\\")[-1]) + 1
        now_path = now_path[:-n]
        if a == n-2:
            print("请检查图片路径 %s 是否存在" % ImageData_path)
            print("系统 5s 后退出...")
            time.sleep(5)
            try:
                os._exit(0)
            except:
                print('Program is dead.')
    path = now_path + "\\" + ImageData_path + str("\\") # 图片的路径，当前路径找不到，会一直往上找
    #print("图片路径：", path)


    # 获取模型路径 当前路径找不到，会一直往上找
    model_path = os.getcwd()
    a = 1
    n = len(model_path.split("\\"))
    while not (Model_path in os.listdir(model_path)):
        #print(now_path)
        a += 1
        n = len(model_path.split("\\")[-1]) + 1
        model_path = model_path[:-n]
        if a == n - 2:
            print("请检查模型路径 %s 是否存在" % Model_path)
            print("系统 5s 后退出...")
            time.sleep(5)
            try:
                os._exit(0)
            except:
                print('Program is dead.')
    model_path = model_path + "\\" + Model_path + "\\"

    # 从配置文件中读取模型文件名
    model_dict = dict()
    for i, content in enumerate(list(configure_content['model'])):
        print(i, ":", content)
        model_dict[i] = content
    while 1:
        try:
            model_name_flag = int(eval(input("请选择")))
            break
        except:
            print("请输入数字")
    while 1:
        try:
            if model_name_flag in list(model_dict.keys()):
                # print(img_type_flag)
                model_name = model_dict[model_name_flag]
                print("所选的模型为：%s" % model_name)
                break
            else:
                print("请重新选择")
                img_type_flag = int(eval(input("请选择")))
        except:
            print("重试")

    #model_name = '1555812970.2107625.h5'

    # 获取模型，获取不到就会直接退出
    try:
        model_path = model_path + model_name  # 模型的路径（F:\jupyter\pycharm\keras_tutorial\pyqt5_tool_demo\ALcolor\）+模型的名字
        model = load_model(model_path) # 模型读取，放在这里性能更好
    except:
        print("无法找到模型，请重试...")
        print("系统 5s 后退出...")
        time.sleep(5)
        try:
            os._exit(0)
        except:
            print('Program is dead.')


def img_processing(img_path, target_size=150):
    img = image.load_img(img_path, target_size=(target_size, target_size))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.
    img_array = img_array.reshape(-1, target_size, target_size, 3)
    return img_array

def img_predict(model, img_array):
    result = model.predict(img_array)
    return result


if __name__ == "__main__":
    guide_txt()
    websocket.enableTrace(False) # 不显示报文
    ws = websocket.WebSocketApp(url=add, # 'ws://127.0.0.1:2019'
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()



