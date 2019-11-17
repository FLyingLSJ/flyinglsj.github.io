import json
import websocket
import time
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np


ImageData_path = 'ImageData'
flag = True
header = {
    "CommandHandler": "AIColor",
    "CommandName": "RegistComm",
    "ParamsJson": ""}  # 注册信息需要为 Json 格式
#header = dict(header)
header = json.dumps(header)
img_type = 'jpg'
path = "./"
model_path = "F:/jupyter/pycharm/keras_tutorial/pyqt5_tool_demo/ALcolor/modelPath/1555812970.2107625.h5"


try:
    import thread
except ImportError:
    import _thread as thread


def on_message(ws, message):
    global img_type, model_path, path
    print(message)
    t1 = time.time()
    print(time.ctime())
    message = json.loads(message)
    img_name = message['msg']['Msg']['ImgFile'] # 得到的是图片的文件名，不包括路径和后缀
    img_path = path + img_name + '.' +img_type # 图片集路径 + 图片名 + . + 后缀
    #print(img_path)
    img_array = img_processing(img_path) # 将图片转换成张量
    #print(img_array)
    result = img_predict(model_path, img_array) # 对图片进行预测
    print(np.argmax(result))
    print(time.time()-t1)
    print(time.ctime())
    if message['msg']['Msg']['ImgFile'] == 'end':
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
    global add, img_type, path
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
    print("请选择本次识别图像的类型")
    print("1: bmp", "2: jpg", "3: 其他", sep='\n') # 图片类型从配置文件中读取，若有新输入，保存
    img_type_flag = input()
    while 1:
        if img_type_flag == '1':
            img_type = 'bmp'
            break
        if img_type_flag == '2':
            img_type = 'jpg'
            break
        if img_type_flag == '3':
            print("请输入图片类型：")
            img_type = input()
            #print(img_type)
            break
        else:
            print("命令无效，请重新输入")
            img_type_flag = input()

    # 读取图片集路径
    path_letter = os.getcwd().split("\\")[0:-1]
    path = ""
    for i in path_letter:
        path = path + str(i) + str("\\")
    path = path + ImageData_path + str("\\") # 型如：'F:\\jupyter\\pycharm\\keras_tutorial\\pyqt5_tool_demo\\ImageData\\'


def img_processing(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.
    img_array = img_array.reshape(-1, 150, 150, 3)
    return img_array

def img_predict(model_path, img_array):
    model = load_model(model_path)
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



