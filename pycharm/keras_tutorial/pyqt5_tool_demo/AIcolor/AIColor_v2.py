# !/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import websocket
import time
from keras.models import load_model
from keras.preprocessing import image
import os
from numpy import argmax
from pandas import read_csv


ImageData_path = 'ImageData'
Model_path = 'ModelPath'
flag = True
header = {
    "CommandHandler": "AIColor",
    "CommandName": "RegistAIComm"
}  # 注册信息需要为 Json 格式
header = json.dumps(header)

error_message = {
    'CommandHandler': "AIColor",
    'CommandName': "AIErrFeedback",
    'ParamsJson': {
        'ErrMsg': "XXXXXXXXXXXX"
    }
}


img_type = 'jpg' # 图片类型标志位
path = "./" #
model_path = "F:/jupyter/pycharm/keras_tutorial/pyqt5_tool_demo/ALcolor/ModelPath/1555812970.2107625.h5"
model = '' # 模型标志位
temp = 1 # 判断成功标志位
configure_content = "" # 配置文件读取变量
try:
    import thread
except ImportError:
    import _thread as thread


def on_message(ws, message):
    global img_type, model_path, path, temp, error_message, model
    while temp:
        regist_status = json.loads(message)
        if bool(regist_status['success']):
            print("与服务器连接成功")
            temp = 0

    message = json.loads(message)
    img_name = message['msg']['Msg']['ImgFile']  # 得到的是图片的文件名，不包括路径和后缀
    Wood = message['msg']['Msg']['Wood'] # 木种代码
    img_path = path + img_name + '.' + img_type  # 图片集路径 + 图片名 + . + 后缀
    if os.path.exists(img_path):  # 判断文件是否存在
        img_array = img_processing(img_path)  # 将图片转换成张量
        prediction = img_predict(model, img_array)  # 对图片进行预测，返回为字符串
        print("预测结果为：%s" % prediction)

        # 返回给服务端
        ColorResult = {
            'CommandHandler': "AIColor",
            'CommandName': "ColorChecked",
            'ParamsJson': {
                'FileName': "X112212334456",
                'Wood': "RO",
                'ColorCode': "RO_RD"
            }
        }
        ColorResult['ParamsJson']['FileName'] = img_name # 文件名写入
        ColorResult['ParamsJson']['Wood'] = Wood # 木种代码写入
        ColorResult['ParamsJson']['ColorCode'] = prediction # 预测结果写入
        ColorResult = json.dumps(ColorResult)
        ws.send(ColorResult)
    else:
        error_message['ParamsJson']['ErrMsg'] = "file no exist img_type? img_name?"
        print("图片文件不存在，请检查图片名或图片类型")  # 通信协议上要约定若文件不存在时的情况
        message = json.dumps(error_message) # error_message：默认错误信息，数据类型为字典，message：数据类型为 json
        ws.send(message)


def on_error(ws, error):
    global temp
    print(error)
    print("与服务器连接上失败，请检查服务器是否开启")
    #print("系统 5s 后退出...")
    # time.sleep(5)
    print("--------------------------")
    print("1: 重新打开服务端并连接")
    print("2: 关闭")
    while 1:
        try:
            restart = int(input("请选择"))
            break
        except BaseException:
            print("请输入数字")
    if restart == 1:
        temp = 1
        main()


def on_close(ws):
    print("### closed ###")


def on_open(ws):
    def run(*args):
        ws.send(header)
        print("thread terminating...")

    thread.start_new_thread(run, ())


def get_imgTtpe(configure_content):
    """

    :param configure_content: 配置文件的内容，存储在 csv 文件中
    :return: 返回图片的类型，如 jpg,jpeg 返回类型：字符
    """
    # print("数据集文件名为 ImageData：")
    print("请选择本次识别图像的类型")
    img_type_dict = dict()
    for i, content in enumerate(list(configure_content['img_type'])):
        if type(content) != float:
            print(i, ":", content)
            img_type_dict[i] = content

    while 1:
        try:
            img_type_flag = int(eval(input("请选择")))
            break
        except BaseException:
            print("请输入数字")

    while img_type_flag not in list(img_type_dict.keys()):
        try:
            print("请重新选择")
            img_type_flag = int(eval(input("请选择")))

        except BaseException:
            print("重试")

    img_type = img_type_dict[img_type_flag]
    print("所选图片格式为：%s" % img_type)
    print("--------------------------")
    return img_type


def get_imgPath():
    """
    获取数据集的路径
    :return: 数据集的路径，不包括图片名
    """
    # 读取图片集路径
    now_path = os.getcwd()
    a = 1
    n = len(now_path.split("\\"))
    while not (ImageData_path in os.listdir(now_path)):
        # print("1:", now_path)
        a += 1
        n = len(now_path.split("\\")[-1]) + 1
        now_path = now_path[:-n]
        if a == n - 2:
            print("请检查图片路径 %s 是否存在" % ImageData_path)
            print("系统 5s 后退出...")
            time.sleep(5)
            try:
                os._exit(0)
            except BaseException:
                print('Program is dead.')
    path = now_path + "\\" + ImageData_path + str("\\")  # 图片的路径，当前路径找不到，会一直往上找
    return path


def get_modelName(configure_content):
    """
    获取模型的名字
    :param configure_content: configure_content: 配置文件的内容，存储在 csv 文件中
    :return:
    """
    print("请选择本次预测的模型：")
    model_dict = dict()
    for i, content in enumerate(list(configure_content['model'])):
        if type(content) != float:
            print(i, ":", content)
            model_dict[i] = content

    while 1:
        try:
            model_name_flag = int(eval(input("请选择模型")))
            break
        except BaseException:
            print("请输入数字")

    while model_name_flag not in list(model_dict.keys()):
        try:
            print("请重新选择模型")
            model_name_flag = int(eval(input("请选择模型")))

        except BaseException:
            print("重试")
    model_name = model_dict[model_name_flag]
    print("所选模型为：%s" % model_name)
    print("--------------------------")
    print("等待服务器发送数据...")
    return model_name


def get_modelPath():
    """
    # 获取模型路径 当前路径找不到，会一直往上找
    :return: 模型的路径，不包含模型文件名
    """
    global Model_path
    model_path = os.getcwd()
    a = 1
    n = len(model_path.split("\\"))
    while not (Model_path in os.listdir(model_path)):
        # print(now_path)
        a += 1
        n = len(model_path.split("\\")[-1]) + 1
        model_path = model_path[:-n]
        if a == n - 2:
            print("请检查模型路径 %s 是否存在" % Model_path)
            print("系统 5s 后退出...")
            time.sleep(5)
            try:
                os._exit(0)
            except BaseException:
                print('Program is dead.')
    model_path = model_path + "\\" + Model_path + "\\"
    return model_path


def get_serverIp(configure_content):
    print("请选择服务器地址: ip:port")  # 从配置文件中读取，若有新输入，保存
    server_dict = dict()
    for i, content in enumerate(list(configure_content['ip'])):
        if type(content) != float:
            print(i, ":", content)
            server_dict[i] = content

    while 1:
        try:
            server_flag = int(eval(input("请选择")))
            break
        except BaseException:
            print("请输入数字")

    while server_flag not in list(server_dict.keys()):
        try:
            print("请重新选择")
            server_flag = int(eval(input("请选择")))
        except BaseException:
            print("重试")

    ip = server_dict[server_flag]
    print("所选的服务器为：%s" % ip)
    print("--------------------------")
    return ip

def get_configure_content():
    configure_file = 'configure.csv'
    configure_path = os.getcwd() + "\\" + configure_file
    if os.path.exists(configure_path):
        print("读取配置文件成功")
        configure_content = read_csv(configure_path)  # 配置文件内容

    else:
        print("配置文件不存在，请重试...")
        print("系统 5s 后退出...")
        time.sleep(5)
        try:
            os._exit(0)
        except BaseException:
            print('Program is dead.')

    return configure_content

def guide_txt():
    """
    信息提示语
    :return:
    """
    global add, img_type, path, model, configure_content

    # 服务器开启提示语
    print("AI COLOR 服务")
    print("请确认服务端是否开启 yes/no:")
    print("1: yes", "2: no", sep='\n')
    start_flag = input()
    while 1:
        try:
            if int(start_flag) == 1:
                break
            else:
                print("请开启后重试")
                start_flag = input()
        except:
            print("请输入数字！")
            start_flag = input()

    print("AI COLOR 服务启动...")
    print("--------------------------")

    # 读取配置文件
    configure_content = get_configure_content()
    # 服务器地址
    ip = get_serverIp(configure_content)
    add = 'ws://' + ip
    # 选择图片类型
    img_type = get_imgTtpe(configure_content)
    # 读取图片集路径
    path = get_imgPath()
    # 获取模型路径
    model_path = get_modelPath()
    # 从配置文件中读取模型文件名
    # model_name = '1555812970.2107625.h5'
    model_name = get_modelName(configure_content)
    # 获取模型，获取不到就会直接退出
    try:
        # 模型的路径（F:\jupyter\pycharm\keras_tutorial\pyqt5_tool_demo\ALcolor\）+模型的名字
        model_path = model_path + model_name
        model = load_model(model_path)  # 模型读取，放在这里性能更好
    except BaseException:
        print("无法找到模型，请重试...")
        print("系统 5s 后退出...")
        time.sleep(5)
        try:
            os._exit(0)
        except BaseException:
            print('Program is dead.')


def img_processing(img_path):
    global model
    target_size = model.input_shape # (None, 150, 150, 3)
    img = image.load_img(img_path, target_size=(target_size[1], target_size[2]))
    img_array = image.img_to_array(img)
    # print(img_array.shape)
    img_array = img_array / 255.
    img_array = img_array.reshape(-1, target_size[1], target_size[2], 3)
    return img_array


def img_predict(model, img_array):
    """
    预测
    :param model: 预测的模型
    :param img_array: 输入图片的张量
    :return: 预测结果：字符
    """
    global configure_content
    result = model.predict(img_array)
    result = argmax(result)

    # 以后考虑从配置文件中读取
    #configure_content = get_configure_content()
    color_code_dict = dict()
    for i, content in enumerate(list(configure_content['color_code'])):
        if type(content) != float:
            #print(i, ":", content)
            color_code_dict[i] = content
    #print(color_code_dict)
    # 在配置文件中确定后就不能再修改
    if 0 == result:
        prediction = color_code_dict[2]#"RO_LB"  # \brown  - 1
    if 1 == result:
        prediction = color_code_dict[5]#'RO_MN'  # \Mineral - 5
    if 2 == result:
        prediction = color_code_dict[4]#'RO_EG'  # \SAPwood - 4
    return prediction


def main():
    guide_txt()
    websocket.enableTrace(False)  # 不显示报文
    ws = websocket.WebSocketApp(url=add,  # 'ws://127.0.0.1:2019'
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()


if __name__ == "__main__":
    main()
