# !/usr/bin/python3

# coding = utf-8

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QLineEdit, QTextBrowser, QComboBox
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QIcon
from random import randint
from websocket import create_connection, WebSocket
import json
import os
import time
from keras.models import load_model
import matplotlib.pyplot as plt
import multiprocessing as mp  # 多线程模块
import pandas as pd

class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.num = randint(1, 100)
        self.ws = 1  # 初始化 wbsocket 的参数
        self.ImageData_path = 'ImageData'  # 图片的存储文件名
        self.close_flag = True

    def get_imgPath(self):
        """
         path = self.get_imgPath() # 获取图片的路径，在这个子程序中完成图像处理工作
         img_type = 'jpg' # 图片的格式
        :return:
        """
        path_letter = os.getcwd().split("\\")[0:-1]
        path = ""
        for i in path_letter:
            path = path + str(i) + r"\\"
        path = path + str("\\") + self.ImageData_path
        return path

    def get_model(self):
        """
        获取预测模型
        :return:
        """
        model_path = "./modelPath/wood_model.h5"
        model = load_model(model_path)
        return model

    def initUI(self):

        self.setGeometry(200, 200, 800, 500)
        self.setWindowTitle('websocket')
        self.setWindowIcon(QIcon('xdbcb8.ico'))

        # 连接按键界面设置
        self.connect_button = QPushButton('open', self)
        self.connect_button.setGeometry(30, 150, 70, 30)
        self.connect_button.setToolTip('<b>连接</b>')
        self.connect_button.clicked.connect(self.connect_showMessage)

        # 断开连接按键界面设置
        self.close_button = QPushButton('close', self)
        self.close_button.setGeometry(115, 150, 70, 30)
        self.close_button.setToolTip('<b>断开连接</b>')
        self.close_button.clicked.connect(self.close_showMessage)

        # 开始接受服务器的按钮
        self.starting_button = QPushButton('starting', self)
        self.starting_button.setGeometry(30, 220, 100, 30)
        self.starting_button.setToolTip('<b>开始工作</b>')
        self.starting_button.clicked.connect(self.starting_showMessage)

        # 信息提示界面
        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setGeometry(QRect(300, 50, 399, 350))
        self.textBrowser.setObjectName("textBrowser")

        # 文本输入框界面设置
        self.text = QLineEdit('ws://127.0.0.1:2019', self)
        self.text.selectAll()
        self.text.setFocus()
        self.text.setGeometry(60, 50, 220, 30)
        self.button_status()  # 最开始设置只要连接按钮  可以操作

        # 下拉复选框
        content = pd.read_csv("./configure.csv")#["jpg", "bmp"] # configure.csv 用来保存配置文件
        img_format = [i for i in content['img_format']] # 从配置文件读取图片的格式参数
        self.combo = QComboBox(self)
        self.combo.addItems(img_format)
        self.combo.setGeometry(60, 100, 220, 20)

        self.show()

    def button_status(
            self,
            connect_button_status=True,
            close_button_status=False,
            starting_button_status=False):
        """
        :param connect_button_status:
        :param close_button_status:
        :param register_button_status:
        :return: # 初始按钮的状态：可以操作，不可操作（灰色）
        """

        self.connect_button.setEnabled(connect_button_status)
        self.close_button.setEnabled(close_button_status)
        self.starting_button.setEnabled(starting_button_status)

    # 连接按钮事件处理
    def connect_showMessage(self):

        connect_flage = True
        while connect_flage:
            add = str(self.text.text())
            try:
                self.ws = WebSocket()
                self.ws.connect(add)
                if self.ws.status == 101:
                    #QMessageBox.about(self, '连接状态', '连接成功')
                    # self.text.setFocus()
                    self.button_status(False, True, True)
                    self.text.setReadOnly(True)  # 设置地址输入框连接以后就不可编辑
                    RegistComm = {
                        "CommandHandler": "AIColor",
                        "CommandName": "RegistComm",
                        "ParamsJson": ""}  # 注册信息需要为 Json 格式
                    RegistComm = json.dumps(RegistComm)
                    self.ws.send(RegistComm)  # 发送注册信息
                    RegResult = self.ws.recv()  # 返回的是字符串
                    RegResult = json.loads(RegResult)  # 转换成字典格式
                    if bool(RegResult['success']):
                        self.textBrowser.append(str(time.ctime()) + ": 连接成功")
                        QMessageBox.about(self, '连接状态', '连接成功')
                        self.text.setFocus()
                        self.button_status(False, True, True)
                        self.combo.setEnabled(False) # 设置连接成功后，图片格式不可选

                    connect_flage = False

            except Exception as e:
                self.textBrowser.append(str(time.ctime()) + ": 连接失败")
                QMessageBox.about(self, '连接状态', '连接失败')
                self.text.setFocus()
                self.register_flag = False
                #print("register_flag", self.register_flag)
                break

    def close_showMessage(self):
        self.ws.close()
        self.textBrowser.append(str(time.ctime()) + ": 断开连接")
        QMessageBox.about(self, '连接状态', '连接断开')
        self.text.setFocus()
        self.button_status()  # 断开以后就是初始状态
        self.text.setReadOnly(False)  # 断开以后地址框可以编辑
        self.combo.setEnabled(True)  # 断开连接后，图片格式可选

    def starting_showMessage(self):
        self.close_flag = True
        self.textBrowser.append(str(time.ctime()) + ": 开始工作")
        QMessageBox.about(self, '工作状态', '开始工作')
        self.text.setFocus()
        #model = self.get_model()
        # print(model.summary())
        img_path = self.get_imgPath()
        while self.close_flag:
            self.receive_data(img_path)



    def receive_data(self, img_path):
        Recognize_info = self.ws.recv()  # 接收到的是字符串
        Recognize_info = json.loads(Recognize_info)  # 转化成字典
        # Recognize_info['msg']['Msg']['ImgFile'] 这个获取的是图片的文件名，不包括路径和后缀

        try:
            if Recognize_info['msg']['Msg']['Wood'] == 'END':
                #self.ws.close()
                self.close_flag = False
                self.textBrowser.append(str(time.ctime()) + ": 断开连接")

            image_path = img_path + str("\\") + str(Recognize_info['msg']['Msg']['ImgFile']) + str(".") + str(self.combo.currentText())
            img = plt.imread(image_path)
            print(image_path)
            #plt.imshow(img)
            #plt.show()
        except Exception as e:
            print("找不到文件")
            self.textBrowser.append(str(time.ctime()) + ": 找不到文件")



    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            '确认',
            '确认退出吗',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
