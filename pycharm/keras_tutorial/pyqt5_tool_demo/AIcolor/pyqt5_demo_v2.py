

# !/usr/bin/python3

# coding = utf-8

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QLineEdit
from PyQt5.QtGui import QIcon
from random import randint
from websocket import create_connection
import json


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.num = randint(1, 100)
        self.ws = 1  # 初始化 wbsocket 的参数
        self.register_flag = True  # 注册标志位，若连接成功，设置为 True，否则为 False

    def initUI(self):
        self.setGeometry(300, 300, 600, 500)
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


        # 注册按键界面设置
        self.register_button = QPushButton('register', self)
        self.register_button.setGeometry(30, 180, 100, 30)
        self.register_button.setToolTip('<b>注册</b>')
        self.register_button.clicked.connect(self.register_showMessage)

        # 信息提示界面

        # 文本输入框界面设置
        self.text = QLineEdit('ws://127.0.0.1:2019', self)
        self.text.selectAll()
        self.text.setFocus()
        self.text.setGeometry(80, 50, 220, 30)
        self.show()

    # 连接按钮事件处理
    def connect_showMessage(self):
        connect_flage = True
        # add = str(self.text.text())  # ws://127.0.0.1:2019
        while connect_flage:
            add = str(self.text.text())
            try:
                self.ws = create_connection(add)
                if self.ws.status == 101:
                    QMessageBox.about(self, '连接状态', '连接成功')
                    self.text.setFocus()
                connect_flage = False
                self.register_flag = True

                print("register_flag", self.register_flag)

            except Exception as e:
                QMessageBox.about(self, '连接状态', '连接失败')
                self.text.setFocus()
                self.register_flag = False
                print("register_flag", self.register_flag)
                break

        # print(add)

        # print(self.ws.status)
        # print(self.ws.connected)

        """
        if self.ws.status != 101:
            QMessageBox.about(self, '连接状态', '连接失败')
            self.text.setFocus()
        """


    def close_showMessage(self):
        self.ws.close()
        QMessageBox.about(self, '连接状态', '连接断开')
        self.text.setFocus()



    # 注册按钮事件处理
    def register_showMessage(self):
        RegistComm = {
            "CommandHandler": "AIColor",
            "CommandName": "RegistComm",
            "ParamsJson": ""}  # 注册信息需要为 Json 格式
        RegistComm = json.dumps(RegistComm)
        print(self.ws.status)
        while self.register_flag:
            try:
                self.ws.send(RegistComm)  # 发送注册信息
                RegResult = self.ws.recv()  # 返回的是字符串
                RegResult = json.loads(RegResult)  # 转换成字典格式
                # print(type(RegResult))
                # print(type(RegResult['success']))

                if bool(RegResult['success']):
                    QMessageBox.about(self, '注册状态', '注册成功')
                    self.text.setFocus()
                    break
            except Exception as e:
                QMessageBox.about(self, '注册状态', '注册失败')
                self.text.setFocus()
                break

        while self.register_flag == False:
            try:
                QMessageBox.about(self, '注册状态', '注册失败')
                self.text.setFocus()
                break

            except Exception as e:
                break


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
