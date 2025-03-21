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
        self.ws = 1

    def initUI(self):
        self.setGeometry(300, 300, 800, 800)
        self.setWindowTitle('websocket')
        self.setWindowIcon(QIcon('xdbcb8.ico'))

        # 连接按键界面设置
        self.connect_button = QPushButton('open', self)
        self.connect_button.setGeometry(30, 150, 70, 30)
        self.connect_button.setToolTip('<b>连接</b>')
        self.connect_button.clicked.connect(self.connect_showMessage)

        # 注册按键界面设置
        self.register_button = QPushButton('register', self)
        self.register_button.setGeometry(115, 150, 100, 30)
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
        add = str(self.text.text())  # ws://127.0.0.1:2019
        # print(add)
        self.ws = create_connection(add)
        # print(self.ws.status)
        # print(self.ws.connected)
        if self.ws.status == 101:
            QMessageBox.about(self, '连接状态', '连接成功')
            self.text.setFocus()

    # 注册按钮事件处理
    def register_showMessage(self):
        RegistComm = {
            "CommandHandler": "AIColor",
            "CommandName": "RegistComm",
            "ParamsJson": ""}  # 注册信息需要为 Json 格式
        RegistComm = json.dumps(RegistComm)
        self.ws.send(RegistComm)  # 发送注册信息
        RegResult = self.ws.recv()  # 返回的是字符串
        RegResult = json.loads(RegResult)  # 转换成字典格式
        # print(type(RegResult))
        # print(type(RegResult['success']))

        if bool(RegResult['success']):
            QMessageBox.about(self, '注册状态', '注册成功')
            self.text.setFocus()


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
