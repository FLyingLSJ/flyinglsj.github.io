# !/usr/bin/python3

# coding = utf-8

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QLineEdit
from PyQt5.QtGui import QIcon
from websocket import create_connection, WebSocket
import websocket
import json
try:
    import thread
except ImportError:
    import _thread as thread

header = {
    "CommandHandler": "AIColor",
    "CommandName": "RegistComm",
    "ParamsJson": ""}  # 注册信息需要为 Json 格式
#header = dict(header)
header = json.dumps(header)



class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        self.setGeometry(200, 200, 500, 500)
        self.setWindowTitle('websocket')
        self.setWindowIcon(QIcon('xdbcb8.ico'))

        # 连接按键界面设置
        self.connect_button = QPushButton('open', self)
        self.connect_button.setGeometry(30, 150, 70, 30)
        self.connect_button.setToolTip('<b>连接</b>')
        self.connect_button.clicked.connect(self.connect_showMessage)

        # 断开连接按键界面设置
        self.close_button = QPushButton('close', self)
        self.close_button.setGeometry(30, 180, 70, 30)
        self.close_button.setToolTip('<b>断开</b>')
        self.close_button.clicked.connect(self.close_showMessage)

        # starting
        self.starting_button = QPushButton('starting', self)
        self.starting_button.setGeometry(30, 210, 70, 30)
        self.starting_button.setToolTip('<b>开始</b>')
        self.starting_button.clicked.connect(self.starting_showMessage)


        # 文本输入框界面设置
        self.text = QLineEdit('ws://127.0.0.1:2019', self)
        self.text.selectAll()
        self.text.setFocus()
        self.text.setGeometry(80, 50, 220, 30)

        self.show()

    # 连接按钮事件处理
    def connect_showMessage(self):
        QMessageBox.about(self, '连接状态', '连接成功')
        self.text.setFocus()
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(url='ws://127.0.0.1:2019',
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        print("建立连接")


    def close_showMessage(self):
        QMessageBox.about(self, '连接状态', '断开连接')
        self.text.setFocus()



    def starting_showMessage(self):
        QMessageBox.about(self, '状态', '开始')
        self.text.setFocus()
        self.ws.on_open = self.on_open
        self.ws.run_forever()



    ########################
    def on_message(self, message):

        print(message)



    def on_error(self, error):
        print(error)

    def on_close(self, ws):
        print("### closed ###")

    def on_open(self):
        global header

        def run(*args):
            self.ws.send(header)
            print("thread terminating...")

        thread.start_new_thread(run, ())
    ########################



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


