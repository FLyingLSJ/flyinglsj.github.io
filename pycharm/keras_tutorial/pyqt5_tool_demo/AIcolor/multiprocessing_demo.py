import multiprocessing as mp
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QLineEdit, QTextBrowser, QComboBox
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QIcon
from websocket import WebSocket
import json
import time
import websocket


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


def on_message(ws, message):
    print(message)



def on_error(ws, error):
    print(error)


def on_close(ws):
    print("### closed ###")


def on_open(ws):
    def run(*args):
        ws.send(header)
        print("thread terminating...")
    thread.start_new_thread(run, ())


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

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
        self.show()

    def connect_showMessage(self):
        #ws.on_open = on_open
        #ws.run_forever()
        p1.start()
        p1.join()
        print("connect_showMessage")

    def close_showMessage(self):
        print("close_showMessage")

    def starting_showMessage(self):
        print("starting_showMessage")

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

def job():
    while 1:
        print("....")


if __name__ == '__main__':
    q = mp.Queue()
    app = QApplication(sys.argv)
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(url='ws://127.0.0.1:2019',
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ex = Example()
    p1 = mp.Process(target=job, args=())


    sys.exit(app.exec_())


"""
import multiprocessing as mp
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QLineEdit, QTextBrowser, QComboBox
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QIcon
from websocket import WebSocket
import json
import time




class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.ws = 1  # 初始化 wbsocket 的参数
        self.job_flag = [False, False]

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
        self.show()

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


                    connect_flage = False
            except Exception as e:
                self.textBrowser.append(str(time.ctime()) + ": 连接失败")
                QMessageBox.about(self, '连接状态', '连接失败')
                self.text.setFocus()
                self.register_flag = False
                #print("register_flag", self.register_flag)
                break

    def close_showMessage(self):
        print("close_showMessage")

        # self.ws.close()
        # QMessageBox.about(self, '连接状态', '断开连接')
        # self.text.setFocus()



    def starting_showMessage(self):
        self.threading_demo()
        #print("starting_showMessage")

        # Recognize_info = self.ws.recv()  # 接收到的是字符串
        # Recognize_info = json.loads(Recognize_info)  # 转化成字典
        # print(Recognize_info)


    def job_a(self):
        while self.job_flag[0]:
            print("job_a")

    def job_b(self):
        while self.job_flag[1]:
            print("job_b")


    def threading_demo(self):
        print("开启进程")




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
    q = mp.Queue()
    ex = Example()

    p1 = mp.Process(target=ex.job_a, args=(q,))
    #p2 = mp.Process(target=ex.job_b, args=(q,))
    p1.start()
    #p2.start()
    p1.join()
    #p2.join()
    sys.exit(app.exec_())
"""
