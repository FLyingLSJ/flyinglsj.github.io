import json
import websocket
import multiprocessing as mp
import time
flag = True
header = {
    "CommandHandler": "AIColor",
    "CommandName": "RegistComm",
    "ParamsJson": ""}  # 注册信息需要为 Json 格式
#header = dict(header)
header = json.dumps(header)


try:
    import thread
except ImportError:
    import _thread as thread


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



if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(url='ws://127.0.0.1:2019',
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()



