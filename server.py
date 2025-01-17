import sys
import time
import multiprocessing
import threading

class setInterval:
    def __init__(self, interval: float, action):
        self.interval = interval
        self.action = action
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.__setInterval)
        self.thread.start()

    def __setInterval(self):
        nextTime = time.time() + self.interval
        while not self.stopEvent.wait(nextTime-time.time()):
            nextTime += self.interval
            if state.value != 4:
                self.action()
                inter.cancel()

    def cancel(self):
        self.stopEvent.set()

standalone: bool = False
dims = multiprocessing.Array('d', 6)
conf = multiprocessing.Array('d', 2)
state = multiprocessing.Value('i')

def internal_runner(dims, conf, state):
    import socket
    import json

    try:
        HOST = '127.0.0.1'
        PORT = 1337

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Listening on {HOST}:{PORT}")

        while True:
            state.value = 4
            try:
                connection = server.accept()[0]
                state.value = 3
                while True:
                    data = connection.recv(1024)
                    if data:
                        msg = json.loads(data)
                        # print(msg)
                        for i in range(4):
                            dims[i] = msg[i]
                        dims[4] = dims[0] + dims[2]/2
                        dims[5] = dims[1] + dims[3]/2
                        conf[0] = msg[4]
                        conf[1] = msg[5]
                        state.value = msg[6]
                        
                        if standalone:
                            printOut()
                    else:
                        break
            except:
                connection.close()
    finally:
        sys.exit(0)

def printOut():
    print(list(dims), list(conf), state.value)
    
def getDims():
    return tuple(dims)

def getConf():
    return tuple(conf)

def getStatus():
    return state.value

def isConnected():
    return False if state.value == 4 else True

def noop():
    pass

def common_handler():
    p1.terminate()
    try:
        p1.kill()
    finally:
        p1.join()
        p1.close()

def unix_handler(sig, frame):
    common_handler()
    sys.exit(0)

def win32_handler(a):
    common_handler()
    sys.exit(0)

def start(callback=noop):
    global inter, p1
    
    p1 = multiprocessing.Process(None, internal_runner, None, (dims, conf, state), daemon=True)

    inter = setInterval(.2, callback)

    if sys.platform == "win32":
        import win32api
        win32api.SetConsoleCtrlHandler(win32_handler, True)
    else:
        import signal
        signal.signal(signal.SIGINT, unix_handler)
        
    p1.start()

if __name__ == "__main__":
    standalone = True
    start()
    p1.join()