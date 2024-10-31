import sys
import os
import signal
import time
import multiprocessing
import threading


# class setInterval:
#     def __init__(self, interval: float, action):
#         self.interval = interval
#         self.action = action
#         self.stopEvent = threading.Event()
#         self.thread = threading.Thread(target=self.__setInterval)
#         self.thread.start()

#     def __setInterval(self):
#         nextTime = time.time() + self.interval
#         while not self.stopEvent.wait(nextTime-time.time()):
#             nextTime += self.interval
#             self.action()

#     def cancel(self):
#         self.stopEvent.set()
#         self.thread.join()


dims = multiprocessing.Array('d', 4)
conf = multiprocessing.Array('d', 2)
state = multiprocessing.Array('i', 4)


def runner(dims, conf, state):
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
            state[3] = 1
            try:
                connection = server.accept()[0]
                state[3] = 0
                while True:
                    data = connection.recv(1024)
                    if data:
                        msg = json.loads(data)
                        for i in range(4):
                            dims[i] = msg[0][i]
                        conf[0] = msg[0][4]
                        conf[1] = msg[0][5]
                        state[0] = msg[1]['0']
                        state[1] = msg[1]['1']
                        state[2] = msg[1]['2']
                    else:
                        break
            except:
                connection.close()
    finally:
        sys.exit(0)

def printOut():
    print(list(dims), list(conf), list(state))

def start():
    p1 = multiprocessing.Process(None, runner, None, (dims, conf, state), daemon=False)

    # inter = setInterval(.25, printOut)

    def common_handler():
        # inter.cancel()
        p1.terminate()
        try:
            p1.kill()
        finally:
            p1.join()
            p1.close()

    def unix_handler(sig, frame):
        sys.stderr = os.open(os.devnull, os.O_RDWR)
        common_handler()
        sys.exit(0)

    def win32_handler(a):
        common_handler()
        sys.exit(0)

    if sys.platform == "win32":
        import win32api
        win32api.SetConsoleCtrlHandler(win32_handler, True)
    else:
        signal.signal(signal.SIGINT, unix_handler)
        
    p1.start()

if __name__ == "__main__":
    start()