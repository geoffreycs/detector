import socket
import json
import sys, signal

def unix_handler(sig, frame):
    sys.exit(0)

def win32_handler(a):
    sys.exit(0)

if sys.platform == "win32":
    import win32api
    win32api.SetConsoleCtrlHandler(win32_handler, True)
else:
    signal.signal(signal.SIGINT, unix_handler)

HOST = '127.0.0.1'
PORT = 1337

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"Listening on {HOST}:{PORT}")

while True:
    global connection
    try:
        connection = server.accept()[0]
        while True:
            data = connection.recv(1024)
            if data:
                msg = json.loads(data)
                print(msg)
            else:
                break
    except:
        connection.close()
