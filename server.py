import socket

# Server IP and port
HOST = '127.0.0.1'  # Localhost for local testing
PORT = 1337        # Port to listen on

# Create a TCP socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
server.bind((HOST, PORT))

# Listen for incoming connections
server.listen(1)
print(f"Listening on {HOST}:{PORT}")

while True:
    # Wait for a connection
    connection, client_address = server.accept()
    try:
        print('Client connected:', client_address)

        # Receive the data
        while True:
            data = connection.recv(1024)
            if data:
                print('Received:', data.decode())
            else:
                break
    finally:
        # Clean up the connection
        connection.close()
