import socket
import threading

data = ''

def start_server(host, port):
    global data
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen(1)

        while True:
            connection, client_address = server_socket.accept()
            with connection:
                print(client_address)
                while connection:
                    raw_data = (connection.recv(1024))
                    data = raw_data.decode('utf-8')
                    if data:
                        print(data)

def get_sensor_data():
    global data
    data_copy = data
    data = ''
    return data_copy

def main():
    print("Server is running...")
    # Any server initialization or startup steps can be added here

if __name__ == "__main__":
    main()

