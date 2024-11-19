import threading
import socket
import time
import os
from label_data import predict_label

host, port = ['192.168.1.20', 12345]
model_path = r'20241119-085834.h5'
LABEL = ''

def get_label():
    global LABEL
    script_dir = os.path.dirname(os.path.abspath(__file__))

    while True:
        csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            data_path = os.path.join(script_dir, csv_file)
            LABEL = predict_label(data_path, model_path)
            print(f"LABEL: {LABEL}")
            os.remove(data_path)


def connect_to_server():
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))
            print(f"Connected to server {host}:{port}")
            return client_socket
        except socket.error as e:
            print(f"Connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

#client_socket = connect_to_server()

label_thread = threading.Thread(target=get_label, daemon=True)
label_thread.start()

while True:
    print('t')
    time.sleep(100)

'''
while True:
    if LABEL:
        client_socket.send(LABEL.encode('utf-8'))
        print(f"Sent label: {LABEL}")
        LABEL = ''

client_socket.close()
'''
