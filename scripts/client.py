import threading
import socket
import time
import os
from label_data import predict_label
import glob


host, port = ['192.168.1.20', 12345]
model_path = r'20241119-085834.h5'
LABEL = ''

def get_label():
    global LABEL
    
    while True:
        data_path = glob.glob('*.csv')
        if data_path:
            LABEL = predict_label(data_path[0], model_path)
            print(f"Predicted label: {LABEL}")
            os.remove(data_path[0])
        time.sleep(0.2)

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

client_socket = connect_to_server()

label_thread = threading.Thread(target=get_label, daemon=True)
label_thread.start()

while True:
    if LABEL != '':
        client_socket.send(str(LABEL).encode('utf-8'))
        print(f"Sent label: {LABEL}")
        LABEL = ''
    time.sleep(0.2)

client_socket.close()
