import threading
import socket
import time
import os
import csv
from label_data import predict_label

LABEL = ''
PATH = "*.csv"
host, port = ['192.168.1.20', 12345]

def decode_file(PATH):
    with open(PATH, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        first_row = next(csv_reader)
        print(first_row[0])
        return first_row[0]

def get_label():
    global LABEL

    while True:
        if os.path.exists(PATH):       
            LABEL = predict_label(PATH)
            print(LABEL)
            os.remove(PATH)

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
    if LABEL:
        client_socket.send(LABEL.encode('utf-8'))
        print(f"Sent label: {LABEL}")
        LABEL = ''

client_socket.close()
