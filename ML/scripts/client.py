import threading
import socket
import time
import os
import glob
from label_data import predict_label

host, port = ['192.168.1.20', 12345]
model_path = r'scripts\model\trained_model_20241121-175654.h5' # Path to the trained model
LABEL = ''
running = True

def get_label():
    global LABEL, running

    while running:
        data_path = glob.glob('*.csv')
        if data_path:
            LABEL = predict_label(data_path[0], model_path)
            print(f"LABEL: {LABEL}")
            os.remove(data_path[0])
        
        time.sleep(0.2)

def connect_to_server():
    while running:
        print(f"Connecting to server...")
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))
            print(f"Connected to server {host}:{port}")
            return client_socket
        except socket.error as e:
            print(f"Connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def user_input_listener():
    global running
    while running:
        user_input = input("Type 'exit' to close the client: ").strip().lower()
        if user_input == 'exit':
            running = False
            break

def send_labels_to_server():
    global client_socket, running, LABEL
    while running:
        try:
            if LABEL != '':
                client_socket.send(str(LABEL).encode('utf-8'))
                print(f"Sent label: {LABEL}")
                LABEL = ''
        except (socket.error, BrokenPipeError):
            print("Lost connection to server. Attempting to reconnect...")
            client_socket = connect_to_server()  # Reconnect to the server

# Start the connection to the server
client_socket = connect_to_server()

# Start the label processing thread
label_thread = threading.Thread(target=get_label, daemon=True)
label_thread.start()

# Start the user input listener thread
input_thread = threading.Thread(target=user_input_listener, daemon=True)
input_thread.start()

# Start the label sending thread
label_sender_thread = threading.Thread(target=send_labels_to_server, daemon=True)
label_sender_thread.start()

try:
    while running:
        time.sleep(0.1)  # Keep the main thread alive
except KeyboardInterrupt:
    print("Interrupted by user. Closing client...")
finally:
    running = False
    client_socket.close()
    print("Client closed.")
