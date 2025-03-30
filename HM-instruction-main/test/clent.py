import socket

# 定义服务器的地址和端口
HOST = '10.24.7.242'
PORT = 65432

# 读取要发送的WAV文件
wav_filename = 'test.wav'
with open(wav_filename, 'rb') as wav_file:
    wav_data = wav_file.read()

# 创建一个TCP/IP套接字
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    # 发送WAV文件长度
    s.sendall(len(wav_data).to_bytes(4, 'big'))

    # 发送WAV文件内容
    s.sendall(wav_data)
    print("WAV file sent")

    # 接收处理后的文本
    data = s.recv(1024)
    processed_text = data.decode('utf-8')
    print(f"Received processed text: {processed_text}")