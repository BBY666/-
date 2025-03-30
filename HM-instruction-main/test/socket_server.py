import socket
import wave

from paddlespeech.cli.tts.infer import TTSExecutor

# 定义服务器的地址和端口
HOST = '10.24.1.51'
PORT = 65432

# 创建一个TCP/IP套接字
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            
            # 接收客户端发送的文本
            data = conn.recv(1024)
            if not data:
                break
            text = data.decode('utf-8')
            print(f"Received text: {text}")
            
            received_text = text

            
            tts = TTSExecutor()
            tts(text=received_text , output="output.wav")

            # 生成WAV文件（这里使用一个预先存在的WAV文件作为示例）
            wav_filename = 'output.wav'

            # 读取WAV文件
            with open(wav_filename, 'rb') as wav_file:
                wav_data = wav_file.read()

            # 发送WAV文件长度
            conn.sendall(len(wav_data).to_bytes(4, 'big'))

            # 发送WAV文件内容
            conn.sendall(wav_data)
            print("WAV file sent")