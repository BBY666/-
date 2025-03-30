import socket
import wave
import os
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.cli.asr.infer import ASRExecutor

# 定义服务器的地址和端口
HOST = '10.24.5.36'
PORT = 5000

def process_text(text):
    tts = TTSExecutor()
    wav_filename = 'output.wav'
    tts(text=text, output=wav_filename)
    return wav_filename

def process_wav_file(wav_data):
    asr = ASRExecutor()
    processed_text = asr(audio_file="received.wav")
    return process_text

# 创建一个TCP/IP套接字
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            
            # 接收类型标识符 (1 字节)
            data_type = conn.recv(1)
            if not data_type:
                break

            if data_type == b'T':  # 文本数据
                # 接收文本数据长度
                data = conn.recv(4)
                text_length = int.from_bytes(data, 'big')

                # 接收文本数据
                text_data = conn.recv(text_length).decode('utf-8')
                print(f"Received text: {text_data}")

                # 处理文本并生成WAV文件
                wav_filename = process_text(text_data)
                
                # 读取WAV文件
                with open(wav_filename, 'rb') as wav_file:
                    wav_data = wav_file.read()

                # 发送WAV文件长度
                conn.sendall(len(wav_data).to_bytes(4, 'big'))

                # 发送WAV文件内容
                conn.sendall(wav_data)
                print("WAV file sent")

            elif data_type == b'W':  # WAV文件数据
                # 接收WAV文件长度
                data = conn.recv(4)
                wav_length = int.from_bytes(data, 'big')
                print(f"Expected WAV file length: {wav_length} bytes")

                # 接收WAV文件内容
                wav_data = b''
                while len(wav_data) < wav_length:
                    packet = conn.recv(1024)
                    if not packet:
                        break
                    wav_data += packet

                print("WAV file received")

                # 将接收到的WAV文件存储下来
                received_wav_filename = 'received.wav'
                with open(received_wav_filename, 'wb') as wav_file:
                    wav_file.write(wav_data)
                print(f"WAV file saved as '{received_wav_filename}'")

                # 处理WAV文件并生成文本
                processed_text = process_wav_file(wav_data)

                # 发送处理后的文本
                conn.sendall(processed_text.encode('utf-8'))
                print("Processed text sent")