import socket

from paddlespeech.cli.asr.infer import ASRExecutor
#from paddlespeech.cli.tts.infer import TTSExecutor

# 定义服务器的地址和端口
HOST = '10.24.5.36'
PORT = 65431

def process_wav_file(wav_data):
    # 这里可以实现对WAV文件的处理逻辑
    # 例如：语音识别，将音频转换为文本
    # 这里假设返回固定文本
    return "This is a processed text from the WAV file."

# 创建一个TCP/IP套接字
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)

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

            received_wav_filename = 'received.wav'
            with open(received_wav_filename, 'wb') as wav_file:
                wav_file.write(wav_data)
            print(f"WAV file saved as '{received_wav_filename}'")

            # 处理WAV文件并生成文本
            asr = ASRExecutor()
            processed_text = asr(audio_file="received.wav")

            # 发送处理后的文本
            conn.sendall(processed_text.encode('utf-8'))
            print("Processed text sent")