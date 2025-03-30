import socket


# 这里是一个翻译的函数

def translate():

def send_image_and_text(image_path, text, server_host='10.50.0.37', server_port=8050):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_host, server_port))

    with open(image_path, 'rb') as f:
        image_data = f.read()

    text_data = text.encode('utf-8')

    try:
        # 发送图片数据长度
        client.send(len(image_data).to_bytes(4, byteorder='big'))
        # 发送图片数据
        client.send(image_data)

        # 发送文本数据长度
        client.send(len(text_data).to_bytes(4, byteorder='big'))
        # 发送文本数据
        client.send(text_data)

        # 接收结果长度
        length_data = client.recv(4)
        if not length_data:
            print("No response from server")
            return
        length = int.from_bytes(length_data, byteorder='big')

        # 接收结果
        result_data = client.recv(length).decode('utf-8')
        return result_data

    except Exception as e:
        print(f"Error: {e}")

    client.close()