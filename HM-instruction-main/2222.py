import socket
import requests
import json

# 要发送的文本和图像路径
text = '你看到了框了吗，有几个'
image_path = '/home/smbu/BBY/Qwen-VL-master/2.jpg'


import requests
import json

def send_text_and_image(text, image_path, ip="10.24.5.36", port=8000):
    # 拼接 URL
    url = f"http://{ip}:{port}/process"
    
    # 打开图像文件并发送请求
    with open(image_path, 'rb') as img_file:
        response = requests.post(
            url,
            files={'image': img_file},
            data={'text': text}
        )
    
    # 检查响应是否成功
    if response.status_code == 200:
        try:
            # 解析 JSON 响应
            data = response.json()
            
            # 提取 history 和 response
            history = data.get("history", [])
            response_text = data.get("response", "")
            
            # 打印并翻译响应内容
            print("History:")
            for item in history:
                print("对话内容:", item[0])
                print("回答:", item[1])
            
            print("\nResponse:")
            print("回复:", response_text)
            
            return data  # 返回完整数据
        except json.JSONDecodeError:
            print("解析 JSON 失败")
            return None
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None

# 调用函数并传入文本、图像路径、IP 地址和端口号
text = '坐下'
image_path = '/home/smbu/BBY/Qwen-VL-master/2.jpg'
#text = '你好，图片里面是什么，向右两米'
input_data = 'input:' f'你是一名操作机器狗的专家。你的任务是提取必要的指令，例如"前进"、"后退"、"停止"、"左转"、"右转"、"伸懒腰"、"坐下"、"站起来"，如果文本中包含相关语义的话，将其转化为基本指令并提取动作的程度，例如前进多少米。以JSON格式返回动作、速度和动作程度。如果没有指定程度，"degrees"字段应为0。例如文字："前进一米" -> `{{"action": "前进", "degrees": "1"}}`文字："停止" -> 返回：`{{"action": "停止", "degrees": "0"}}`文字："左转三十度" -> `{{"action": "左转", "degrees": "30"}}`文字："向左三米" -> `{{"action": "向左", "degrees": "3"}}`文字："趴下" -> `{{"action": "趴下", "degrees": "0"}}`如果文字中不存在包含的指令。例如：文字：你看到了什么->`{{"action": "无", "degrees": "0"}}`以下是我提供的文字：{text}'

send_text_and_image(input_data, image_path)



'''

# 打开图像文件并构建请求
with open(image_path, 'rb') as img_file:
    response = requests.post(
        "http://127.0.0.1:8000/process",
        files={'image': img_file},
        data={'text': text}
    )



# 输出服务器响应

print(response.text)

# 解析 JSON 数据
data = json.loads(response.text)

# 提取内容并进行 Unicode 解码
history = data["history"]
response = data["response"]

# 打印并检查内容
print("History:")
for item in history:
    print("对话内容:", item[0])
    print("回答:", item[1])

print("\nResponse:")
print("回复:", response)

'''








'''

# 127.0.0.2
def send_image_and_text(image_path, text, server_host='0.0.0.0', server_port=8000):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_host, server_port))

    with open(image_path, 'rb') as f:
        image_data = f.read()

    # 替换掉 \x08 字符（退格符），确保不影响后续操作
    image_data = image_data.replace(b"\x08", b"")

    text_data = text.encode('utf-8')

    
    #image_data = image_data.replace(b"\x08", "")

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

# Qwen-VL-master/2.jpg
image_path = '/home/smbu/BBY/Qwen-VL-master/2.jpg'
#image_path = '../Qwen-VL-master/2.jpg'

text = 'hi.'
text = text.replace("\x08", "")

print(send_image_and_text(image_path, text))
'''

# http://127.0.0.2:5001/chat