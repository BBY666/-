# from paddlespeech.server.bin.paddlespeech_client import ASRClientExecutor

# asrclient_executor = ASRClientExecutor()
# res = asrclient_executor(
#     input="/home/gr/dog/zh.wav",
#     server_ip="127.0.0.1",
#     port=8090,
#     sample_rate=16000,
#     lang="zh_cn",
#     audio_format="wav")
# import torch
# print(torch.backends.cudnn.version())

# print(res)
# from paddlespeech.cli.asr.infer import ASRExecutor # 语音转文本
# asr = ASRExecutor()
# result = asr(audio_file="two.wav")
# print(result)



# from paddlespeech.cli.tts.infer import TTSExecutor # 文本转语音
# tts = TTSExecutor()
# tts(text="范忆萌干什么呢？", output="output.wav")

import requests
import json
from pydub import AudioSegment

# 音频文件修改为服务器需要的格式
# 输入：需要修改的音频文件的路径
# 输出：输出的音频文件路径
def convert_to_16k(input_wav_path, output_wav_path):
    # 加载音频文件
    audio = AudioSegment.from_wav(input_wav_path)
    
    # 设置采样率为16kHz
    audio = audio.set_frame_rate(16000)
    
    # 保存新的音频文件
    audio.export(output_wav_path, format="wav")


# 输入的是音频文件的路径
def stt(input_path): 
    # 音频文件的本地路径

    # print('down')
    # Flask 服务端的 URL 和端口号
    url = 'http://10.24.7.245:5000/asr'

    # 打开音频文件并读取内容
    with open(input_path, 'rb') as audio_file:
        # 构造文件上传的表单数据
        files = {'file': (audio_file.name, audio_file, 'audio/wav')}
        
        # 发送 POST 请求到 Flask 服务端
        response = requests.post(url, files=files)
        
        # 检查请求是否成功
        if response.status_code == 200:
            # 解析返回的 JSON 数据
            data = response.json()
            # 打印 ASR 结果
            print("ASR Result:", data['result'])
            return(data['result'])
        else:
            # 打印错误信息
            print("Error:", response.json()['error'])

# 输入的是一段字符串
def tts(text, output_name):
    # 定义 Flask 服务端的 URL 和端口
    url = 'http://10.24.7.245:5000/tts'

    # 定义要合成的文本
    # text_to_speak = '今天的天气不错！'

    # 构建请求的参数
    params = {
        'text': text
    }

    # 发送 GET 请求到 Flask 服务端
    response = requests.get(url, params=params)

    # 检查请求是否成功
    if response.status_code == 200:
        # 获取响应的内容
        audio_content = response.content

        # 将语音文件保存到本地
        with open(output_name, 'wb') as audio_file:
            audio_file.write(audio_content)
        print(f'语音文件已保存为{output_name}')
        return('received_audio.wav')
    else:
        print('请求失败，状态码：', response.status_code)

# 将语音修改成需要的类型 ： 16k wav文件
# chusi_wav = 'output.wav'
# fixed_wav = 'fixed.wav'
# convert_to_16k(chusi_wav,fixed_wav)

# stt(fixed_wav)

# text = '吾皇万岁万岁万万岁！'
# tts(text)
def get_instructions(text):
    url = 'http://10.50.0.37:5001/chat'

    input_data = {
        'input': f'你是一个机器狗使用专家，我将给你一段描述机械狗运动的文字，\
        你给我提取出来机械狗需要进行的动作，如前进，左转，右转等基本动作，如果文字里包含的话，提取出来运动程度，如前进的是多少米，\
        向左转多少度，向右转多少度等，并且按照json格式返回动作以及动作程度。\
        如{{"actions":"前进","degrees":"50"}}。，如果有多个动作就将其处理为一个list，list中的每一项就是一个动作\
        文字：{text}'
    }

    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(input_data))
    response_data = None
    if response.status_code == 200:
        response_json = response.json()  
        response_str = response_json['response']  
        print('Response Data:', response_str)
        response_data = json.loads(response_str)  
        print('Response Data:', response_data)
        print('Actions:', response_data['actions'])  
        print('Degrees:', response_data['degrees'])  
    else:
        print('Error:', response.status_code, response.text)
    return response_data

text_result= get_instructions("前进五十米再右转六十度，再前进二十米")
print(text_result)

def get_key_instructions(text):
    url = 'http://10.50.0.37:5001/chat'

    input_data = {
        'input': f'你是一个机器狗使用专家，请你将接下来的这句话提炼为一句对机器狗施加的指令：{text}'
    }

    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(input_data))
    response_data = None
    if response.status_code == 200:
        response_json = response.json()  
        response_str = response_json['response']  
        print('Response Data:', response_str)
        response_data = json.loads(response_str)  
        print('Response Data:', response_data)
    else:
        print('Error:', response.status_code, response.text)
    return response_data