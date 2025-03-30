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
    # 10.24.5.36
    url = 'http://10.24.5.36:5000/asr'
    #url = 'http://10.24.7.245:5000/asr'

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
        return(output_name)
    else:
        print('请求失败，状态码：', response.status_code)


# m4t_predict <输入文本> --task t2tt --tgt_lang <目标语言> --src_lang <源语言>



def get_instructions(text):
    print(text)
    url = 'http://10.50.0.37:5001/chat'
    input_data = {
        'input': f'你是一名操作机器狗的专家。我将提供给你一段描述机器狗动作的文字。你的任务是提取必要的指令，例如"前进"、"后退"、"停止"、"左转"、"右转"、"伸懒腰"、"坐下"、"站起来"，如果文本中包含相关语义的话，将其转化为基本指令并提取动作的程度，例如前进多少米，向左转多少度，或向右转多少度。以JSON格式返回动作、速度和动作程度。如果没有指定程度，"degrees"字段应为1。\
        例如：\
        - 文字："前进一米" -> 返回：`{{"action": "前进", "degrees": "1"}}`\
        - 文字："后退一米" -> 返回：`{{"action": "后退", "degrees": "1"}}`\
        - 文字："停止" -> 返回：`{{"action": "停止", "degrees": "0"}}`\
        - 文字："左转三十度" -> 返回：`{{"action": "左转", "degrees": "30"}}`\
        - 文字："站起来" -> 返回：`{{"action": "站起来", "degrees": "0"}}`\
        - 文字："趴下" -> 返回：`{{"action": "趴下", "degrees": "0"}}`\
        以下是我提供的文字：{text}'
    }
    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(input_data))
    
    print(json.loads(response.json()['response'].replace('json\n',"").replace("`","")))
    return json.loads(response.json()['response'].replace('json\n',"").replace("`",""))

# text_result= get_instructions("前进五十米再右转六十度，再前进二十米")
# print(text_result)

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
        # response_data = json.loads(response_str)  
        response_data=response_str
        print('Response Data:', response_data)
    else:
        print('Error:', response.status_code, response.text)
    return response_data