import pyaudio
import wave


CHUNK = 1024


def play_pcm(file_path, channels=1, rate=16000, width=2):
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=p.get_format_from_width(width),
                    channels=channels,
                    rate=rate,
                    output=True)

    with open(file_path, 'rb') as pcmfile:
        data = pcmfile.read(1024)
        while data:
            stream.write(data)
            data = pcmfile.read(1024)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PortAudio interface
    p.terminate()


def record_audio(filename, duration):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def pcm_to_wav(pcm_file_path, wav_file_path):
    sample_rate = 16000  # 例如 16 kHz
    bit_depth = 16       # 例如 16-bit

    # 读取 PCM 数据
    with open(pcm_file_path, 'rb') as f:
        pcm_data = f.read()
    
    # 根据位深度将 PCM 数据转换为 numpy 数组
    dtype = np.int16 if bit_depth == 16 else np.uint8
    audio_data = np.frombuffer(pcm_data, dtype=dtype)

    # 保存为 WAV 文件
    write(wav_file_path, sample_rate, audio_data)

def play_wav_file(file_path):
    # 打开 .wav 文件
    wf = wave.open(file_path, 'rb')

    # 创建 PyAudio 流
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 读取并播放音频数据
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    # 关闭流和 PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

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
        return(output_name)
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