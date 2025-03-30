
import requests
import json
import time
import hashlib
import base64
import websocket
import socket
import ssl
import threading
from pydub import AudioSegment
from pydub.playback import play
import hmac
import os
import urllib
from all_for_audio.En2Cn import get_result

from ipdb import set_trace
import pyaudio
import wave
import hashlib
import base64
import hmac
from all_for_audio.test_audi import main_fun
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from time import mktime
import _thread as thread
import os
import time
from all_for_audio.prompt_example import get_instructions
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
import math
from scipy.io.wavfile import write
import numpy as np
import all_for_audio.speech_utils as tool
from PIL import Image
from io import BytesIO


from TTS.api import TTS
import whisper
import torch
import re

# Recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 4.5
WAVE_OUTPUT_FILENAME = "recorded.wav"
wsParam = 0



#robot control
class SportModeTest:
    def __init__(self) -> None:
        # Time count
        self.t = 0
        self.dt = 0.01

        # Initial poition and yaw
        self.px0 = 0
        self.py0 = 0
        self.yaw0 = 0

        self.client = SportClient()  # Create a sport client
        self.client.SetTimeout(10.0)
        self.client.Init()

        # 全局事件，用于控制函数执行
        self.execution_event = threading.Event()

        # 当前执行的函数
        self.current_function = None

    def GetInitState(self, robot_state: SportModeState_):
        self.px0 = robot_state.position[0]
        self.py0 = robot_state.position[1]
        self.yaw0 = robot_state.imu_state.rpy[2]

    def StandUpDown(self,measurement,velocity=1):
        self.client.StandDown()
        print("Stand down !!!")
        time.sleep(1)

        self.client.StandUp()
        print("Stand up !!!")
        time.sleep(1)

        self.client.StandDown()
        print("Stand down !!!")
        time.sleep(1)

        self.client.Damp()

    def VelocityMove(self,measurement,velocity=1):
        elapsed_time = 1
        for i in range(int(elapsed_time / self.dt)):
            self.client.Move(0.3, 0, 0)  # vx, vy vyaw
            time.sleep(self.dt)
        self.client.StopMove()

    def Move_forward(self,measurement,velocity=1):
        elapsed_time = measurement*4/velocity*1
        v=velocity/1*0.24
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(v, 0, 0)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break

    def StopMove(self,measurement,velocity=1):
        self.client.StopMove()

    def Move_backward(self,measurement,velocity=1):
        elapsed_time = measurement*5.5/velocity*1
        v = velocity / 1 * 0.24
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(-v, 0, 0)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break

    def Move_left(self,measurement,velocity=1):
        elapsed_time = measurement*5.5/velocity*1
        v = velocity / 1 * 0.24
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(0, v, 0)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break
        
    def Move_right(self,measurement,velocity=1):
        elapsed_time = measurement*5.5/velocity*1
        v = velocity / 1 * 0.24
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(0, -v, 0)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break
    
    def Move_cycle_right(self,measurement,velocity=1):
        elapsed_time = (float(measurement)/90.0)*2
        # v = velocity / 5 * 1
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(0.1, 0, -1)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break

    def Move_cycle_left(self,measurement,velocity=1):
        elapsed_time = (float(measurement)/90.0)*2
        # v = velocity / 5 * 1
        while True:
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(0.1, 0, 1)  # vx, vy vyaw
                if self.execution_event.is_set():
                    break
                time.sleep(self.dt)
            self.client.StopMove()
            break

    def BalanceAttitude(self,measurement,velocity=1):
        self.client.Euler(0.1, 0.2, 0.3)  # roll, pitch, yaw
        self.client.BalanceStand()

    def TrajectoryFollow(self,measurement=1,velocity=1):
        time_seg = 0.2
        time_temp = self.t - time_seg
        path = []
        for i in range(SPORT_PATH_POINT_SIZE):
            time_temp += time_seg

            px_local = 0.5 * math.sin(0.5 * time_temp)
            py_local = 0
            yaw_local = 0
            vx_local = 0.25 * math.cos(0.5 * time_temp)
            vy_local = 0
            vyaw_local = 0

            path_point_tmp = PathPoint(0, 0, 0, 0, 0, 0, 0)

            path_point_tmp.timeFromStart = i * time_seg
            path_point_tmp.x = (
                px_local * math.cos(self.yaw0)
                - py_local * math.sin(self.yaw0)
                + self.px0
            )
            path_point_tmp.y = (
                px_local * math.sin(self.yaw0)
                + py_local * math.cos(self.yaw0)
                + self.py0
            )
            path_point_tmp.yaw = yaw_local + self.yaw0
            path_point_tmp.vx = vx_local * math.cos(self.yaw0) - vy_local * math.sin(
                self.yaw0
            )
            path_point_tmp.vy = vx_local * math.sin(self.yaw0) + vy_local * math.cos(
                self.yaw0
            )
            path_point_tmp.vyaw = vyaw_local

            path.append(path_point_tmp)

            self.client.TrajectoryFollow(path)
    
    def Stretch(self,measurement=1,velocity=1):
        
        self.client.Stretch()
        print("Stretch !!!")
        time.sleep(1)  
        
    def Stand(self,measurement=1,velocity=1):
        self.client.RecoveryStand()
        print("RecoveryStand !!!")
        time.sleep(1)
       
    def Sit(self,measurement=1,velocity=1):   
        self.client.StandDown()
        print("Stand down !!!")
        time.sleep(1)
       
            
    def SpecialMotions(self,measurement=1,velocity=1):
        self.client.RecoveryStand()
        print("RecoveryStand !!!")
        time.sleep(1)
        
        self.client.Stretch()
        print("Sit !!!")
        time.sleep(1)  
        
        self.client.RecoveryStand()
        print("RecoveryStand !!!")
        time.sleep(1)
 
    def execute_function(self,func,measurement=1,velocity=1):
        self.current_function = func.__name__
        func(measurement)

    def start_execution(self,func,measurement,velocity=1):
        self.execution_thread = threading.Thread(target=self.execute_function, args=(func,measurement))
        self.execution_thread.start()

    def stop_execution(self):
        if self.current_function:
            self.execution_event.set()
            self.execution_thread.join()
            self.execution_event.clear()


class RemoteControl:
    def __init__(self, ip, port):
        self.address = f"http://{ip}:{port}/"
        self.dt = 0.01

    def send_cmd_get(self, cmd):
        url = self.address + cmd
        return requests.get(url)
    
    def send_cmd_post(self, cmd, data):
        url = self.address + cmd
        return requests.post(url, data)
    
    def play_sound(self, sound_file):
        url = os.path.join(self.address, 'play_sound', sound_file)
        return requests.get(url)
    
    def run_action(self, action):
        url = os.path.join(self.address, 'run_action', action)
        response = requests.get(url)
        if action == "Dance":
            time.sleep(14)
        time.sleep(2)
        return response
        
    def move_forward(self, measurement, velocity=0.5):
        url = os.path.join(self.address, 'move')
        elapsed_time = abs(measurement / 2) / velocity
        if measurement < 0:
            velocity = -velocity
        while True:
            for _ in range(int(elapsed_time / self.dt)):
                data = {"vx": velocity, "vy": 0, "vyaw": 0}
                response = requests.post(url, json=data)
                time.sleep(self.dt)
            self.run_action("Stop")
            break
        #return response

    def rotate(self, measurement, velocity=0.5):
        url = os.path.join(self.address, 'move')
        #elapsed_time = measurement / velocity
        measurement = measurement / 0.65
        elapsed_time = (abs(measurement) / 90)
        if measurement < 0:
            velocity = -velocity
        while True:
            for _ in range(int(elapsed_time / self.dt)):
                data = {"vx": 0, "vy": 0, "vyaw": velocity}
                response = requests.post(url, json=data)
                time.sleep(self.dt)
            # self.run_action("Stop")
            break
        return response

    def transpose(self, measurement, velocity=0.5):
        url = os.path.join(self.address, 'move')
        elapsed_time = abs(measurement) / velocity
        if measurement < 0:
            velocity = -velocity
        while True:
            for _ in range(int(elapsed_time / self.dt)):
                data = {"vx": 0, "vy": velocity, "vyaw": 0}
                response = requests.post(url, json=data)
                time.sleep(self.dt)
            self.run_action("Stop")
            break
        #return response

    def get_image(self):
        url = os.path.join(self.address, 'video_feed')
        response = requests.get(url, stream=True)
        boundary = b'\r\n'  # 假设边界是 'frame'
        buffer = b''

        for chunk in response.iter_content(chunk_size=1024):
            buffer += chunk
            while boundary in buffer:
                # 找到帧边界
                frame_end = buffer.find(boundary)
                frame_data = buffer[:frame_end]
                buffer = buffer[frame_end + len(boundary):]

                # 尝试解析帧
                try:
                    img = Image.open(BytesIO(frame_data))
                    img.save("obs.png")  # 显示帧
                    return img
                except Exception as e:
                    # print(f"Error processing frame: {e}")
                    pass
        return response
    
    def upload_file(self, file_path):
        url = os.path.join(self.address, 'upload')
        with open(file_path, 'rb') as file:
            files = {'file': (file_path, file)}  # 文件字段名是 'file'
            response = requests.post(url, files=files)
        return response


class AudioTextTransfer:
    def __init__(self, model_a2t, model_t2a, speaker, output_file):
        self.model_stt = whisper.load_model(model_a2t)
        self.model_tts = TTS(model_t2a)
        self.speaker = speaker
        self.output_file = output_file

    def tts(self, text):
        self.model_tts.tts_to_file(text=text, speaker_wav=self.speaker, file_path="output.wav")

    def stt(self, audio):
        result = self.model_stt.transcribe(audio)
        return result["text"]



# Robot state
robot_state = unitree_go_msg_dds__SportModeState_()
def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg


def record_audio(filename, duration):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    play_wav_file("/home/smbu/桌面/dog/HM-instruction-main/SFX/1.wav")
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
    time.sleep(1)


# BBY 的
# 10.24.5.36:8000
# 指令有三种格式    运动    描述    对话    0 1 2

def understanding_query(text, ip="10.24.6.194", port=5000):
    url = f"http://{ip}:{port}/process"
    
    text = ''' 
        你是一名操作机器狗的专家。机器狗的指令集包括以下指令：
        move_forward(distance)
        move_backward(distance)
        rotate_right(yaw)
        rotate_left(yaw)
        run_action(action)
        vqa(question)
        其中action包括["Lay Down","Stand Up"],distance和yaw是数字,question为字符串。
        请解析用户的请求，使用上述指令完成任务，前和左为正方向，后和右为负方向，以list格式输出指令序列，仅输出指令序列，例如：
        "前进1米，然后后退3米" -> [move_forward(1), move_backward(3)];
        "前进2米然后右转三十度" -> [move_forward(2), rotate_right(30)];
        "右转40度" -> [rotate_right(40)];
        "坐下" -> [run_action("Lay Down")];
        "你看到了什么？" -> [vqa("你看到了什么？")];
        "掉头" -> [rotate_left(180)];
        "跳舞" -> [run_action("Dance")];
        以下是用户请求：
        ''' + "\"" +text + "\"。"
    response = requests.post(url, data={'text': text})
    if response.status_code == 200:
        data = response.json()
        response_text = data.get("response", "")
        print(response_text)
        return response_text

def vqa(text, image_path, ip="10.24.6.194", port=5000):
    # 拼接 URL
    url = f"http://{ip}:{port}/process"
    
    with open(image_path, 'rb') as img_file:
        response = requests.post(url, files={'image': img_file}, data={'text': text})

    if response.status_code == 200:
        data = response.json()
        print(data)
        response_text = data.get("response", "")
        return response_text
    return response


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()  # 获取总帧数
        rate = wav_file.getframerate()  # 获取帧率（采样率）
        duration = frames / float(rate)  # 计算时长（秒）
        return duration + 2

def parsing_cmd(text):
    match = re.search(r'\[.*\]', text)
    if match:
        cmd_seq = match.group()
        cmd_seq = cmd_seq.strip("[]").split(", ")
        print(cmd_seq)
    else:
        print("未找到匹配的内容")
    return cmd_seq


def listen_and_wake():
    # 不用连线的模块
    ctl = RemoteControl('10.24.5.63', 5000)
    transfer = AudioTextTransfer("turbo", "tts_models/zh-CN/baker/tacotron2-DDC-GST", "1111.wav", "output.wav")
    image_path = '/home/smbu/桌面/temp.jpg'
    
    global wsParam
    
    #robot control init
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
        
    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(HighStateHandler, 10)
    time.sleep(1)

    test = SportModeTest()
    test.GetInitState(robot_state)

    print("Listening control instruction!!!")
    
    while True:
        time.sleep(2)

        print("Listening for wake word...")
        record_audio(WAVE_OUTPUT_FILENAME, RECORD_SECONDS)
        
        recognized_text = transfer.stt("recorded.wav")

        print(f"1: {recognized_text}")

        response = understanding_query(recognized_text)
        cmd_seq = parsing_cmd(response)
        for cmd in cmd_seq:
            cmd = cmd.split("(")
            cmd, arg = cmd[0], cmd[1].strip(")")
            print(cmd, arg)
            if cmd == "move_forward":
                ctl.move_forward(float(arg))
            elif cmd == "move_backward":
                ctl.move_forward(-float(arg))
            elif cmd == "rotate_left":
                ctl.rotate(float(arg))
            elif cmd == "rotate_right":
                ctl.rotate(-float(arg))
            elif cmd == "run_action":
                #ctl.run_action("Heart")
                ctl.run_action(arg.strip("\""))
            elif cmd == "transpose_left":
                ctl.transpose(float(arg))
            elif cmd == "transpose_right":
                ctl.transpose(-float(arg))
            elif cmd in ["vqa"]:
                img = ctl.get_image()
                answer = vqa(arg, "obs.png")
                print(answer)
                transfer.tts(answer)
                ctl.upload_file("output.wav")
                duration = get_wav_duration("output.wav")
                ctl.play_sound("output.wav")
                time.sleep(duration)
          


if __name__ == "__main__":
    listen_and_wake()
