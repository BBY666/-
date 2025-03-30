import pyaudio
import wave
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
from ipdb import set_trace
import websocket
import pyaudio
import wave
import datetime
import hashlib
import base64
import hmac
from test_audi import main_fun
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os

# Recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "recorded.wav"
TTS_OUTPUT_FILENAME = "result_audio.mp3"
wsParam = 0
# API parameters
API_KEY = "6a7c2610ac213fd66e2217213ffef7f1"
API_SECRET = "YWUyNjliMzZhY2VjYWY1ZThhZjgyMjRh"
ASR_API_URL = "wss://iat-api.xfyun.cn/v2/iat"
TTS_API_URL = "wss://tts-api.xfyun.cn/v2/tts"
APP_ID = "d4e7815d"
lfasr_host = 'https://raasr.xfyun.cn/v2/api'
api_upload = '/upload'
api_get_result = '/getResult'

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": "xiaoyan", "tte": "utf8"}
        self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")}
        #使用小语种须使用以下方式，此处的unicode指的是 utf16小端的编码方式，即"UTF-16LE"”
        #self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-16')), "UTF8")}

    # 生成url
    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        # url='https://api-dx.xf-yun.com/v1/private/dts_create'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url

def on_message(ws, message):
    try:
        message =json.loads(message)
        code = message["code"]
        sid = message["sid"]
        audio = message["data"]["audio"]
        audio = base64.b64decode(audio)
        status = message["data"]["status"]
        print(message)
        if status == 2:
            print("ws is closed")
            ws.close()
        if code != 0:
            errMsg = message["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
        else:

            with open('./demo.pcm', 'ab') as f:
                f.write(audio)

    except Exception as e:
        print("receive msg,but parse exception:", e)



# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws,A1,A2):
    print("### closed ###",A1,A2)

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
# 收到websocket连接建立的处理
def on_open(ws):
    def run(*args):
        d = {"common": wsParam.CommonArgs,
             "business": wsParam.BusinessArgs,
             "data": wsParam.Data,
             }
        d = json.dumps(d)
        print("------>开始发送文本数据")
        ws.send(d)
        if os.path.exists('./demo.pcm'):
            os.remove('./demo.pcm')

    thread.start_new_thread(run, ())


def send_image_and_text(image_path, text, server_host='10.50.0.37', server_port=8010):
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
class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        return signa

    def upload(self):
        print("Uploading file...")
        upload_file_path = self.upload_file_path
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)

        param_dict = {
            'appId': self.appid,
            'signa': self.signa,
            'ts': self.ts,
            "fileSize": file_len,
            "fileName": file_name,
            "duration": "200"
        }

        data = open(upload_file_path, 'rb').read(file_len)
        response = requests.post(url=lfasr_host + api_upload + "?" + urllib.parse.urlencode(param_dict),
                                 headers={"Content-type": "application/json"}, data=data)
        result = json.loads(response.text)
        print("Upload response:", result)
        return result

    def get_result(self):
        uploadresp = self.upload()
        orderId = uploadresp['content']['orderId']
        param_dict = {
            'appId': self.appid,
            'signa': self.signa,
            'ts': self.ts,
            'orderId': orderId,
            'resultType': "transfer,predict"
        }
        print("Fetching result...")
        status = 3
        while status == 3:
            response = requests.post(url=lfasr_host + api_get_result + "?" + urllib.parse.urlencode(param_dict),
                                     headers={"Content-type": "application/json"})
            result = json.loads(response.text)
            status = result['content']['orderInfo']['status']
            if status == 4:
                break
            time.sleep(5)
        set_trace()
        print("Result:", result)
        return result


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


def listen_and_wake():
    image_path = '/home/smbu/Desktop/temp.jpg'
    global wsParam
    while True:
        play_wav_file("1.wav")
        print("Listening for wake word...")
        record_audio(WAVE_OUTPUT_FILENAME, RECORD_SECONDS)
        recognized_text = main_fun(WAVE_OUTPUT_FILENAME)
        print(f"1: {recognized_text}")

        if "你好" in recognized_text:
            print("----------Wake word detected, starting recording...")
            while True:
                play_wav_file("2.wav")
                record_audio(WAVE_OUTPUT_FILENAME, RECORD_SECONDS)
                recognized_text = main_fun(WAVE_OUTPUT_FILENAME)
              
                print(f"2: {recognized_text}")
                result_EN = send_image_and_text(image_path, recognized_text)
                print(result_EN)
                wsParam1 = Ws_Param(APPID='3676e2d5', APISecret='MWVhMTUwZWRiZDhmNzhlNTlhNDdjOWM4',
                       APIKey='ec07d0b6e326d00e50de170f34c43346',
                       Text=result_EN)
                wsParam = wsParam1
                websocket.enableTrace(False)
                wsUrl = wsParam1.create_url()
                ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
                ws.on_open = on_open
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
                play_pcm('demo.pcm')
                # Here you can add the text-to-speech and other processing as needed
                # xunfei_tts(recognized_text, TTS_OUTPUT_FILENAME)
                # audio_player(TTS_OUTPUT_FILENAME)


if __name__ == "__main__":
    listen_and_wake()

