
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
from En2Cn import get_result
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
from time import mktime
import _thread as thread
import os
import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
from scipy.io.wavfile import write
import math
import numpy as np

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

    def GetInitState(self, robot_state: SportModeState_):
        self.px0 = robot_state.position[0]
        self.py0 = robot_state.position[1]
        self.yaw0 = robot_state.imu_state.rpy[2]

    def StandUpDown(self):
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

    def VelocityMove(self):
        elapsed_time = 1
        for i in range(int(elapsed_time / self.dt)):
            self.client.Move(0.3, 0, 0)  # vx, vy vyaw
            time.sleep(self.dt)
        self.client.StopMove()

    def Move_forward(self):
        elapsed_time = 1
        for i in range(int(elapsed_time / self.dt)):
            self.client.Move(0.3, 0, 0)  # vx, vy vyaw
            time.sleep(self.dt)
        self.client.StopMove()

    def Move_backward(self):
        elapsed_time = 1
        for i in range(int(elapsed_time / self.dt)):
            self.client.Move(-0.3, 0, 0)  # vx, vy vyaw
            time.sleep(self.dt)
        self.client.StopMove()

    def Move_left(self):
        elapsed_time = 1
        for i in range(int(elapsed_time / self.dt)):
            self.client.Move(0, 3, 0)  # vx, vy vyaw
            time.sleep(self.dt)
        self.client.StopMove()
    
    def Move_right(self):
        elapsed_time = 1
        for i in range(int(elapsed_time / self.dt)):
            self.client.Move(0, -3, 0)  # vx, vy vyaw
            time.sleep(self.dt)
        self.client.StopMove()
    
    def Move_cycle(self):
        elapsed_time = 1
        for i in range(int(elapsed_time / self.dt)):
            self.client.Move(3, 0, 0.3)  # vx, vy vyaw
            time.sleep(self.dt)
        self.client.StopMove()


    def BalanceAttitude(self):
        self.client.Euler(0.1, 0.2, 0.3)  # roll, pitch, yaw
        self.client.BalanceStand()

    def TrajectoryFollow(self):
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

    def Stretch(self):
        
        self.client.Stretch()
        print("Stretch !!!")
        time.sleep(1)  
        
    def Stand(self):
        self.client.RecoveryStand()
        print("RecoveryStand !!!")
        time.sleep(1)
       
    def Sit(self):   
        self.client.StandDown()
        print("Stand down !!!")
        time.sleep(1)

    def SpecialMotions(self):
        self.client.RecoveryStand()
        print("RecoveryStand !!!")
        time.sleep(1)
        
        self.client.Stretch()
        print("Sit !!!")
        time.sleep(1)  
        
        self.client.RecoveryStand()
        print("RecoveryStand !!!")
        time.sleep(1)

# 翻译的工具
class get_result(object):
    def __init__(self,host,text):
        # 应用ID（到控制台获取）
        self.APPID = "d4e7815d"
        # 接口APISercet（到控制台机器翻译服务页面获取）
        self.Secret = "YWUyNjliMzZhY2VjYWY1ZThhZjgyMjRh"
        # 接口APIKey（到控制台机器翻译服务页面获取）
        self.APIKey= "6a7c2610ac213fd66e2217213ffef7f1"

        self.APPID = "ca2af9b3"
        # 接口APISercet（到控制台机器翻译服务页面获取）
        self.Secret = "Y2IzMTc4YzU4Mjk4MDkzMTRkMWNlM2I0"
        # 接口APIKey（到控制台机器翻译服务页面获取）
        self.APIKey= "1ec414e887cae842209fe2035a862832"
        
        
        # 以下为POST请求
        self.Host = host
        self.RequestUri = "/v2/its"
        # 设置url
        # print(host)
        self.url="https://"+host+self.RequestUri
        self.HttpMethod = "POST"
        self.Algorithm = "hmac-sha256"
        self.HttpProto = "HTTP/1.1"
        self.data = 0
        # 设置当前时间
        curTime_utc = datetime.datetime.utcnow()
        self.Date = self.httpdate(curTime_utc)
        # 设置业务参数
        # 语种列表参数值请参照接口文档：https://www.xfyun.cn/doc/nlp/xftrans/API.html
        self.Text=text
        self.BusinessArgs={
                "from": "en",
                "to": "cn",
            }

    def hashlib_256(self, res):
        m = hashlib.sha256(bytes(res.encode(encoding='utf-8'))).digest()
        result = "SHA-256=" + base64.b64encode(m).decode(encoding='utf-8')
        return result

    def httpdate(self, dt):
        """
        Return a string representation of a date according to RFC 1123
        (HTTP/1.1).

        The supplied date must be in UTC.

        """
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()]
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                 "Oct", "Nov", "Dec"][dt.month - 1]
        return "%s, %02d %s %04d %02d:%02d:%02d GMT" % (weekday, dt.day, month,
                                                        dt.year, dt.hour, dt.minute, dt.second)

    def generateSignature(self, digest):
        signatureStr = "host: " + self.Host + "\n"
        signatureStr += "date: " + self.Date + "\n"
        signatureStr += self.HttpMethod + " " + self.RequestUri \
                        + " " + self.HttpProto + "\n"
        signatureStr += "digest: " + digest
        signature = hmac.new(bytes(self.Secret.encode(encoding='utf-8')),
                             bytes(signatureStr.encode(encoding='utf-8')),
                             digestmod=hashlib.sha256).digest()
        result = base64.b64encode(signature)
        return result.decode(encoding='utf-8')

    def init_header(self, data):
        digest = self.hashlib_256(data)
        #print(digest)
        sign = self.generateSignature(digest)
        authHeader = 'api_key="%s", algorithm="%s", ' \
                     'headers="host date request-line digest", ' \
                     'signature="%s"' \
                     % (self.APIKey, self.Algorithm, sign)
        #print(authHeader)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Method": "POST",
            "Host": self.Host,
            "Date": self.Date,
            "Digest": digest,
            "Authorization": authHeader
        }
        return headers

    def get_body(self):
        content = str(base64.b64encode(self.Text.encode('utf-8')), 'utf-8')
        postdata = {
            "common": {"app_id": self.APPID},
            "business": self.BusinessArgs,
            "data": {
                "text": content,
            }
        }
        body = json.dumps(postdata)
        #print(body)
        return body

    def call_url(self):
        if self.APPID == '' or self.APIKey == '' or self.Secret == '':
            print('Appid 或APIKey 或APISecret 为空！请打开demo代码，填写相关信息。')
        else:
            code = 0
            body=self.get_body()
            headers=self.init_header(body)
            #print(self.url)
            response = requests.post(self.url, data=body, headers=headers,timeout=8)
            status_code = response.status_code
            #print(response.content)
            if status_code!=200:
                # 鉴权失败
                print("Http请求失败，状态码：" + str(status_code) + "，错误信息：" + response.text)
                print("请根据错误信息检查代码，接口文档：https://www.xfyun.cn/doc/nlp/xftrans/API.html")
            else:
                # 鉴权成功
                respData = json.loads(response.text)
                self.data=respData
                # 以下仅用于调试
                code = str(respData["code"])
                if code!='0':
                    print("请前往https://www.xfyun.cn/document/error-code?code=" + code + "查询解决办法")
class get_result_CtoE(object):
    def __init__(self,host,text):
        # 应用ID（到控制台获取）
        self.APPID = "d4e7815d"
        # 接口APISercet（到控制台机器翻译服务页面获取）
        self.Secret = "YWUyNjliMzZhY2VjYWY1ZThhZjgyMjRh"
        # 接口APIKey（到控制台机器翻译服务页面获取）
        self.APIKey= "6a7c2610ac213fd66e2217213ffef7f1"

        self.APPID = "ca2af9b3"
        # 接口APISercet（到控制台机器翻译服务页面获取）
        self.Secret = "Y2IzMTc4YzU4Mjk4MDkzMTRkMWNlM2I0"
        # 接口APIKey（到控制台机器翻译服务页面获取）
        self.APIKey= "1ec414e887cae842209fe2035a862832"
        
        
        # 以下为POST请求
        self.Host = host
        self.RequestUri = "/v2/its"
        # 设置url
        # print(host)
        self.url="https://"+host+self.RequestUri
        self.HttpMethod = "POST"
        self.Algorithm = "hmac-sha256"
        self.HttpProto = "HTTP/1.1"
        self.data = 0
        # 设置当前时间
        curTime_utc = datetime.datetime.utcnow()
        self.Date = self.httpdate(curTime_utc)
        # 设置业务参数
        # 语种列表参数值请参照接口文档：https://www.xfyun.cn/doc/nlp/xftrans/API.html
        self.Text=text
        self.BusinessArgs={
                "from": "cn",
                "to": "en",
            }

    def hashlib_256(self, res):
        m = hashlib.sha256(bytes(res.encode(encoding='utf-8'))).digest()
        result = "SHA-256=" + base64.b64encode(m).decode(encoding='utf-8')
        return result

    def httpdate(self, dt):
        """
        Return a string representation of a date according to RFC 1123
        (HTTP/1.1).

        The supplied date must be in UTC.

        """
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()]
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                 "Oct", "Nov", "Dec"][dt.month - 1]
        return "%s, %02d %s %04d %02d:%02d:%02d GMT" % (weekday, dt.day, month,
                                                        dt.year, dt.hour, dt.minute, dt.second)

    def generateSignature(self, digest):
        signatureStr = "host: " + self.Host + "\n"
        signatureStr += "date: " + self.Date + "\n"
        signatureStr += self.HttpMethod + " " + self.RequestUri \
                        + " " + self.HttpProto + "\n"
        signatureStr += "digest: " + digest
        signature = hmac.new(bytes(self.Secret.encode(encoding='utf-8')),
                             bytes(signatureStr.encode(encoding='utf-8')),
                             digestmod=hashlib.sha256).digest()
        result = base64.b64encode(signature)
        return result.decode(encoding='utf-8')

    def init_header(self, data):
        digest = self.hashlib_256(data)
        #print(digest)
        sign = self.generateSignature(digest)
        authHeader = 'api_key="%s", algorithm="%s", ' \
                     'headers="host date request-line digest", ' \
                     'signature="%s"' \
                     % (self.APIKey, self.Algorithm, sign)
        #print(authHeader)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Method": "POST",
            "Host": self.Host,
            "Date": self.Date,
            "Digest": digest,
            "Authorization": authHeader
        }
        return headers

    def get_body(self):
        content = str(base64.b64encode(self.Text.encode('utf-8')), 'utf-8')
        postdata = {
            "common": {"app_id": self.APPID},
            "business": self.BusinessArgs,
            "data": {
                "text": content,
            }
        }
        body = json.dumps(postdata)
        #print(body)
        return body

    def call_url(self):
        if self.APPID == '' or self.APIKey == '' or self.Secret == '':
            print('Appid 或APIKey 或APISecret 为空！请打开demo代码，填写相关信息。')
        else:
            code = 0
            body=self.get_body()
            headers=self.init_header(body)
            #print(self.url)
            response = requests.post(self.url, data=body, headers=headers,timeout=8)
            status_code = response.status_code
            #print(response.content)
            if status_code!=200:
                # 鉴权失败
                print("Http请求失败，状态码：" + str(status_code) + "，错误信息：" + response.text)
                print("请根据错误信息检查代码，接口文档：https://www.xfyun.cn/doc/nlp/xftrans/API.html")
            else:
                # 鉴权成功
                respData = json.loads(response.text)
                self.data=respData
                # 以下仅用于调试
                code = str(respData["code"])
                if code!='0':
                    print("请前往https://www.xfyun.cn/document/error-code?code=" + code + "查询解决办法")


# Robot state
robot_state = unitree_go_msg_dds__SportModeState_()
def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg



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

def pcm_to_wav(pcm_file_path,wav_file_path):
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

def listen_and_wake():
    image_path = '/home/smbu/Desktop/temp.jpg'
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
                
                host = "itrans.xfyun.cn"
                gClass=get_result(host,result_EN)
                gClass.call_url()
                result_cn = gClass.data['data']['result']['trans_result']['dst']
                print(result_cn)
                
                wsParam1 = Ws_Param(APPID='3676e2d5', APISecret='MWVhMTUwZWRiZDhmNzhlNTlhNDdjOWM4',
                       APIKey='ec07d0b6e326d00e50de170f34c43346',
                       Text=result_cn)
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

        # 我的代码在下面改的
        if "描述" in recognized_text or "看到" in recognized_text:
            print("-----Start look forward------")
            text = '描述一下图片'
            
            host = "itrans.xfyun.cn"
            gClass_CtoE = get_result_CtoE(host,text)
            gClass_CtoE.call_url()
            print(gClass_CtoE.data['data']['result']['trans_result']['dst'])
            
            result_text = gClass_CtoE.data['data']['result']['trans_result']['dst']
            
            t = send_image_and_text(image_path, result_text)
            
            gClass=get_result(host,t)           # 转成汉字
            gClass.call_url()
            result_t = gClass.data['data']['result']['trans_result']['dst']
            # 语音
            wsParam2 = Ws_Param(APPID='d4e7815d', APISecret='YWUyNjliMzZhY2VjYWY1ZThhZjgyMjRh',APIKey='6a7c2610ac213fd66e2217213ffef7f1',Text=result_t)
            wsParam = wsParam2
            websocket.enableTrace(False)
            wsUrl = wsParam.create_url()
            ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
            ws.on_open = on_open
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

            # 转成 wav文件
            pcm_to_wav("demo.pcm",'output.wav')

            wav_file_path = 'output.wav'
            play_wav_file(wav_file_path)


        # ========================================================================================
        if "往前" in recognized_text or "前" in recognized_text:
            print("-----Start moving forward------")
            test.Move_forward()
            print("-----Stop moving forward------")
        if "往后" in recognized_text or "后" in recognized_text:
            print("-----Start moving backward------")
            test.Move_backward()
            print("-----Stop moving backward------")
        if "往左" in recognized_text or "左" in recognized_text:
            print("-----Start moving left------")
            test.Move_left()
            print("-----Stop moving left------")  
        if "往右" in recognized_text or "右" in recognized_text:
            print("-----Start moving right------")
            test.Move_right()
            print("-----Stop moving right------")    
        if "转圈" in recognized_text or "转" in recognized_text:
            print("-----Start moving circle------")
            test.Move_cycle()
            print("-----Stop moving circle------")    

        if"伸懒腰" in recognized_text or "懒" in recognized_text:
            print("-----Start Stretching------")
            test.Stretch()
            print("-----Stop Stretching------")


        if "坐下" in recognized_text or "坐" in recognized_text:
            print("-----Start sit------")
            test.Sit()
            print("-----Stop sit------")   
           
        if "站立" in recognized_text or "站" in recognized_text:
            print("-----Start stand------")
            test.Stand()
            print("-----Stop stand------")   
            
            
if __name__ == "__main__":
    listen_and_wake()

