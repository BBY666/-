
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
from prompt_example import get_instructions
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
import nltk
from scipy.io.wavfile import write
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
import numpy as np
import utils.speech_utils as tool
#import pyttsx3


# Recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "recorded.wav"
TTS_OUTPUT_FILENAME = "result_audio.mp3"
wsParam = 0
# API parameterstext_result= get_instructions("前进五十米再右转六十度，再前进二十米")
# print(text_result)
API_KEY = "6a7c2610ac213fd66e2217213ffef7f1"
API_SECRET = "YWUyNjliMzZhY2VjYWY1ZThhZjgyMjRh"
ASR_API_URL = "wss://iat-api.xfyun.cn/v2/iat"
TTS_API_URL = "wss://tts-api.xfyun.cn/v2/tts"
APP_ID = "d4e7815d"
lfasr_host = 'https://raasr.xfyun.cn/v2/api'
api_upload = '/upload'
api_get_result = '/getResult'






#def tts(text='testing'):
#    engine=pyttsx3.init()
#    engine.say(text)
#    engine.runAndWait()

#robot control



def listen_and_wake():
#     image_path = '/home/smbu/Desktop/temp.jpg'
#     global wsParam
    
    #robot control init
    # 初始化信道，if代表了是否传入了信道参数，如果传入了，使用那个信道进行通信，如果没有传入，代码会自动选择一个合适的配置
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
    
    
    
    ##############################
    while True:
        tool.play_wav_file("./SFX/1.wav")
        print("Listening for wake word...")
        tool.record_audio(WAVE_OUTPUT_FILENAME, RECORD_SECONDS)
        recognized_text = tool.stt(WAVE_OUTPUT_FILENAME)
        print(f"1: {recognized_text}")
        client = VideoClient()
        client.SetTimeout(3.0)
        client.Init()

        #regin 唤醒代码，有问题，需重新考虑逻辑问题

        # if "你好" in recognized_text:
        #     print("----------Wake word detected, starting recording...")
        #     while True:


        #         print("##################GetImageSample###################")
        #         code, data = client.GetImageSample()

        #         if code != 0:
        #             print("get image sample error. code:", code)
        #         else:
        #             imageName = "/home/smbu/Desktop/temp.jpg"
        #         imageName = "/home/smbu/Desktop/temp.jpg"
        #         print("ImageName:", imageName)

        #         with open(imageName, "+wb") as f:
        #             f.write(bytes(data))

        #         time.sleep(1)
        #         play_wav_file("2.wav")

        #         record_audio(WAVE_OUTPUT_FILENAME, RECORD_SECONDS)
        #         recognized_text = main_fun(WAVE_OUTPUT_FILENAME)
              
        #         print(f"2: {recognized_text}")
        #         result_EN = send_image_and_text(image_path, recognized_text)
        #         print(result_EN)
                
        #         host = "itrans.xfyun.cn"
        #         gClass=get_result(host,result_EN)
        #         gClass.call_url()
        #         result_cn = gClass.data['data']['result']['trans_result']['dst']
        #         print(result_cn)
                
        #         wsParam1 = Ws_Param(APPID='3676e2d5', APISecret='MWVhMTUwZWRiZDhmNzhlNTlhNDdjOWM4',
        #                APIKey='ec07d0b6e326d00e50de170f34c43346',
        #                Text=result_cn)
        #         wsParam = wsParam1
        #         websocket.enableTrace(False)
        #         wsUrl = wsParam1.create_url()
        #         ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
        #         ws.on_open = on_open
        #         ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        #         play_pcm('demo.pcm')
        #         # Here you can add the text-to-speech and other processing as needed
        #         # xunfei_tts(recognized_text, TTS_OUTPUT_FILENAME)
        #         # audio_player(TTS_OUTPUT_FILENAME)
        
        #         # 我的代码在下面改的
        #endregion

        """
        描述看到了什么
        """
        # if "描述" in recognized_text or "看到" in recognized_text:
        #     print("-----Start look forward------")
        #     text = '描述一下图片'
            
        #     host = "itrans.xfyun.cn"
        #     gClass_CtoE = get_result_CtoE(host,text)
        #     gClass_CtoE.call_url()
        #     print(gClass_CtoE.data['data']['result']['trans_result']['dst'])
            
        #     result_text = gClass_CtoE.data['data']['result']['trans_result']['dst']
            
        #     # paizhao

        #     print("##################GetImageSample###################")
        #     code, data = client.GetImageSample()

        #     if code != 0:
        #         print("get image sample error. code:", code)
        #     else:
        #         imageName = "/home/smbu/Desktop/temp.jpg"
        #     imageName = "/home/smbu/Desktop/temp.jpg"
        #     print("ImageName:", imageName)

        #     with open(imageName, "+wb") as f:
        #         f.write(bytes(data))

        #     t = send_image_and_text(imageName, result_text)
            
        #     gClass=get_result(host,t)           # 转成汉字
        #     gClass.call_url()
        #     result_t = gClass.data['data']['result']['trans_result']['dst']
        #     # 语音
        #     wsParam2 = Ws_Param(APPID='d4e7815d', APISecret='YWUyNjliMzZhY2VjYWY1ZThhZjgyMjRh',APIKey='6a7c2610ac213fd66e2217213ffef7f1',Text=result_t)
        #     wsParam = wsParam2
        #     websocket.enableTrace(False)
        #     wsUrl = wsParam.create_url()
        #     ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
        #     ws.on_open = on_open
        #     ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

        #     # 转成 wav文件
        #     pcm_to_wav("demo.pcm",'output.wav')

        #     wav_file_path = 'output.wav'
        #     play_wav_file(wav_file_path)


        #=====================================================================



        # 基于规则获取指令
        # command=map_sentence_to_command(recognized_text)

        # 基于大模型获取指令
        # text_from_llm = tool.get_key_instructions(recognized_text)
        print(recognized_text)
        ############################################
        command_from_llm=get_instructions(recognized_text)
        
        command=command_from_llm['action']
        print(command)
        measurement=int(command_from_llm['degrees'])
        # command_velocity=['velocity']

        if command == "未识别的指令" or "无":
            # text_from_llm = command
            respond_text = f'未识别的指令,请重新输入指令.'
        # text_from_llm = '前进'
        else:
            respond_text = f'是要执行{command}指令吗？'
        respond_wav = tool.tts(respond_text, 'speak_by_dog.wav')

        play_wav_file(respond_wav)
        print("Listening for respond word...")
        record_audio(WAVE_OUTPUT_FILENAME, RECORD_SECONDS)
        recognized_text = main_fun(WAVE_OUTPUT_FILENAME)
        print(f"1: {recognized_text}")

        # measurement=1
        command_velocity=1
        keywords = ["是的", "对", "正确"]
        #tts(command)
        if any(keyword in recognized_text for keyword in keywords): # if 
            if command == "停止":
                print("-----Stop moving-----")
                test.stop_execution()
                test.start_execution(test.StopMove,measurement,command_velocity)
                print("-----Stop moving-----")
            if command == "前进":
                print("-----Start moving forward------")
                test.stop_execution()
                test.start_execution(test.Move_forward,measurement,command_velocity)
                print("-----Stop moving forward------")
            if command == "后退":
                print("-----Start moving backward------")
                test.stop_execution()
                test.start_execution(test.Move_backward,measurement,command_velocity)
                print("-----Stop moving backward------")
            if command == "向左":
                print("-----Start moving left------")
                test.stop_execution()
                test.start_execution(test.Move_left,measurement,command_velocity)

                # test.Move_left()
                print("-----Stop moving left------")  
            if command == "向右":
                
                print("-----Start moving right------")
                test.stop_execution()
                test.start_execution(test.Move_right,measurement,command_velocity)

                print("-----Stop moving right------")    
            if command == "右转":
                print("-----Start moving right circle------")
                test.stop_execution()
                test.start_execution(test.Move_cycle_right,measurement,command_velocity)
                print("-----Stop moving circle------")    

            if command == "左转":
                print("-----Start moving left circle------")
                test.stop_execution()
                test.start_execution(test.Move_cycle_left,measurement,command_velocity)
                print("-----Stop moving circle------")




            if command == "伸懒腰":
                print("-----Start Stretching------")
                test.Stretch()
                print("-----Stop Stretching------")


            if command == "坐下":
                print("-----Start sit------")
                test.Sit()
                print("-----Stop sit------")   
            
            if command == "站起来":
                print("-----Start stand------")
                test.Stand()
                print("-----Stop stand------")   
            
            
if __name__ == "__main__":
    listen_and_wake()

#v请重新输入指令