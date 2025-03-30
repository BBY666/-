import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Micorphone()

wakeup_words = ['你好','你在吗']

def on_wake():
    print("听到了唤醒词！")

def recognize_speech():
    with mic as source:
        print("开始录音。。。")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        test = r.recognize_google(audio, language='zh-CN')
        print('识别结果：', text)
        if text in wakeup_words:
            on_wake()
    
    except sr.UnknownValueError:
        print("听不清请再说一遍。")
    except sr.RequestError as e:
        print('请求出错：{0}'.format(e))

with mic as source:
    r.adjust_for_ambient_noise(source)
print('语音唤醒已启动！')
while True:
    recognize_speech()