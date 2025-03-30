import speech_utils as tool

# 机器狗接收到语音
received_wav_from_dog = 'received_audio.wav'

# 对接收到的语音进行修改 指定转化后的wav文件名
fixed_wav = 'fixed_1.wav'
tool.convert_to_16k(received_wav_from_dog, fixed_wav)

# 将语音转化为文字
received_wav_text = tool.stt(fixed_wav)

# 将文字输入给大语言模型进行精简，归纳出我们想要的指令
text_from_llm = '往前走'

respond_text = f'是要执行{text_from_llm}指令吗？'

respond_wav = tool.tts(respond_text, 'speak_by_dog.wav')

############# 
# 机器狗产生质询：调用接口输出语音，然后开始听
# ----

# ----
received_feedback_wav = 'feedback.wav'
# 语音转化为文字
fixed_wav = 'fixed_2.wav'
tool.convert_to_16k(received_feedback_wav, fixed_wav)
received_feedback_text = tool.stt(fixed_wav)

keywords = ["是的", "对", "是", "正确"]

if any(keyword in received_feedback_text for keyword in keywords):
    print("Understanding right!")
    # 执行指令
else:
    print("Understanding wrong！")




