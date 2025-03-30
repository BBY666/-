import speech_utils as tool

text = '是的！'

received_wav = tool.tts(text, 'feedback.wav')

# print(received_wav)