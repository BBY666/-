import pyaudio
import sys
from pydub import AudioSegment

def audio_player(path):
    # 使用pydub加载MP3文件
    audio = AudioSegment.from_mp3(path)

    # 将音频转换为PCM格式数据
    audio_data = audio.raw_data
    channels = audio.channels
    sample_width = audio.sample_width
    frame_rate = audio.frame_rate

    # 实例化一个PyAudio对象，并初始化PortAudio系统资源
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(channels=channels,
                    format=p.get_format_from_width(sample_width),
                    rate=frame_rate,
                    output=True)

    # 播放音频数据
    stream.write(audio_data)

    # 关闭音频流，释放PortAudio系统资源
    stream.close()
    p.terminate()

    # 退出程序
    sys.exit(0)

