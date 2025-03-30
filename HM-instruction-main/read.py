from pydub import AudioSegment

def pcm_to_wav(pcm_file, wav_file, sample_width=2, channels=1, frame_rate=16000):
    # 读取PCM文件并创建音频对象
    audio = AudioSegment.from_raw(pcm_file, sample_width=sample_width, channels=channels, frame_rate=frame_rate)
    
    # 将音频对象导出为WAV文件
    audio.export(wav_file, format="wav")
    print(f"PCM file converted to {wav_file}")

# 示例
pcm_file = '/home/smbu/BBY/HM-instruction-main/demo1.pcm'  # PCM 文件路径
wav_file = '/home/smbu/BBY/HM-instruction-main/1111.wav'  # 转换后的 WAV 文件路径
pcm_to_wav(pcm_file, wav_file)
