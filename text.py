import gradio as gr
from speech_paraformer.demo_online import asr

title = "语音转文本"

# def save_tuple_as_wav(filename, data, sample_width, sample_rate, num_channels):
#     wav_file = wave.open(filename, 'w')
#     wav_file.setparams((num_channels, sample_width, sample_rate, len(data), 'NONE', 'not compressed'))
    
#     # 将tuple数据转换为二进制字符串
#     if sample_width == 1:
#         fmt = "%dB" % len(data)
#         data = struct.pack(fmt, *data)
#     elif sample_width == 2:
#         scaled_data = [max(min(int(d * 32767), 32767), -32768) for d in data]
#         fmt = "%dh" % len(scaled_data)
#         data = struct.pack(fmt, *scaled_data)
#     else:
#         raise ValueError("Unsupported sample width")
    
#     # 写入二进制数据到wav文件
#     wav_file.writeframes(data)
#     wav_file.close()

# # 示例数据
# data = (0, 1, 2, 3, 4, 5)
# sample_width = 2  # 2字节表示一个样本
# sample_rate = 44100  # 采样率44100Hz
# num_channels = 1  # 单声道


def generateAudio(audio):
    # path = audio
    # list = path.split(sep='\\')
    # list[2] = "LiuJiacheng"
    # ans = ""
    # for i in range(len(list)):
    #     ans += list[i]
    #     if (i+1 == len(list)):
    #         continue
    #     else:
    #         ans += '/'

    return asr(audio)

app = gr.Interface(
    fn=generateAudio, 
    inputs=gr.Audio(source="microphone", type="filepath"),  
    outputs="text", 
    title=title
    )

app.launch()
