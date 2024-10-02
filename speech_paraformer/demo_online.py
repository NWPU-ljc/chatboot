import os
import logging
import torch
import soundfile

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

def asr(audio_path):
    logger = get_logger(log_level=logging.CRITICAL)
    logger.setLevel(logging.CRITICAL)

    os.environ["MODELSCOPE_CACHE"] = "./"
    # inference_pipeline = pipeline(
    #     task=Tasks.auto_speech_recognition,
    #     model='speech_paraformer/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
    #     model_revision='v1.0.6',
    #     update_model=False,
    #     mode="paraformer_streaming"
    # )

    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950',
        model_revision='v3.0.0'
    )

    # model_dir = os.path.join(os.environ["MODELSCOPE_CACHE"], "speech_paraformer/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
    # # speech, sample_rate = soundfile.read(os.path.join(model_dir, "example/asr_example.wav"))
    # speech, sample_rate = soundfile.read(audio_path)
    # speech_length = speech.shape[0]

    # sample_offset = 0
    # chunk_size = [5, 10, 5] #[5, 10, 5] 600ms, [8, 8, 4] 480ms
    # stride_size =  chunk_size[1] * 960
    # param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size}
    final_result = ""

    print("start asr!!!")
    # for sample_offset in range(0, speech_length, min(stride_size, speech_length - sample_offset)):
    #     if sample_offset + stride_size >= speech_length - 1:
    #         stride_size = speech_length - sample_offset
    #         param_dict["is_final"] = True

    #     print(sample_offset)
    #     print(stride_size)
    #     rec_result = inference_pipeline(audio_in=speech[sample_offset: sample_offset + stride_size],
    #                                     param_dict=param_dict)
    #     if len(rec_result) != 0:
    #         final_result += rec_result['text']
    #         print(rec_result)
    final_result = inference_pipeline(audio_in=audio_path)
    # print(str(final_result['text']).replace(" ", ""))
    # return str(final_result['text']).replace(" ", "")
    print(str(final_result['text'][0]))
    return str(final_result['text'][0])

# asr("speech_paraformer/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/example/asr_example.wav")
