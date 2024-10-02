from Chinese_speech_recognition.model.models import speech_model
import torchaudio as ta
import numpy as np
import torch

def get_fu(path_):
    _wavform, _ = ta.load( path_ )
    _feature = ta.compliance.kaldi.fbank(_wavform, num_mel_bins=40) 
    _mean = torch.mean(_feature)
    _std = torch.std(_feature)
    _T_feature =  (_feature - _mean) / _std
    inst_T = _T_feature.unsqueeze(0)
    return inst_T

def run(path_, model_lo, num_wor):
    inst_T = get_fu( path_ )
    log_  = model_lo( inst_T )
    _pre_ = log_.transpose(0,1).detach().numpy()[0]
    liuiu = [dd for dd in _pre_.argmax(-1) if dd != 0]
    str_end = ''.join([ num_wor[dd] for dd in liuiu ])
    return str_end

def asr():
    model_lo = speech_model()
    device_ = torch.device('cpu')
    model_lo.load_state_dict(torch.load('Chinese_speech_recognition/models/sp_model.pt' , map_location=device_))
    model_lo.eval()

    num_wor = np.load('Chinese_speech_recognition/models/dic.dic.npy').item()
    path_ = 'Chinese_speech_recognition/speech1.wav'
    result_ = run(path_, model_lo, num_wor)
    print ( '识别结果是: ' ,  result_ )
    return result_
