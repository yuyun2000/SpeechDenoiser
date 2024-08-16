import numpy as np
import onnxruntime as ort
import torchaudio
import torch
from torch.nn import functional as F


#------------------------------init model
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
)
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL


ort_session = ort.InferenceSession(
    "./denoiser_model.onnx",
    sess_options,
    providers=["CPUExecutionProvider"],
)

input_names = ["input_frame", "states", "atten_lim_db"]
output_names = ["enhanced_audio_frame", "new_states", "lsnr"]




#------------------------------load wav
hop_size = 480
fft_size = 960
input_audio, sr = torchaudio.load('./inp.wav', channels_first=True)
input_audio = input_audio.mean(dim=0).unsqueeze(0)  # stereo to mono

input_audio = input_audio.squeeze(0)
orig_len = input_audio.shape[0]

# padding taken from
# https://github.com/Rikorose/DeepFilterNet/blob/fa926662facea33657c255fd1f3a083ddc696220/DeepFilterNet/df/enhance.py#L229
hop_size_divisible_padding_size = (hop_size - orig_len % hop_size) % hop_size

orig_len += hop_size_divisible_padding_size
input_audio = F.pad(
    input_audio, (0, fft_size + hop_size_divisible_padding_size)
)

chunked_audio = torch.split(input_audio, hop_size)


#-------------------------------inference
state = np.zeros(45304,dtype=np.float32)
atten_lim_db = np.zeros(1,dtype=np.float32)
enhanced = []
for frame in chunked_audio:
    out = ort_session.run(None,input_feed={"input_frame":frame.numpy(),"states":state,"atten_lim_db":atten_lim_db})
    enhanced.append(torch.tensor(out[0]))
    state = out[1]



#-------------------------------save
enhanced_audio = torch.cat(enhanced).unsqueeze(
    0
)  # [t] -> [1, t] typical mono format

d = fft_size - hop_size
enhanced_audio = enhanced_audio[:, d: orig_len + d]

torchaudio.save(
    "out.wav",
    enhanced_audio,
    sr,
    encoding="PCM_S",
    bits_per_sample=16,
)
