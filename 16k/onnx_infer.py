import torch
import soundfile as sf
from librosa import istft
import onnxruntime
import numpy as np


#------------------load wav
x = torch.from_numpy(sf.read('./inp_16k.wav', dtype='float32')[0])
x = torch.stft(x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)[None]


#------------------init model & state
session = onnxruntime.InferenceSession('gtcrn_simple.onnx', None,
                                       providers=['CPUExecutionProvider'])
conv_cache = np.zeros([2, 1, 16, 16, 33], dtype="float32")
tra_cache = np.zeros([2, 3, 1, 1, 16], dtype="float32")
inter_cache = np.zeros([2, 1, 33, 16], dtype="float32")

T_list = []
outputs = []

#------------------infer
inputs = x.numpy()
for i in range(inputs.shape[-2]):

    out_i, conv_cache, tra_cache, inter_cache \
        = session.run([], {'mix': inputs[..., i:i + 1, :],
                           'conv_cache': conv_cache,
                           'tra_cache': tra_cache,
                           'inter_cache': inter_cache})


    outputs.append(out_i)

outputs = np.concatenate(outputs, axis=2)
enhanced = istft(outputs[..., 0] + 1j * outputs[..., 1], n_fft=512, hop_length=256, win_length=512,
                 window=np.hanning(512) ** 0.5)
sf.write('./out_16k.wav', enhanced.squeeze(), 16000)