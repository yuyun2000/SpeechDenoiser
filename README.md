# SpeechDenoiser
SpeechDenoiser: Real-Time Speech Denoising with ONNX  Welcome to SpeechDenoiser, a simple and effective solution for real-time speech denoising using an ONNX model. This repository contains everything you need to get started with enhancing audio quality by reducing noise, making it perfect for improving voice recordings and live communication.


---
支持48khz和16khz，其中48khz模型使用deepfilternet3；16khz模型使用gtcrn。他们都是流式的，但是48k模型的实时要求更高，我认为骁龙865足以实时推理，16k模型实时要求低一些，我觉得树莓派4B就有希望，但是48k直接输入的就是音频，而16k输入的stft后的特征，端侧实现可能麻烦一点

---

Supports 48kHz and 16kHz. The 48kHz model uses DeepFilterNet3, while the 16kHz model uses GTCRN. Both models are streaming, but the 48kHz model has higher real-time requirements. I believe that a Snapdragon 865 should be sufficient for real-time inference. The 16kHz model has lower real-time requirements, and I think the Raspberry Pi 4B might be capable. However, the 48kHz model takes raw audio as input, while the 16kHz model takes STFT features as input, which could make implementation on edge devices a bit more challenging.

---
感谢：
https://github.com/Xiaobin-Rong/gtcrn

https://github.com/Rikorose/DeepFilterNet

https://github.com/grazder/DeepFilterNet/tree/torchDF-changes

```
@inproceedings{schroeter2022deepfilternet,
  title={{DeepFilterNet}: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering}, 
  author = {Schröter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
  booktitle={ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
  organization={IEEE}
}