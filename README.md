# SpeechDenoiser
SpeechDenoiser: Real-Time Speech Denoising with ONNX  Welcome to SpeechDenoiser, a simple and effective solution for real-time speech denoising using an ONNX model. This repository contains everything you need to get started with enhancing audio quality by reducing noise, making it perfect for improving voice recordings and live communication.


---
仅支持48kHz的音频；推理代码十分粗糙，但是它可以工作！如果你有经验，你会发现它是一个流式模型，所以可以进行实时降噪！经过我的测试，在高通骁龙865上可以无压力的实时降噪！

---
Only supports 48kHz audio; the inference code is quite rough, but it works! If you have experience, you'll notice that it is a streaming model, so it can perform real-time denoising! Based on my tests, it can handle real-time denoising effortlessly on a Qualcomm Snapdragon 865!


---
感谢：

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