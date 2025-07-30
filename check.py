import torch, platform, sys
print("PyTorch", torch.__version__)
print("Python ", platform.python_version())
print("CUDA   available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))