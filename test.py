    File "/mnt/diskb/penglong/anaconda3/envs/war/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1050, in cuda
    return self._apply(lambda t: t.cuda(device))
  File "/mnt/diskb/penglong/anaconda3/envs/war/lib/python3.9/site-packages/torch/nn/modules/module.py", line 900, in _apply
    module._apply(fn)
  File "/mnt/diskb/penglong/anaconda3/envs/war/lib/python3.9/site-packages/torch/nn/modules/module.py", line 900, in _apply
    module._apply(fn)
  File "/mnt/diskb/penglong/anaconda3/envs/war/lib/python3.9/site-packages/torch/nn/modules/module.py", line 927, in _apply
    param_applied = fn(param)
  File "/mnt/diskb/penglong/anaconda3/envs/war/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1050, in <lambda>
    return self._apply(lambda t: t.cuda(device))
  File "/mnt/diskb/penglong/anaconda3/envs/war/lib/python3.9/site-packages/torch/cuda/__init__.py", line 319, in _lazy_init
    torch._C._cuda_init()
RuntimeError: The NVIDIA driver on your system is too old (found version 11020). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver.
